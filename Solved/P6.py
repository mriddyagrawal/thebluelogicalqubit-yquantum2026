import time
import warnings
import sys
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CommutativeCancellation,
    InverseCancellation,
)
from qiskit.circuit.library import CZGate
import bluequbit
from pathlib import Path
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    FullPeepholeOptimise,
    RemoveRedundancies,
    CommuteThroughMultis,
    AutoRebase,
    SequencePass,
)
from pytket.circuit import OpType

sys.path.append(os.path.abspath(".."))
from modules import *

os.environ["BLUEQUBIT_DEQART_INTERNAL_DISABLE_STRICT_VALIDATIONS"] = "1"


def load_simple_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_simple_env(Path.cwd() / ".env")
load_simple_env(Path.cwd().parent / ".env")
load_simple_env(Path.cwd().parent.parent / ".env")

api_token = os.getenv("bluequbitapi") or os.getenv("BLUEQUBIT_API_TOKEN")
if not api_token:
    raise ValueError("Missing BlueQubit API token. Set bluequbitapi in .env")

os.environ["BLUEQUBIT_API_TOKEN"] = api_token

warnings.filterwarnings("ignore", category=UserWarning)

bq = bluequbit.init(api_token=api_token)

# ── Load & inspect P6 circuit ──
qc = load_qasm('P6_low_hill.qasm')

print(f"Original circuit:")
print(f"  Gates: {qc.count_ops()}")
print(f"  Depth: {qc.depth()}")
print(f"  Total gates: {sum(qc.count_ops().values())}")

# ── Step 1: Strip barriers ──
qc.data = [inst for inst in qc.data if inst.operation.name != 'barrier']


# ── Step 2: Custom near-identity U gate removal ──
def remove_near_identity_u(circuit, threshold=0.05):
    """Remove U gates that are close to identity (small rotation angles)."""
    new_data = []
    removed = 0
    for inst in circuit.data:
        if inst.operation.name == 'u':
            theta, phi, lam = inst.operation.params
            if abs(theta) < threshold:
                phase = (phi + lam) % (2 * np.pi)
                if phase < threshold or abs(phase - 2 * np.pi) < threshold:
                    removed += 1
                    continue
        new_data.append(inst)
    circuit.data = new_data
    return removed


removed = remove_near_identity_u(qc, threshold=0.05)
print(f"\nRemoved {removed} near-identity U gates")
print(f"  Total gates: {sum(qc.count_ops().values())}")

# ── Step 3: Aggressive Qiskit transpilation (sweep approx_degree & seeds) ──
basis = ['cz', 'u']

best_qc = qc
best_count = sum(qc.count_ops().values())

for approx in [0.999, 0.995, 0.99]:
    for seed in range(5):
        candidate = transpile(
            qc,
            basis_gates=basis,
            approximation_degree=approx,
            optimization_level=3,
            seed_transpiler=seed,
        )
        count = sum(candidate.count_ops().values())
        if count < best_count:
            best_count = count
            best_qc = candidate
            print(f"  New best: approx={approx}, seed={seed} -> {count} gates")

qc_opt = best_qc
print(f"\nAfter Qiskit transpilation:")
print(f"  Gates: {qc_opt.count_ops()}")
print(f"  Total gates: {sum(qc_opt.count_ops().values())}")

# ── Step 4: Qiskit cancellation passes ──
pm = PassManager([
    InverseCancellation([CZGate()]),
    CommutativeCancellation(),
    Optimize1qGates(basis=['u']),
])

for i in range(5):
    prev_count = sum(qc_opt.count_ops().values())
    qc_opt = pm.run(qc_opt)
    new_count = sum(qc_opt.count_ops().values())
    if new_count < prev_count:
        print(f"  Qiskit pass round {i+1}: {prev_count} -> {new_count} gates")
    if new_count >= prev_count:
        break

print(f"\nAfter Qiskit cancellation passes:")
print(f"  Gates: {qc_opt.count_ops()}")
print(f"  Total gates: {sum(qc_opt.count_ops().values())}")

# ── Step 5: pytket optimization (stay in CZ basis via TK2 intermediate) ──
tk_circ = qiskit_to_tk(qc_opt)

tket_pipeline = SequencePass([
    CommuteThroughMultis(),
    FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
    RemoveRedundancies(),
    AutoRebase({OpType.CZ, OpType.U3}),
    RemoveRedundancies(),
])
tket_pipeline.apply(tk_circ)

qc_opt = tk_to_qiskit(tk_circ, replace_implicit_swaps=True)

print(f"\nAfter pytket (TK2 -> CZ rebase):")
print(f"  Gates: {qc_opt.count_ops()}")
print(f"  Depth: {qc_opt.depth()}")
print(f"  Total gates: {sum(qc_opt.count_ops().values())}")

# ── Step 6: Final cleanup — remove near-identity & one more transpile ──
removed = remove_near_identity_u(qc_opt, threshold=0.05)
if removed:
    print(f"  Removed {removed} more near-identity U gates")

qc_opt = transpile(qc_opt, basis_gates=basis, optimization_level=3)

print(f"\nFinal optimized circuit:")
print(f"  Gates: {qc_opt.count_ops()}")
print(f"  Depth: {qc_opt.depth()}")
print(f"  Total gates: {sum(qc_opt.count_ops().values())}")

# ── Gate removal analysis ──
print(f"\n{'='*60}")
print("GATE REMOVAL ANALYSIS")
print(f"{'='*60}")

# Original gate counts per type
orig_ops = qc.count_ops()
final_ops = qc_opt.count_ops()
for gate_type in ['cz', 'u']:
    orig = orig_ops.get(gate_type, 0)
    final = final_ops.get(gate_type, 0)
    print(f"  {gate_type.upper()}: {orig} -> {final}  (removed {orig - final})")

# Which CZ qubit pairs survived
print(f"\n  Surviving CZ pairs:")
for inst in qc_opt.data:
    if inst.operation.name == 'cz':
        qubits = [qc_opt.qubits.index(q) for q in inst.qubits]
        print(f"    CZ q[{qubits[0]}], q[{qubits[1]}]")

# Which qubits still have U gates vs which had them removed
orig_u_qubits = set()
for inst in qc.data:
    if inst.operation.name == 'u':
        orig_u_qubits.add(qc.qubits.index(inst.qubits[0]))

final_u_qubits = set()
for inst in qc_opt.data:
    if inst.operation.name == 'u':
        final_u_qubits.add(qc_opt.qubits.index(inst.qubits[0]))

removed_u_qubits = orig_u_qubits - final_u_qubits
print(f"\n  Qubits that kept U gates ({len(final_u_qubits)}): {sorted(final_u_qubits)}")
print(f"  Qubits whose U gates were removed ({len(removed_u_qubits)}): {sorted(removed_u_qubits)}")

# Original CZ pairs vs surviving
from collections import Counter as Ctr
orig_cz_pairs = Ctr()
for inst in qc.data:
    if inst.operation.name == 'cz':
        pair = tuple(sorted([qc.qubits.index(inst.qubits[0]), qc.qubits.index(inst.qubits[1])]))
        orig_cz_pairs[pair] += 1

print(f"\n  Original CZ pair counts (even = fully canceled, odd = 1 survives):")
for pair, cnt in sorted(orig_cz_pairs.items()):
    status = "CANCELED" if cnt % 2 == 0 else "SURVIVES (odd count)"
    print(f"    q[{pair[0]}]-q[{pair[1]}]: {cnt} occurrences -> {status}")

# Sample examples of canceled vs surviving gates
import random
random.seed(42)

canceled_cz = [(p, c) for p, c in orig_cz_pairs.items() if c % 2 == 0]
survived_cz = [(p, c) for p, c in orig_cz_pairs.items() if c % 2 == 1]

print(f"\n  --- Sample CZ cancellation examples ---")
for pair, cnt in random.sample(canceled_cz, min(3, len(canceled_cz))):
    print(f"    CZ q[{pair[0]}]-q[{pair[1]}]: appeared {cnt}x (even) -> all canceled as CZ·CZ = I")
for pair, cnt in random.sample(survived_cz, min(3, len(survived_cz))):
    print(f"    CZ q[{pair[0]}]-q[{pair[1]}]: appeared {cnt}x (odd) -> {cnt-1} canceled, 1 survives")

# Sample examples of U gate removal
orig_u_per_qubit = Ctr()
for inst in qc.data:
    if inst.operation.name == 'u':
        orig_u_per_qubit[qc.qubits.index(inst.qubits[0])] += 1

final_u_per_qubit = Ctr()
for inst in qc_opt.data:
    if inst.operation.name == 'u':
        final_u_per_qubit[qc_opt.qubits.index(inst.qubits[0])] += 1

print(f"\n  --- Sample U gate merging/removal examples ---")
sample_kept = random.sample(sorted(final_u_qubits), min(3, len(final_u_qubits)))
for q in sample_kept:
    print(f"    q[{q}]: {orig_u_per_qubit[q]} U gates -> {final_u_per_qubit[q]} (merged consecutive U's into fewer)")

sample_removed = random.sample(sorted(removed_u_qubits), min(3, len(removed_u_qubits)))
for q in sample_removed:
    print(f"    q[{q}]: {orig_u_per_qubit[q]} U gates -> 0 (all merged to near-identity, approx removed)")

# ── Step 7: MPS Sampling on simplified circuit ──
print(f"\n{'='*60}")
print("MPS Sampling on simplified circuit")
print(f"{'='*60}")

results = bq.run(
    qc_opt,
    device="mps.cpu",
    shots=5000,
    options={"mps_bond_dimension": 64},
)
counts = results.get_counts()
total_shots = sum(counts.values())
top10_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

peak_bitstring_mps = top10_counts[0][0]

print(f"Top 10 most frequent bitstrings from MPS Sampling ({total_shots} shots, {len(counts)} unique):\n")
for rank, (bitstring, count) in enumerate(top10_counts, 1):
    prob = count / total_shots
    print(f"  {rank:2d}. {bitstring}: {count:5d}  (prob ~ {prob:.4f})")

# Confidence: ratio of top count to runner-up
if len(top10_counts) > 1:
    top_count = top10_counts[0][1]
    runner_up = top10_counts[1][1]
    confidence = top_count / runner_up if runner_up > 0 else float('inf')
    top_prob = top_count / total_shots
    print(f"\n  Peak bitstring:     {peak_bitstring_mps}")
    print(f"  Approx probability: {top_prob:.4f} ({top_prob*100:.2f}%)")
    print(f"  Confidence (top/2nd): {confidence:.2f}x")
else:
    print(f"\n  Peak bitstring:     {peak_bitstring_mps}")
    print(f"  Approx probability: {top10_counts[0][1]/total_shots:.4f}")

# ── Summary ──
original_gates = 2148
final_gates = sum(qc_opt.count_ops().values())
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Original gates:   {original_gates}")
print(f"  Optimized gates:  {final_gates}")
print(f"  Reduction:        {100*(1 - final_gates/original_gates):.1f}%")
print(f"  Peak bitstring:   {peak_bitstring_mps}")

# ── Save results to bitstrings.txt ──
with open("bitstrings.txt", "w") as f:
    f.write(f"P6 Results\n")
    f.write(f"{'='*60}\n")
    f.write(f"Original gates: {original_gates}\n")
    f.write(f"Optimized gates: {final_gates}\n")
    f.write(f"Reduction: {100*(1 - final_gates/original_gates):.1f}%\n\n")
    f.write(f"Top 10 bitstrings ({total_shots} shots, {len(counts)} unique):\n")
    for rank, (bitstring, count) in enumerate(top10_counts, 1):
        prob = count / total_shots
        f.write(f"  {rank:2d}. {bitstring}: {count:5d}  (prob ~ {prob:.4f})\n")
    f.write(f"\nPeak bitstring: {peak_bitstring_mps}\n")
    top_prob = top10_counts[0][1] / total_shots
    f.write(f"Approx probability: {top_prob:.4f} ({top_prob*100:.2f}%)\n")
    if len(top10_counts) > 1:
        confidence = top10_counts[0][1] / top10_counts[1][1]
        f.write(f"Confidence (top/2nd): {confidence:.2f}x\n")

print(f"\nResults saved to bitstrings.txt")
