import time
import warnings
import sys
import os
import pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit_aer import AerSimulator

sys.path.append(os.path.abspath("../.."))
from modules import *

warnings.filterwarnings("ignore")

# ── Load P8 circuit ──
qc_orig = QuantumCircuit.from_qasm_file('../P8_bold_peak.qasm')

print(f"Original circuit:")
print(f"  Qubits: {qc_orig.num_qubits}")
print(f"  Gates: {qc_orig.count_ops()}")
print(f"  Depth: {qc_orig.depth()}")
print(f"  Total gates: {sum(qc_orig.count_ops().values())}")
n_qubits = qc_orig.num_qubits

# ── Step 1: Canonicalize QASM through Qiskit (reference approach) ──
qasm_str = qasm2.dumps(qc_orig)

# ── Step 2: PyZX simplification (clifford_simp, as in reference) ──
print(f"\n[2] PyZX Simplification (clifford_simp)...")
circuit = zx.Circuit.from_qasm(qasm_str)
print(f"  Original: {circuit.qubits} qubits, {len(circuit.gates)} gates")

g = circuit.to_graph()
zx.simplify.clifford_simp(g)
circ_reduced = zx.extract_circuit(g)
print(f"  After clifford_simp: {len(circ_reduced.gates)} gates")
qasm_opt = circ_reduced.to_qasm()

# ── Step 3: Qiskit transpile (reference: basis_gates=["u3", "cx"], level 3) ──
print(f"\n[3] Qiskit Transpile (level 3, basis=[u3, cx])...")
qc = QuantumCircuit.from_qasm_str(qasm_opt)
print(f"  Before transpile: depth={qc.depth()}, size={qc.size()}")

qc.measure_all()

optimized = transpile(
    qc,
    basis_gates=["u3", "cx"],
    optimization_level=3,
)
print(f"  After transpile: depth={optimized.depth()}, size={optimized.size()}")
print(f"  Gates: {optimized.count_ops()}")

# ── Step 4: MPS Simulation (reference: bond_dim=16 from run_all.py) ──
# for bond_dim in [16, 32, 64]:
#     print(f"\n{'='*60}")
#     print(f"[4] MPS Simulation (bond_dim={bond_dim})")
#     print(f"{'='*60}")

#     try:
#         backend = AerSimulator(method="matrix_product_state")
#         backend.set_options(matrix_product_state_max_bond_dimension=bond_dim)

#         t0 = time.perf_counter()
#         job = backend.run(optimized, shots=5000)
#         result = job.result()

#         if result.status != 'COMPLETED':
#             print(f"  Failed: {result.status}")
#             continue

#         counts = result.get_counts()
#         dt = time.perf_counter() - t0

#         total_shots = sum(counts.values())
#         top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
#         peak_raw = top10[0][0]
#         peak_bs = peak_raw.replace(" ", "")[:n_qubits]

#         print(f"  Time: {dt:.1f}s")
#         print(f"  Top 10 ({total_shots} shots, {len(counts)} unique):\n")
#         for rank, (bs, count) in enumerate(top10, 1):
#             prob = count / total_shots
#             print(f"    {rank:2d}. {bs}: {count:5d}  (prob ~ {prob:.4f})")

#         if len(top10) > 1:
#             top_prob = top10[0][1] / total_shots
#             confidence = top10[0][1] / top10[1][1]
#             print(f"\n    Peak: {peak_bs}")
#             print(f"    Approx probability: {top_prob:.4f} ({top_prob*100:.2f}%)")
#             print(f"    Confidence (top/2nd): {confidence:.2f}x")

#     except Exception as e:
#         print(f"  Failed: {e}")
#         continue

# # ── Summary ──
# print(f"\n{'='*60}")
# print("SUMMARY")
# print(f"{'='*60}")
# print(f"  Peak bitstring: {peak_bs}")

# with open("bitstrings_new.txt", "w") as f:
#     f.write(f"P8 Results (reference approach: PyZX + Qiskit + Aer MPS)\n")
#     f.write(f"Peak bitstring: {peak_bs}\n")

# print(f"\nResults saved to bitstrings_new.txt")
