import warnings
import sys
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from pathlib import Path
import networkx as nx

sys.path.append(os.path.abspath(".."))
from modules import *

warnings.filterwarnings("ignore", category=UserWarning)

# ── Load P7 circuit ──
qc = load_qasm('P7_rolling_ridge.qasm')

print(f"Original circuit:")
print(f"  Qubits: {qc.num_qubits}")
print(f"  Gates: {qc.count_ops()}")
print(f"  Depth: {qc.depth()}")
print(f"  Total gates: {sum(qc.count_ops().values())}")

# ── Connectivity analysis ──
G = nx.Graph()
G.add_nodes_from(range(qc.num_qubits))
for inst in qc.data:
    if inst.operation.name == 'cz':
        i = qc.qubits.index(inst.qubits[0])
        j = qc.qubits.index(inst.qubits[1])
        G.add_edge(i, j)

components = list(nx.connected_components(G))
print(f"\n  {len(components)} independent subcircuit(s):")
for idx, comp in enumerate(components):
    print(f"    Component {idx}: {sorted(comp)} ({len(comp)} qubits)")

# ── Simulate each component exactly via statevector ──
# For each component, build a sub-circuit with only those qubits
# 20 qubits -> 2^20 = 1M amplitudes (trivial)
# 22 qubits -> 2^22 = 4M amplitudes (trivial)

component_peaks = {}

for idx, comp in enumerate(components):
    comp_qubits = sorted(comp)
    n_comp = len(comp_qubits)

    # Map original qubit indices to sub-circuit indices
    qubit_map = {orig: new for new, orig in enumerate(comp_qubits)}

    # Build sub-circuit
    sub_qc = QuantumCircuit(n_comp)
    for inst in qc.data:
        op = inst.operation
        orig_indices = [qc.qubits.index(q) for q in inst.qubits]

        # Skip gates not involving this component
        if not all(qi in qubit_map for qi in orig_indices):
            continue

        mapped = [qubit_map[qi] for qi in orig_indices]

        if op.name in ('u3', 'u'):
            sub_qc.u(op.params[0], op.params[1], op.params[2], mapped[0])
        elif op.name == 'cz':
            sub_qc.cz(mapped[0], mapped[1])

    sub_qc.measure_all()

    print(f"\n{'='*60}")
    print(f"Component {idx}: {n_comp} qubits (exact statevector)")
    print(f"{'='*60}")
    print(f"  Sub-circuit gates: {sub_qc.count_ops()}")
    print(f"  Sub-circuit depth: {sub_qc.depth()}")

    # Exact statevector simulation
    sim = AerSimulator(method='statevector')
    compiled = transpile(sub_qc, sim)
    result = sim.run(compiled, shots=100000).result()
    counts = result.get_counts()

    total_shots = sum(counts.values())
    top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\n  Top 10 bitstrings ({total_shots} shots, {len(counts)} unique):")
    for rank, (bs, count) in enumerate(top10, 1):
        prob = count / total_shots
        print(f"    {rank:2d}. {bs}: {count:6d}  (prob ~ {prob:.4f})")

    peak_bs = top10[0][0]
    top_prob = top10[0][1] / total_shots
    if len(top10) > 1:
        confidence = top10[0][1] / top10[1][1]
        print(f"\n  Peak: {peak_bs}  (prob ~ {top_prob:.4f}, confidence {confidence:.2f}x)")
    else:
        print(f"\n  Peak: {peak_bs}  (prob ~ {top_prob:.4f})")

    # Store the peak bitstring mapped back to original qubit positions
    component_peaks[idx] = (comp_qubits, peak_bs)

# ── Combine component bitstrings into full 42-qubit answer ──
# Qiskit uses little-endian: bitstring[0] = last qubit
# The sub-circuit bitstrings are also little-endian within their own qubit ordering

full_bits = ['0'] * qc.num_qubits

for idx, (comp_qubits, peak_bs) in component_peaks.items():
    # peak_bs is little-endian: peak_bs[0] corresponds to comp_qubits[-1]
    for bit_idx, orig_qubit in enumerate(comp_qubits):
        # In little-endian, bit position for qubit k is (n_comp - 1 - local_index)
        # But peak_bs is already a string where index 0 = highest qubit in sub-circuit
        le_pos = len(comp_qubits) - 1 - bit_idx
        full_bits[orig_qubit] = peak_bs[le_pos]

# Convert to little-endian bitstring (Qiskit convention: bit[0] = qubit[n-1])
full_bitstring = ''.join(full_bits[i] for i in reversed(range(qc.num_qubits)))

print(f"\n{'='*60}")
print("COMBINED RESULT")
print(f"{'='*60}")
print(f"  Full 42-qubit peak bitstring: {full_bitstring}")
print(f"  Length: {len(full_bitstring)}")

#  Save results to a text file
with open("bitstrings.txt", "w") as f:
    f.write(f"P7 Results (Exact Statevector via Independent Components)\n")
    f.write(f"{'='*60}\n\n")
    for idx, (comp_qubits, peak_bs) in component_peaks.items():
        f.write(f"Component {idx} ({len(comp_qubits)} qubits): {peak_bs}\n")
        f.write(f"  Qubits: {comp_qubits}\n")
    f.write(f"\nFull 42-qubit peak bitstring: {full_bitstring}\n")

print(f"\nResults saved to bitstrings.txt")
