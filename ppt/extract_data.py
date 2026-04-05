"""
Run real optimizations on P5, P6, P7 circuits and export results as JSON
for the presentation slides.
"""
import sys, json, warnings, os
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from modules import load_qasm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    InverseCancellation,
    CommutativeCancellation,
    Optimize1qGates,
)
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise
import networkx as nx
import numpy as np

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
img = Path(__file__).resolve().parent / "img"

# ── helpers ──────────────────────────────────────────────
def circuit_stats(qc, label=""):
    ops = qc.count_ops()
    total = sum(ops.values())
    d = qc.depth()
    print(f"  {label}: {total} gates (depth {d})  {dict(ops)}")
    return {"gates": dict(ops), "total_gates": total, "depth": d, "num_qubits": qc.num_qubits}

def strip_barriers(qc):
    qc2 = qc.copy()
    qc2.data = [inst for inst in qc2.data if inst.operation.name != "barrier"]
    return qc2

def tket_opt(qc):
    tk = qiskit_to_tk(qc)
    FullPeepholeOptimise().apply(tk)
    return tk_to_qiskit(tk, replace_implicit_swaps=True)

# ═══════════════════════════════════════════════════════════
# P5  –  Soft Rise  (50 qubits)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("P5: Soft Rise")
print("=" * 60)

qc5 = load_qasm(str(repo / "P5" / "P5_soft_rise.qasm"))
p5_orig = circuit_stats(qc5, "Original")

qc5_nb = strip_barriers(qc5)
qc5_qiskit = transpile(qc5_nb, basis_gates=["u", "cz"], optimization_level=3, coupling_map=None)
p5_qiskit = circuit_stats(qc5_qiskit, "After Qiskit opt=3")

qc5_tket = tket_opt(qc5_qiskit)
p5_final = circuit_stats(qc5_tket, "After Tket")

# Read verified answer
p5_bitstring = (repo / "P5" / "mridul" / "answer.txt").read_text().strip()
print(f"  Peak bitstring: {p5_bitstring}")

p5_data = {
    "problem": "P5",
    "name": "Soft Rise",
    "num_qubits": qc5.num_qubits,
    "original": p5_orig,
    "after_qiskit": p5_qiskit,
    "final": p5_final,
    "peak_bitstring": p5_bitstring,
    "simulation": {"method": "BlueQubit MPS", "shots": 5000, "bond_dimension": 32},
}

# Save optimized circuit drawing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = qc5_tket.draw("mpl", fold=-1)
fig.savefig(img / "P5_optimized_circuit.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════
# P6  –  Low Hill  (60 qubits)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P6: Low Hill")
print("=" * 60)

qc6 = load_qasm(str(repo / "P6" / "P6_low_hill.qasm"))
p6_orig = circuit_stats(qc6, "Original")

# ── Rahul's optimization pipeline (from P6/rahul/P6.py) ──
qc6_nb = strip_barriers(qc6)

# Remove near-identity U gates
THRESHOLD = 0.05
cleaned_data = []
removed_identity = 0
for inst in qc6_nb.data:
    if inst.operation.name == "u" and len(inst.qubits) == 1:
        params = [float(p) for p in inst.operation.params]
        if all(abs(p) < THRESHOLD or abs(abs(p) - 2 * np.pi) < THRESHOLD for p in params):
            removed_identity += 1
            continue
    cleaned_data.append(inst)
qc6_nb.data = cleaned_data
print(f"  Removed {removed_identity} near-identity U gates")

# Qiskit transpile with approximation
best_qc6 = None
best_count = float("inf")
for approx in [0.999, 0.995, 0.99]:
    for seed in range(5):
        t = transpile(
            qc6_nb,
            basis_gates=["u", "cz"],
            optimization_level=3,
            coupling_map=None,
            seed_transpiler=seed,
            approximation_degree=approx,
        )
        c = sum(t.count_ops().values())
        if c < best_count:
            best_count = c
            best_qc6 = t
print(f"  Best Qiskit transpile: {best_count} gates")

# Qiskit pass manager cancellations
pm = PassManager([
    InverseCancellation([CZGate()]),
    CommutativeCancellation(),
    Optimize1qGates(basis=["u"]),
])
qc6_opt = best_qc6
for i in range(5):
    prev = sum(qc6_opt.count_ops().values())
    qc6_opt = pm.run(qc6_opt)
    cur = sum(qc6_opt.count_ops().values())
    if cur >= prev:
        break
p6_after_qiskit = circuit_stats(qc6_opt, "After Qiskit passes")

# Tket optimization
qc6_tket = tket_opt(qc6_opt)
p6_final = circuit_stats(qc6_tket, "After Tket")

# Analyze CZ cancellation
orig_cz = sum(1 for inst in qc6.data if inst.operation.name == "cz")
final_ops = qc6_tket.count_ops()
final_cz = final_ops.get("cx", 0) + final_ops.get("cz", 0)
final_u = sum(v for k, v in final_ops.items() if k in ("u", "u3", "u2", "u1"))
reduction = (1 - p6_final["total_gates"] / p6_orig["total_gates"]) * 100

# Read verified results from bitstrings.txt
p6_results_txt = (repo / "P6" / "bitstrings.txt").read_text()
# Parse top bitstrings from the file
p6_top_bitstrings = []
for line in p6_results_txt.splitlines():
    line = line.strip()
    if line and line[0].isdigit() and "." in line[:4]:
        parts = line.split()
        # format: "1. 101100...: 4952  (prob ~ 0.9904)"
        bs = parts[1].rstrip(":")
        count = int(parts[2])
        p6_top_bitstrings.append({"bitstring": bs, "count": count})

p6_peak = "101100101001010001110111100101100011101011100000000000110111"
print(f"  Peak bitstring: {p6_peak}")
print(f"  CZ: {orig_cz} -> {final_cz}, Reduction: {reduction:.1f}%")

p6_data = {
    "problem": "P6",
    "name": "Low Hill",
    "num_qubits": qc6.num_qubits,
    "original": p6_orig,
    "after_qiskit": p6_after_qiskit,
    "final": p6_final,
    "original_cz": orig_cz,
    "final_cz": final_cz,
    "final_u": final_u,
    "reduction_pct": round(reduction, 1),
    "peak_bitstring": p6_peak,
    "top_bitstrings": p6_top_bitstrings[:10],
    "simulation": {"method": "quimb MPS", "shots": 5000, "bond_dimension": 64},
}

fig = qc6_tket.draw("mpl", fold=-1)
fig.savefig(img / "P6_optimized_circuit.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════
# P7  –  Rolling Ridge  (42 qubits)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P7: Rolling Ridge")
print("=" * 60)

qc7 = load_qasm(str(repo / "P7" / "P7_rolling_ridge.qasm"))
p7_orig = circuit_stats(qc7, "Original")

# Build connectivity graph
G = nx.Graph()
G.add_nodes_from(range(qc7.num_qubits))
cz_edges = []
for inst in qc7.data:
    if inst.operation.name == "cz":
        i = qc7.qubits.index(inst.qubits[0])
        j = qc7.qubits.index(inst.qubits[1])
        G.add_edge(i, j)
        cz_edges.append([i, j])

components = [sorted(c) for c in nx.connected_components(G)]
print(f"  Components: {len(components)}")
for idx, comp in enumerate(components):
    print(f"    Component {idx}: {len(comp)} qubits -> {comp}")

# Read verified results
p7_results_txt = (repo / "P7" / "bitstrings.txt").read_text()
p7_peak = "101101010100001000100011001011101011000100"

# Parse component peaks from results file
comp_peaks = []
for line in p7_results_txt.splitlines():
    if "Peak:" in line or "peak:" in line:
        parts = line.split()
        for p in parts:
            if len(p) > 10 and all(c in "01" for c in p):
                comp_peaks.append(p)

print(f"  Component peaks: {comp_peaks}")
print(f"  Final peak: {p7_peak}")

# Export graph data for interactive visualization
pos = nx.spring_layout(G, seed=42, k=1.5)
nodes_data = []
for n in G.nodes():
    comp_idx = 0 if n in components[0] else 1
    nodes_data.append({
        "id": n,
        "x": float(pos[n][0]),
        "y": float(pos[n][1]),
        "component": comp_idx,
    })
edges_data = [[int(u), int(v)] for u, v in G.edges()]

p7_data = {
    "problem": "P7",
    "name": "Rolling Ridge",
    "num_qubits": qc7.num_qubits,
    "original": p7_orig,
    "num_components": len(components),
    "components": [{"index": i, "num_qubits": len(c), "qubits": c} for i, c in enumerate(components)],
    "component_peaks": comp_peaks,
    "peak_bitstring": p7_peak,
    "graph": {"nodes": nodes_data, "edges": edges_data},
    "simulation": {"method": "Exact Statevector (AerSimulator)", "shots_per_component": 100000},
}

# Generate P6 gate reduction chart with real numbers
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["CZ Gates", "U Gates", "Total Gates", "Depth"]
before = [p6_data["original_cz"], p6_orig["total_gates"] - p6_data["original_cz"] - p6_orig["gates"].get("measure", 0),
          p6_orig["total_gates"], p6_orig["depth"]]
after = [p6_data["final_cz"], p6_data["final_u"], p6_final["total_gates"], p6_final["depth"]]
x = range(len(categories))
w = 0.35
bars1 = ax.bar([i - w/2 for i in x], before, w, label="Before Optimization", color="#EF5350")
bars2 = ax.bar([i + w/2 for i in x], after, w, label="After Optimization", color="#66BB6A")
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"P6: Gate Cancellation Results ({p6_data['reduction_pct']}% Reduction)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.set_yscale("log")
for bar, val in zip(bars1, before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, str(val), ha="center", fontsize=9)
for bar, val in zip(bars2, after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, str(val), ha="center", fontsize=9)
fig.savefig(img / "P6_gate_reduction.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Generate P7 graph partition image with real data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
colors = ["#4FC3F7" if n in components[0] else "#FF8A65" for n in G.nodes()]
nx.draw(G, pos, ax=ax, node_color=colors, with_labels=True, node_size=400,
        font_size=8, font_weight="bold", edge_color="#888", width=0.8)
ax.set_title("P7: Circuit Decomposes into 2 Independent Components", fontsize=14, fontweight="bold")
fig.savefig(img / "P7_graph_partition.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Qubit scaling chart
fig, ax = plt.subplots(figsize=(9, 4))
problems = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
qubits = [4, 20, 30, 40, 50, 60, 42]
colors_bar = ["#90CAF9"] * 4 + ["#4FC3F7", "#FF8A65", "#AB47BC"]
bars = ax.bar(problems, qubits, color=colors_bar, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Qubits", fontsize=12)
ax.set_title("Challenge: Scaling Quantum Circuit Simulation", fontsize=14, fontweight="bold")
for bar, q in zip(bars, qubits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(q), ha="center", fontsize=11, fontweight="bold")
fig.savefig(img / "qubit_scaling.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── Save all data as JSON ──
all_data = {"p5": p5_data, "p6": p6_data, "p7": p7_data}
(out / "results.json").write_text(json.dumps(all_data, indent=2))

print("\n" + "=" * 60)
print("All data exported to ppt/data/results.json")
print("All images regenerated in ppt/img/")
print("=" * 60)
