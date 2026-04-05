"""
MPO Test Suite — Sampling-based validation at 10-15 qubits
============================================================
Tests the full stack: mpo_core.py + mpo_absorber.py + mpo_unswap.py

Each test builds R→P→R†, transpiles, absorbs into MPO, samples 1000
shots from the canonicalized MPS, and compares with brute-force peak.

Usage:
    python mpo_tests.py

Requires: mpo_core.py, mpo_absorber.py, mpo_unswap.py in the same directory.
"""

import numpy as np_cpu
from gpu_backend import xp, to_numpy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile
from collections import Counter

from mpo_core import (
    identity_mpo,
    absorb_single_qubit_gate,
    absorb_two_qubit_gate,
    get_bond_dims,
    mpo_total_elements,
    mpo_to_mps,
    extract_peak_greedy,
    sample_mps,
)
from mpo_absorber import parse_circuit_ops
from mpo_unswap import unswap_mpo


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def build_peaked_circuit(n_qubits, n_r_gates, peak_qubits,
                         overlap_r_with_p=True, seed=42):
    """Build R → P → R† where P = X on each qubit in peak_qubits.

    peak_qubits: int or list of ints
    """
    if isinstance(peak_qubits, int):
        peak_qubits = [peak_qubits]

    rng = np_cpu.random.default_rng(seed)
    available = (list(range(n_qubits)) if overlap_r_with_p
                 else [i for i in range(n_qubits) if i not in peak_qubits])

    qc_r = QuantumCircuit(n_qubits)
    for _ in range(n_r_gates):
        if rng.random() < 0.5:
            q = int(rng.choice(available))
            theta = float(rng.uniform(0, 2 * np_cpu.pi))
            gate = rng.choice(['ry', 'rz', 'rx'])
            getattr(qc_r, gate)(theta, q)
        else:
            if len(available) >= 2:
                q1, q2 = rng.choice(available, 2, replace=False)
                qc_r.cx(int(q1), int(q2))

    qc = qc_r.copy()
    for pq in peak_qubits:
        qc.x(pq)
    qc = qc.compose(qc_r.inverse())
    return qc


def absorb_full_circuit(qc_transpiled, n_qubits, max_bond=256):
    """Parse and absorb a transpiled circuit into an MPO."""
    ops = parse_circuit_ops(qc_transpiled)
    mid = len(ops) // 2
    left_ops = ops[:mid][::-1]
    right_ops = ops[mid:]

    mpo = identity_mpo(n_qubits)
    for i in range(max(len(left_ops), len(right_ops))):
        if i < len(left_ops):
            op = left_ops[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='left')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo, qs[0], qs[1], op['matrix'],
                                          side='left', max_bond=max_bond)
        if i < len(right_ops):
            op = right_ops[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='right')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo, qs[0], qs[1], op['matrix'],
                                          side='right', max_bond=max_bond)
    return mpo


def run_sampling_test(label, n_qubits, n_r_gates, peak_qubits,
                      overlap_r_with_p, n_samples=1000, max_bond=256, seed=42):
    """
    Full pipeline test: build → transpile → absorb → sample → compare with BF.
    Returns True if the most-sampled bitstring matches the brute-force peak.
    """
    if isinstance(peak_qubits, int):
        peak_qubits = [peak_qubits]

    print(f"\n{'='*60}")
    print(f"  {label}: {n_qubits}q, {n_r_gates} R-gates, "
          f"P=X(q{peak_qubits}), overlap={overlap_r_with_p}")
    print(f"{'='*60}")

    # Build circuit
    qc = build_peaked_circuit(n_qubits, n_r_gates, peak_qubits,
                              overlap_r_with_p, seed)
    print(f"Original: {qc.size()} gates, depth {qc.depth()}")

    # Transpile
    qc_t = transpile(qc, coupling_map=CouplingMap.from_line(n_qubits),
                     basis_gates=['u3', 'cx'], optimization_level=0,
                     seed_transpiler=seed)
    print(f"Transpiled: {qc_t.size()} gates, depth {qc_t.depth()}")

    # Brute force on transpiled circuit
    sv = Statevector.from_instruction(qc_t)
    probs = sv.probabilities_dict()
    bf_peak = max(probs, key=probs.get)
    bf_prob = probs[bf_peak]
    print(f"BF peak (Qiskit): {bf_peak} (prob={bf_prob:.6f})")

    # MPO absorption
    mpo = absorb_full_circuit(qc_t, n_qubits, max_bond=max_bond)
    bonds = get_bond_dims(mpo)
    elems = mpo_total_elements(mpo)
    print(f"MPO: max_bond={max(bonds)}, elems={elems}")

    # MPS (canonicalized) → sample
    mps = mpo_to_mps(mpo)
    samples = sample_mps(mps, n_samples=n_samples, seed=seed)
    freq = Counter(samples)
    top3 = freq.most_common(3)

    sample_peak_qiskit = top3[0][0][::-1]
    sample_count = top3[0][1]

    print(f"Sample peak (Qiskit): {sample_peak_qiskit} ({sample_count}/{n_samples})")
    print(f"Top 3: {[(s[::-1], c) for s, c in top3]}")

    match = (sample_peak_qiskit == bf_peak)
    expected = int(bf_prob * n_samples)
    within_3sigma = abs(sample_count - expected) < 3 * np_cpu.sqrt(max(expected, 1))

    print(f"Peak match: {match}")
    print(f"Frequency: {sample_count} vs expected ~{expected} (3σ ok: {within_3sigma})")

    passed = match and within_3sigma
    print(f"{'PASSED' if passed else 'FAILED'}")

    return passed


# ══════════════════════════════════════════════════
# UNSWAPPING VALIDATION
# ══════════════════════════════════════════════════

def mpo_to_full_operator(mpo):
    """Contract MPO into a 2^n × 2^n operator matrix."""
    n = len(mpo)
    op = mpo[0][0, :, :, :]
    for i in range(1, n):
        op = xp.einsum('...b,bpqr->...pqr', op, mpo[i])
    op = op[..., 0]
    indices = list(range(2 * n))
    op = op.transpose(indices[0::2] + indices[1::2])
    return op.reshape(2**n, 2**n)


def perm_to_matrix(perm, n):
    """Convert qubit permutation to 2^n × 2^n permutation matrix."""
    N = 2**n
    mat = xp.zeros((N, N), dtype=xp.complex128)
    inv_perm = [0] * n
    for old, new in enumerate(perm):
        inv_perm[new] = old
    for i in range(N):
        bits = format(i, f'0{n}b')
        new_bits = ''.join(bits[inv_perm[j]] for j in range(n))
        mat[int(new_bits, 2), i] = 1.0
    return mat


def run_unswap_test(label, n_qubits):
    """Planted permutation + operator preservation test."""
    print(f"\n{'='*60}")
    print(f"  {label}: Planted permutation recovery ({n_qubits}q)")
    print(f"{'='*60}")

    from copy import deepcopy

    SWAP_4x4 = xp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                        dtype=xp.complex128)
    CZ = xp.diag(xp.array([1, 1, 1, -1], dtype=xp.complex128))
    X = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
    Ry = lambda t: xp.array([[np_cpu.cos(t/2), -np_cpu.sin(t/2)],
                              [np_cpu.sin(t/2), np_cpu.cos(t/2)]], dtype=xp.complex128)

    # Build nontrivial base MPO
    mpo_base = identity_mpo(n_qubits)
    absorb_single_qubit_gate(mpo_base, 0, X, side='right')
    absorb_single_qubit_gate(mpo_base, n_qubits//2, Ry(0.8), side='right')
    absorb_two_qubit_gate(mpo_base, 1, 2, CZ, side='right', max_bond=256)
    if n_qubits > 4:
        absorb_two_qubit_gate(mpo_base, 3, 4, CZ, side='left', max_bond=256)
        absorb_single_qubit_gate(mpo_base, 4, Ry(1.3), side='left')

    bonds_base = get_bond_dims(mpo_base)
    print(f"Base MPO bonds: {bonds_base}")

    # Plant known swaps
    mpo_inflated = deepcopy(mpo_base)
    mid = n_qubits // 2

    # Left swap at (mid, mid+1)
    absorb_two_qubit_gate(mpo_inflated, mid, mid+1, SWAP_4x4, side='left', max_bond=256)
    planted_P_L = list(range(n_qubits))
    planted_P_L[mid], planted_P_L[mid+1] = planted_P_L[mid+1], planted_P_L[mid]

    # Right swap at (0, 1)
    absorb_two_qubit_gate(mpo_inflated, 0, 1, SWAP_4x4, side='right', max_bond=256)
    planted_P_R = list(range(n_qubits))
    planted_P_R[0], planted_P_R[1] = planted_P_R[1], planted_P_R[0]

    bonds_inflated = get_bond_dims(mpo_inflated)
    print(f"Inflated bonds: {bonds_inflated}")

    M_inflated = mpo_to_full_operator(mpo_inflated)

    # Unswap
    mpo_unswapped, P_L, P_R, stats = unswap_mpo(mpo_inflated, verbose=True)
    bonds_after = get_bond_dims(mpo_unswapped)
    M_unswapped = mpo_to_full_operator(mpo_unswapped)

    # Check operator preservation: M_inflated = P_R @ M_unswapped @ P_L
    P_L_mat = perm_to_matrix(P_L, n_qubits)
    P_R_mat = perm_to_matrix(P_R, n_qubits)
    M_reconstructed = P_R_mat @ M_unswapped @ P_L_mat
    diff = float(xp.max(xp.abs(M_inflated - M_reconstructed)))

    print(f"Operator preservation: diff={diff:.2e} {'✓' if diff < 1e-10 else '✗'}")
    print(f"Planted P_L: {planted_P_L}")
    print(f"Found   P_L: {P_L}")
    print(f"Planted P_R: {planted_P_R}")
    print(f"Found   P_R: {P_R}")
    print(f"Bonds: {bonds_inflated} → {bonds_after}")

    passed = diff < 1e-10 and max(bonds_after) <= max(bonds_inflated)
    print(f"{'PASSED' if passed else 'FAILED'}")
    return passed


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    results = []

    # --- Single-peak sampling tests ---
    results.append(("T1: 10q, 1 peak, separate",
        run_sampling_test("T1", 10, 15, peak_qubits=5,
                          overlap_r_with_p=False, seed=100)))

    results.append(("T2: 10q, 1 peak, overlap",
        run_sampling_test("T2", 10, 15, peak_qubits=3,
                          overlap_r_with_p=True, seed=200)))

    results.append(("T3: 12q, 1 peak, separate",
        run_sampling_test("T3", 12, 25, peak_qubits=6,
                          overlap_r_with_p=False, seed=300)))

    results.append(("T4: 12q, 1 peak, overlap",
        run_sampling_test("T4", 12, 25, peak_qubits=4,
                          overlap_r_with_p=True, seed=400)))

    results.append(("T5: 15q, 1 peak, separate",
        run_sampling_test("T5", 15, 35, peak_qubits=7,
                          overlap_r_with_p=False, seed=500)))

    results.append(("T6: 15q, 1 peak, overlap",
        run_sampling_test("T6", 15, 35, peak_qubits=7,
                          overlap_r_with_p=True, seed=600)))

    # --- Multi-peak sampling tests ---
    results.append(("T7: 10q, 2 peaks, separate",
        run_sampling_test("T7", 10, 15, peak_qubits=[2, 7],
                          overlap_r_with_p=False, seed=700)))

    results.append(("T8: 10q, 3 peaks, separate",
        run_sampling_test("T8", 10, 15, peak_qubits=[1, 5, 8],
                          overlap_r_with_p=False, seed=800)))

    results.append(("T9: 12q, 2 peaks, overlap",
        run_sampling_test("T9", 12, 25, peak_qubits=[3, 9],
                          overlap_r_with_p=True, seed=900)))

    results.append(("T10: 12q, 4 peaks, overlap",
        run_sampling_test("T10", 12, 25, peak_qubits=[1, 4, 7, 10],
                          overlap_r_with_p=True, seed=1000)))

    results.append(("T11: 15q, 3 peaks, separate",
        run_sampling_test("T11", 15, 35, peak_qubits=[2, 7, 12],
                          overlap_r_with_p=False, seed=1100)))

    results.append(("T12: 15q, 5 peaks, overlap",
        run_sampling_test("T12", 15, 35, peak_qubits=[1, 4, 7, 10, 13],
                          overlap_r_with_p=True, seed=1200, max_bond=512)))

    # --- Unswapping tests ---
    results.append(("U1: 5q planted permutation",
        run_unswap_test("U1", 5)))

    results.append(("U2: 6q planted permutation",
        run_unswap_test("U2", 6)))

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, passed in results:
        print(f"  {'✓' if passed else '✗'} {name}")
    n_passed = sum(p for _, p in results)
    print(f"\n  {n_passed}/{len(results)} passed")
