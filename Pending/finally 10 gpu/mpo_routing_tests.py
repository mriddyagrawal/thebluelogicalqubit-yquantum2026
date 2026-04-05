"""
Routing Isolation Tests
========================
Tests the SWAP-chain routing in isolation from the rest of the pipeline.

Key question: does relabel + SWAP-chain route preserve the operator?

Test levels:
  R1: Single non-adjacent gate on identity MPO
  R2: Single non-adjacent gate on a real (post-absorption) MPO
  R3: All remaining ops after unswap, relabeled + routed, full operator check
  R4: Pipeline end-to-end with max_bond=4096 to rule out truncation

Usage:
    python mpo_routing_tests.py
"""

import numpy as np_cpu
from gpu_backend import xp, to_numpy
from copy import deepcopy
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

from mpo_core import (
    identity_mpo, absorb_single_qubit_gate, absorb_two_qubit_gate,
    get_bond_dims, mpo_total_elements, mpo_to_mps, sample_mps,
)
from mpo_absorber import parse_circuit_ops
from mpo_unswap import unswap_mpo


SWAP_4x4 = xp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                     dtype=xp.complex128)


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def absorb_gate_routed(mpo, q1, q2, matrix, side, max_bond,
                       svd_cutoff=1e-10, log=None):
    """Absorb 2-qubit gate with SWAP-chain routing if non-adjacent.
    If log is a list, appends (side, distance, max_bond_during) tuples."""
    if abs(q1 - q2) == 1:
        absorb_two_qubit_gate(mpo, q1, q2, matrix, side, max_bond, svd_cutoff)
        if log is not None:
            log.append((side, 1, max(get_bond_dims(mpo))))
        return

    lo, hi = min(q1, q2), max(q1, q2)
    distance = hi - lo
    peak_bond = 0

    # SWAP down
    for pos in range(hi - 1, lo, -1):
        absorb_two_qubit_gate(mpo, pos, pos + 1, SWAP_4x4, side,
                              max_bond, svd_cutoff)
        peak_bond = max(peak_bond, max(get_bond_dims(mpo)))

    # Gate
    if q1 > q2:
        absorb_two_qubit_gate(mpo, lo + 1, lo, matrix, side,
                              max_bond, svd_cutoff)
    else:
        absorb_two_qubit_gate(mpo, lo, lo + 1, matrix, side,
                              max_bond, svd_cutoff)
    peak_bond = max(peak_bond, max(get_bond_dims(mpo)))

    # SWAP back
    for pos in range(lo + 1, hi):
        absorb_two_qubit_gate(mpo, pos, pos + 1, SWAP_4x4, side,
                              max_bond, svd_cutoff)
        peak_bond = max(peak_bond, max(get_bond_dims(mpo)))

    if log is not None:
        log.append((side, distance, peak_bond))


def mpo_to_full(mpo):
    """Contract MPO to full 2^n x 2^n operator."""
    n = len(mpo)
    op = mpo[0][0, :, :, :]
    for i in range(1, n):
        op = xp.einsum('...b,bpqr->...pqr', op, mpo[i])
    op = op[..., 0]
    idx = list(range(2 * n))
    return op.transpose(idx[0::2] + idx[1::2]).reshape(2**n, 2**n)


def perm_to_matrix(perm, n):
    """Convert permutation array to 2^n x 2^n permutation matrix."""
    N = 2**n
    mat = xp.zeros((N, N), dtype=xp.complex128)
    inv = [0] * n
    for o, nw in enumerate(perm):
        inv[nw] = o
    for i in range(N):
        b = format(i, f'0{n}b')
        mat[int(''.join(b[inv[j]] for j in range(n)), 2), i] = 1.0
    return mat


def build_peaked_circuit(n, nr, pq, overlap=True, seed=42):
    if isinstance(pq, int):
        pq = [pq]
    rng = np_cpu.random.default_rng(seed)
    av = list(range(n)) if overlap else [i for i in range(n) if i not in pq]
    qc_r = QuantumCircuit(n)
    for _ in range(nr):
        if rng.random() < 0.5:
            getattr(qc_r, rng.choice(['ry', 'rz', 'rx']))(
                float(rng.uniform(0, 2 * np_cpu.pi)), int(rng.choice(av)))
        else:
            if len(av) >= 2:
                q1, q2 = rng.choice(av, 2, replace=False)
                qc_r.cx(int(q1), int(q2))
    qc = qc_r.copy()
    for p in pq:
        qc.x(p)
    return qc.compose(qc_r.inverse())


def build_test_mpo(qc, n_absorb, seed=42):
    """Build an MPO by absorbing n_absorb steps of a transpiled circuit.
    Returns (mpo, left_ops_remaining, right_ops_remaining, all_left_ops, all_right_ops)."""
    n = qc.num_qubits
    qc_t = transpile(qc, coupling_map=CouplingMap.from_line(n),
                     basis_gates=['u3', 'cx'], optimization_level=0,
                     seed_transpiler=seed)
    ops = parse_circuit_ops(qc_t)
    mid = len(ops) // 2
    left_ops = list(reversed(ops[:mid]))
    right_ops = list(ops[mid:])

    mpo = identity_mpo(n)
    for i in range(n_absorb):
        for side_ops, side in [(left_ops, 'left'), (right_ops, 'right')]:
            if i < len(side_ops):
                op = side_ops[i]
                if op['matrix'] is not None:
                    qs = op['qubits']
                    if len(qs) == 1:
                        absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side=side)
                    elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                        absorb_two_qubit_gate(mpo, qs[0], qs[1], op['matrix'],
                                              side=side, max_bond=4096)

    return mpo, left_ops[n_absorb:], right_ops[n_absorb:], left_ops, right_ops


# ══════════════════════════════════════════════════
# R1: Single gate on identity MPO
# ══════════════════════════════════════════════════

def test_R1():
    """SWAP-chain routing of single gates at various distances on identity MPO."""
    print(f"\n{'='*60}")
    print(f"  R1: Single gate routing on identity MPO")
    print(f"{'='*60}")

    CX = xp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=xp.complex128)
    CZ = xp.diag(xp.array([1, 1, 1, -1], dtype=xp.complex128))

    n = 8
    all_pass = True

    for gate_name, gate in [("CZ", CZ), ("CX", CX)]:
        for q1, q2 in [(0, 3), (2, 6), (7, 1), (0, 7)]:
            for side in ['left', 'right']:
                mpo = identity_mpo(n)
                log = []
                absorb_gate_routed(mpo, q1, q2, gate, side, max_bond=4096, log=log)

                # Reference: build operator directly
                M_test = mpo_to_full(mpo)

                # The operator should be: gate on (q1,q2) applied on the given side
                # For identity MPO, left and right give same result
                mpo_ref = identity_mpo(n)
                # Route manually for reference
                lo, hi = min(q1, q2), max(q1, q2)
                for pos in range(hi - 1, lo, -1):
                    absorb_two_qubit_gate(mpo_ref, pos, pos + 1, SWAP_4x4, side, 4096)
                if q1 > q2:
                    absorb_two_qubit_gate(mpo_ref, lo + 1, lo, gate, side, 4096)
                else:
                    absorb_two_qubit_gate(mpo_ref, lo, lo + 1, gate, side, 4096)
                for pos in range(lo + 1, hi):
                    absorb_two_qubit_gate(mpo_ref, pos, pos + 1, SWAP_4x4, side, 4096)
                M_ref = mpo_to_full(mpo_ref)

                diff = float(xp.max(xp.abs(M_test / xp.linalg.norm(M_test)
                                     - M_ref / xp.linalg.norm(M_ref))))
                ok = diff < 1e-10
                if not ok:
                    all_pass = False
                dist = log[0][2] if log else 0
                print(f"  {gate_name}({q1},{q2}) side={side}: "
                      f"diff={diff:.2e}, peak_bond={dist} {'✓' if ok else '✗'}")

    print(f"  R1: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# ══════════════════════════════════════════════════
# R2: Single gate on a real MPO
# ══════════════════════════════════════════════════

def test_R2():
    """SWAP-chain routing on a real (post-absorption) MPO."""
    print(f"\n{'='*60}")
    print(f"  R2: Single gate routing on real MPO")
    print(f"{'='*60}")

    qc = build_peaked_circuit(10, 15, 3, overlap=True, seed=200)
    mpo, rem_left, rem_right, _, _ = build_test_mpo(qc, 25, seed=200)

    mpo, P_L, P_R, _ = unswap_mpo(mpo, max_bond=4096, verbose=False)
    P_R_inv = [0] * 10
    for k in range(10):
        P_R_inv[P_R[k]] = k

    P_L_mat = perm_to_matrix(P_L, 10)
    P_R_mat = perm_to_matrix(P_R, 10)

    print(f"  P_L={P_L}")
    print(f"  P_R={P_R}")
    print(f"  Bonds after unswap: {get_bond_dims(mpo)}")

    all_pass = True

    # Test each remaining gate individually
    for label, ops_list, side, perm in [
        ("left", rem_left[:5], 'left', P_L),
        ("right", rem_right[:5], 'right', P_R_inv),
    ]:
        for i, op in enumerate(ops_list):
            if op['matrix'] is None or len(op['qubits']) != 2:
                continue

            qs_new = [perm[q] for q in op['qubits']]
            dist = abs(qs_new[0] - qs_new[1])

            # Reference: absorb on original (pre-unswap) MPO
            mpo_ref_base, _, _, _, _ = build_test_mpo(qc, 25, seed=200)
            absorb_two_qubit_gate(mpo_ref_base, op['qubits'][0], op['qubits'][1],
                                  op['matrix'], side=side, max_bond=4096)
            M_ref = mpo_to_full(mpo_ref_base)

            # Test: absorb on unswapped MPO with relabeling + routing
            mpo_test = deepcopy(mpo)
            log = []
            absorb_gate_routed(mpo_test, qs_new[0], qs_new[1], op['matrix'],
                               side, max_bond=4096, log=log)
            M_test = mpo_to_full(mpo_test)

            M_recon = P_R_mat @ M_test @ P_L_mat
            diff = float(xp.max(xp.abs(M_ref / xp.linalg.norm(M_ref)
                                 - M_recon / xp.linalg.norm(M_recon))))
            ok = diff < 1e-10
            if not ok:
                all_pass = False
            peak_b = log[0][2] if log else 0
            print(f"  {side} gate {i}: {op['qubits']}→{qs_new} "
                  f"(d={dist}), diff={diff:.2e}, peak_bond={peak_b} "
                  f"{'✓' if ok else '✗'}")

    print(f"  R2: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# ══════════════════════════════════════════════════
# R3: All remaining ops, full operator check
# ══════════════════════════════════════════════════

def test_R3():
    """Absorb ALL remaining ops with relabeling+routing, check final operator."""
    print(f"\n{'='*60}")
    print(f"  R3: Full remaining ops, operator preservation")
    print(f"{'='*60}")

    all_pass = True

    for label, n, nr, pq, overlap, seed, n_absorb in [
        ("10q sep",  10, 15, 5,    False, 100, 25),
        ("10q ovlp", 10, 15, 3,    True,  200, 25),
        ("12q 2pk",  12, 25, [3,9], True,  300, 25),
    ]:
        qc = build_peaked_circuit(n, nr, pq, overlap, seed)
        mpo, rem_left, rem_right, all_left, all_right = build_test_mpo(
            qc, n_absorb, seed)

        # Reference: absorb ALL without unswapping
        mpo_ref = deepcopy(mpo)
        for i in range(len(rem_left)):
            op = rem_left[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo_ref, qs[0], op['matrix'], side='left')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo_ref, qs[0], qs[1], op['matrix'],
                                          side='left', max_bond=4096)
        for i in range(len(rem_right)):
            op = rem_right[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo_ref, qs[0], op['matrix'], side='right')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo_ref, qs[0], qs[1], op['matrix'],
                                          side='right', max_bond=4096)
        M_ref = mpo_to_full(mpo_ref)
        M_ref_n = M_ref / xp.linalg.norm(M_ref)

        # Test: unswap + relabel + route + absorb
        mpo, P_L, P_R, _ = unswap_mpo(mpo, max_bond=4096, verbose=False)
        P_R_inv = [0] * n
        for k in range(n):
            P_R_inv[P_R[k]] = k

        P_L_mat = perm_to_matrix(P_L, n)
        P_R_mat = perm_to_matrix(P_R, n)

        log = []
        for op in rem_left:
            if op['matrix'] is not None:
                qs = [P_L[q] for q in op['qubits']]
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='left')
                elif len(qs) == 2:
                    absorb_gate_routed(mpo, qs[0], qs[1], op['matrix'],
                                       'left', 4096, log=log)
        for op in rem_right:
            if op['matrix'] is not None:
                qs = [P_R_inv[q] for q in op['qubits']]
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='right')
                elif len(qs) == 2:
                    absorb_gate_routed(mpo, qs[0], qs[1], op['matrix'],
                                       'right', 4096, log=log)

        M_test = mpo_to_full(mpo)
        M_recon = P_R_mat @ M_test @ P_L_mat
        diff = float(xp.max(xp.abs(M_ref_n - M_recon / xp.linalg.norm(M_recon))))
        ok = diff < 1e-10

        # Log stats
        non_adj = [l for l in log if l[1] > 1]
        max_peak = max((l[2] for l in log), default=0)

        print(f"  {label}: diff={diff:.2e}, "
              f"routed={len(non_adj)} non-adj gates, "
              f"peak_bond={max_peak} {'✓' if ok else '✗'}")

        if not ok:
            all_pass = False

    print(f"  R3: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# ══════════════════════════════════════════════════
# R4: Pipeline with very large max_bond
# ══════════════════════════════════════════════════

def test_R4():
    """Pipeline end-to-end with max_bond=4096 to rule out truncation."""
    print(f"\n{'='*60}")
    print(f"  R4: Pipeline with max_bond=4096 (no truncation)")
    print(f"{'='*60}")

    from mpo_pipeline import run_pipeline, _bf_transpiled

    all_pass = True

    tests = [
        ("10q ovlp LOW",  10, 15, 3,      True,  200, 8),
        ("12q 2pk LOW",   12, 25, [3, 9],  True,  300, 8),
        ("15q 3pk LOW",   15, 35, [2,7,12],True,  400, 16),
    ]

    for label, n, nr, pq, overlap, seed, bt in tests:
        qc = build_peaked_circuit(n, nr, pq, overlap, seed)
        bf, bf_prob = _bf_transpiled(qc, n, seed)

        result = run_pipeline(
            qc, max_bond=4096, bond_threshold=bt,
            n_samples=5000, seed=seed, verbose=False)

        match = result['peak_bitstring_qiskit'] == bf
        if not match:
            all_pass = False

        print(f"  {label}: {'✓' if match else '✗'} "
              f"(unsw={result['total_unswaps']}, "
              f"final_bond={max(result['final_bonds'])}, "
              f"bf_prob={bf_prob:.3f})")

    print(f"  R4: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    results = []
    results.append(("R1: Single gate on identity", test_R1()))
    results.append(("R2: Single gate on real MPO", test_R2()))
    results.append(("R3: All remaining ops",       test_R3()))
    results.append(("R4: Pipeline max_bond=4096",  test_R4()))

    print(f"\n{'='*60}")
    print(f"  ROUTING TEST SUMMARY")
    print(f"{'='*60}")
    for name, passed in results:
        print(f"  {'✓' if passed else '✗'} {name}")
    n_pass = sum(p for _, p in results)
    print(f"  {n_pass}/{len(results)} passed")
    print(f"{'='*60}")
