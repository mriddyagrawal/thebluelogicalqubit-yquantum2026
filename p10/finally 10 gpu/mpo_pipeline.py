"""
MPO Pipeline — Full Orchestrator
===================================
Transpile → absorb → unswap → relabel → route → absorb → sample → peak.

GPU: All tensor operations through xp (CuPy or NumPy).
"""

import time
from gpu_backend import xp, to_numpy
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

from mpo_core import (
    identity_mpo, absorb_single_qubit_gate, absorb_two_qubit_gate,
    get_bond_dims, mpo_total_elements, mpo_to_mps,
    extract_peak_greedy, sample_mps,
)
from mpo_absorber import parse_circuit_ops
from mpo_unswap import unswap_mpo


SWAP_4x4 = xp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                     dtype=xp.complex128)


def absorb_gate_routed(mpo, q1, q2, matrix, side, max_bond, svd_cutoff=1e-10):
    """Absorb 2-qubit gate with SWAP-chain routing if non-adjacent."""
    if abs(q1 - q2) == 1:
        absorb_two_qubit_gate(mpo, q1, q2, matrix, side, max_bond, svd_cutoff)
        return
    lo, hi = min(q1, q2), max(q1, q2)
    for pos in range(hi - 1, lo, -1):
        absorb_two_qubit_gate(mpo, pos, pos + 1, SWAP_4x4, side, max_bond, svd_cutoff)
    if q1 > q2:
        absorb_two_qubit_gate(mpo, lo + 1, lo, matrix, side, max_bond, svd_cutoff)
    else:
        absorb_two_qubit_gate(mpo, lo, lo + 1, matrix, side, max_bond, svd_cutoff)
    for pos in range(lo + 1, hi):
        absorb_two_qubit_gate(mpo, pos, pos + 1, SWAP_4x4, side, max_bond, svd_cutoff)


def run_pipeline(qc, max_bond=256, svd_cutoff=1e-10,
                 bond_threshold=64, element_threshold=5_000_000,
                 check_every=50, n_samples=5000, seed=42, verbose=True):
    n = qc.num_qubits
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MPO PIPELINE: {n} qubits, {qc.size()} gates")
        print(f"{'='*60}")

    qc_t = transpile(qc, coupling_map=CouplingMap.from_line(n),
                     basis_gates=['u3', 'cx'], optimization_level=0,
                     seed_transpiler=seed)

    if verbose:
        print(f"Transpiled: {qc.size()} → {qc_t.size()} gates")

    ops = parse_circuit_ops(qc_t)
    n_ops = len(ops)
    mid = n_ops // 2
    left_ops = list(reversed(ops[:mid]))
    right_ops = list(ops[mid:])

    if verbose:
        print(f"Ops: {n_ops}, split at {mid}")

    mpo = identity_mpo(n)
    acc_P_R = list(range(n))
    lp = rp = total = total_unswaps = 0

    while lp < len(left_ops) or rp < len(right_ops):
        if lp < len(left_ops):
            op = left_ops[lp]; lp += 1
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='left')
                elif len(qs) == 2:
                    absorb_gate_routed(mpo, qs[0], qs[1], op['matrix'],
                                       'left', max_bond, svd_cutoff)
                total += 1

        if rp < len(right_ops):
            op = right_ops[rp]; rp += 1
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'], side='right')
                elif len(qs) == 2:
                    absorb_gate_routed(mpo, qs[0], qs[1], op['matrix'],
                                       'right', max_bond, svd_cutoff)
                total += 1

        if total % 200 == 0 and total > 0:
            for idx in range(n):
                t_norm = float(xp.linalg.norm(mpo[idx]))
                if t_norm > 0 and t_norm < float('inf'):
                    mpo[idx] = mpo[idx] / t_norm

        if total % check_every == 0 and total > 0:
            bonds = get_bond_dims(mpo)
            max_b = max(bonds) if bonds else 1
            elems = mpo_total_elements(mpo)

            if verbose and total % (check_every * 4) == 0:
                print(f"  [{total:5d}] max_bond={max_b:4d}, elems={elems:10d}")

            if max_b > bond_threshold or elems > element_threshold:
                if verbose:
                    print(f"  [{total:5d}] UNSWAP (bond={max_b}, elems={elems})")

                mpo, P_L, P_R, ustats = unswap_mpo(
                    mpo, svd_cutoff=svd_cutoff, max_bond=max_bond,
                    verbose=verbose)
                total_unswaps += 1

                if P_L != list(range(n)) or P_R != list(range(n)):
                    P_R_inv = [0] * n
                    for k in range(n):
                        P_R_inv[P_R[k]] = k

                    new_left = [dict(op, qubits=[P_L[q] for q in op['qubits']])
                                for op in left_ops[lp:]]
                    new_right = [dict(op, qubits=[P_R_inv[q] for q in op['qubits']])
                                 for op in right_ops[rp:]]
                    left_ops = new_left
                    right_ops = new_right
                    lp = 0; rp = 0

                    # Accumulate P_R (right-composition)
                    acc_P_R = [acc_P_R[P_R[j]] for j in range(n)]

                if verbose:
                    bonds_after = get_bond_dims(mpo)
                    print(f"  After: max_bond={max(bonds_after)}, "
                          f"swaps={ustats['total_swaps']}")

    # Extract
    total_time = time.time() - t_start
    bonds = get_bond_dims(mpo)

    if verbose:
        print(f"\n--- Extraction ---")
        print(f"Final: max_bond={max(bonds)}, elems={mpo_total_elements(mpo)}")
        print(f"Absorbed: {total}, unswaps: {total_unswaps}")

    mps = mpo_to_mps(mpo)
    samples_mpo = sample_mps(mps, n_samples=n_samples, seed=42)
    freq = Counter(samples_mpo)
    peak_mpo = freq.most_common(1)[0][0]
    greedy_mpo = extract_peak_greedy(mps)

    # Apply accumulated P_R to undo output permutation
    def apply_perm(bitstring, perm):
        result = [''] * len(bitstring)
        for i in range(len(bitstring)):
            result[perm[i]] = bitstring[i]
        return ''.join(result)

    peak_corrected = apply_perm(peak_mpo, acc_P_R)
    greedy_corrected = apply_perm(greedy_mpo, acc_P_R)

    peak_qiskit = peak_corrected[::-1]
    greedy_qiskit = greedy_corrected[::-1]

    if verbose:
        top3 = freq.most_common(3)
        print(f"Sample peak (Qiskit): {peak_qiskit}")
        print(f"Greedy peak (Qiskit): {greedy_qiskit}")
        print(f"Top 3: {[(s[::-1], c) for s, c in top3]}")
        print(f"Time: {total_time:.1f}s")

    return {
        'peak_bitstring_qiskit': peak_qiskit,
        'greedy_bitstring_qiskit': greedy_qiskit,
        'mpo': mpo,
        'final_bonds': bonds,
        'total_absorbed': total,
        'total_unswaps': total_unswaps,
        'accumulated_P_R': acc_P_R,
        'total_time': total_time,
    }


# ══════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════

def build_peaked_circuit(n_qubits, n_r_gates, peak_qubits, overlap=True, seed=42):
    import numpy as np
    if isinstance(peak_qubits, int): peak_qubits = [peak_qubits]
    rng = np.random.default_rng(seed)
    av = list(range(n_qubits)) if overlap else [i for i in range(n_qubits) if i not in peak_qubits]
    qc_r = QuantumCircuit(n_qubits)
    for _ in range(n_r_gates):
        if rng.random() < 0.5:
            getattr(qc_r, rng.choice(['ry','rz','rx']))(
                float(rng.uniform(0,2*np.pi)), int(rng.choice(av)))
        else:
            if len(av) >= 2:
                q1, q2 = rng.choice(av, 2, replace=False)
                qc_r.cx(int(q1), int(q2))
    qc = qc_r.copy()
    for pq in peak_qubits: qc.x(pq)
    return qc.compose(qc_r.inverse())


def _bf_transpiled(qc, n, seed=42):
    from qiskit.quantum_info import Statevector
    qc_t = transpile(qc, coupling_map=CouplingMap.from_line(n),
                     basis_gates=['u3','cx'], optimization_level=0, seed_transpiler=seed)
    probs = Statevector.from_instruction(qc_t).probabilities_dict()
    peak = max(probs, key=probs.get)
    return peak, probs[peak]


def run_test(label, qc, **kwargs):
    n = qc.num_qubits; seed = kwargs.get('seed', 42)
    bf, bf_prob = _bf_transpiled(qc, n, seed)
    result = run_pipeline(qc, verbose=False, **kwargs)
    match = result['peak_bitstring_qiskit'] == bf
    print(f"  {label}: {'✓' if match else '✗'} "
          f"(unsw={result['total_unswaps']}, p={bf_prob:.3f})")
    return match


if __name__ == "__main__":
    results = []
    results.append(("P1", run_test("P1: 10q sep", build_peaked_circuit(10,15,5,False,100), max_bond=128, bond_threshold=32, seed=100)))
    results.append(("P2", run_test("P2: 10q ovlp", build_peaked_circuit(10,15,3,True,200), max_bond=128, bond_threshold=32, seed=200)))
    results.append(("P3", run_test("P3: 12q 2pk", build_peaked_circuit(12,25,[3,9],True,300), max_bond=256, bond_threshold=64, seed=300)))
    results.append(("P4", run_test("P4: 15q 3pk", build_peaked_circuit(15,35,[2,7,12],True,400), max_bond=512, bond_threshold=128, seed=400)))
    results.append(("P5", run_test("P5: 10q LOW", build_peaked_circuit(10,15,5,False,100), max_bond=512, bond_threshold=8, seed=100, n_samples=5000)))
    results.append(("P6", run_test("P6: 10q ovlp LOW", build_peaked_circuit(10,15,3,True,200), max_bond=512, bond_threshold=8, seed=200, n_samples=5000)))
    results.append(("P7", run_test("P7: 12q LOW", build_peaked_circuit(12,25,[3,9],True,300), max_bond=512, bond_threshold=8, seed=300, n_samples=5000)))
    results.append(("P8", run_test("P8: 15q LOW", build_peaked_circuit(15,35,[2,7,12],True,400), max_bond=512, bond_threshold=16, seed=400, n_samples=5000)))

    print(f"\n  {sum(p for _,p in results)}/{len(results)} passed")
