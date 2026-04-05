"""
MPO Absorber — Circuit Parsing + Absorption Pipeline
=====================================================
Transpiles Qiskit circuits to linear connectivity, parses gates,
and absorbs them into an MPO.

GPU: Gate matrices are created as numpy (from Qiskit), stored as-is.
     mpo_core.py converts them to xp arrays during absorption.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

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


# ══════════════════════════════════════════════════
# CIRCUIT PARSING
# ══════════════════════════════════════════════════

SWAP_2Q = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                    dtype=np.complex128)


def transpile_to_linear(qc, seed=42):
    """Transpile a circuit to linear connectivity."""
    n = qc.num_qubits
    qc_t = transpile(
        qc,
        coupling_map=CouplingMap.from_line(n),
        basis_gates=['u3', 'cx'],
        optimization_level=0,
        seed_transpiler=seed,
    )
    return qc_t


def parse_circuit_ops(qc):
    """Extract gate operations with matrices from a transpiled circuit.

    Returns list of dicts with:
        'qubits': list of qubit indices
        'matrix': gate matrix (numpy, SWAP-conjugated for 2q gates) or None
        'name': gate name
    """
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        if name in ('barrier', 'measure'):
            ops.append({'qubits': [], 'matrix': None, 'name': name})
            continue

        qubit_indices = [qc.find_bit(q).index for q in inst.qubits]
        mat = Operator(inst.operation).data.astype(np.complex128)

        if len(qubit_indices) == 2:
            mat = SWAP_2Q @ mat @ SWAP_2Q

        ops.append({
            'qubits': qubit_indices,
            'matrix': mat,
            'name': name,
        })

    return ops


# ══════════════════════════════════════════════════
# FULL ABSORPTION
# ══════════════════════════════════════════════════

def absorb_circuit(qc, max_bond=256, svd_cutoff=1e-10, seed=42, verbose=True):
    """Transpile and absorb a full circuit into an MPO."""
    n = qc.num_qubits
    qc_t = transpile_to_linear(qc, seed)
    ops = parse_circuit_ops(qc_t)
    mid = len(ops) // 2
    left_ops = ops[:mid][::-1]
    right_ops = ops[mid:]

    if verbose:
        print(f"Transpiled: {qc.size()} → {qc_t.size()} gates")
        print(f"Ops: {len(ops)}, split at {mid}")

    mpo = identity_mpo(n)

    for i in range(max(len(left_ops), len(right_ops))):
        if i < len(left_ops):
            op = left_ops[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'],
                                             side='left')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo, qs[0], qs[1], op['matrix'],
                                          side='left', max_bond=max_bond,
                                          svd_cutoff=svd_cutoff)

        if i < len(right_ops):
            op = right_ops[i]
            if op['matrix'] is not None:
                qs = op['qubits']
                if len(qs) == 1:
                    absorb_single_qubit_gate(mpo, qs[0], op['matrix'],
                                             side='right')
                elif len(qs) == 2 and abs(qs[0] - qs[1]) == 1:
                    absorb_two_qubit_gate(mpo, qs[0], qs[1], op['matrix'],
                                          side='right', max_bond=max_bond,
                                          svd_cutoff=svd_cutoff)

        if verbose and (i + 1) % 100 == 0:
            bonds = get_bond_dims(mpo)
            print(f"  [{i+1}] max_bond={max(bonds)}, elems={mpo_total_elements(mpo)}")

    return mpo
