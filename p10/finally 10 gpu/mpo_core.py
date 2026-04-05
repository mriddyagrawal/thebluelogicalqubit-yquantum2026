"""
MPO Core Primitives
=====================
Foundation: identity MPO, gate absorption, MPS conversion, peak extraction.

Convention:
    mpo[i].shape = (bond_left, phys_out, phys_in, bond_right)
    All physical dimensions are 2.

GPU: Uses CuPy when available, NumPy otherwise.
"""

from gpu_backend import xp, np_cpu, to_numpy
from copy import deepcopy


# ══════════════════════════════════════════════════
# MPO CONSTRUCTION
# ══════════════════════════════════════════════════

def identity_mpo(n_qubits, dtype=None):
    if dtype is None:
        dtype = xp.complex128
    mpo = []
    for i in range(n_qubits):
        T = xp.zeros((1, 2, 2, 1), dtype=dtype)
        T[0, 0, 0, 0] = 1.0
        T[0, 1, 1, 0] = 1.0
        mpo.append(T)
    return mpo


# ══════════════════════════════════════════════════
# GATE ABSORPTION
# ══════════════════════════════════════════════════

def absorb_single_qubit_gate(mpo, qubit, gate_matrix, side='right'):
    G = xp.array(gate_matrix, dtype=mpo[qubit].dtype)
    if side == 'right':
        mpo[qubit] = xp.einsum('ab,lbpr->lapr', G, mpo[qubit])
    elif side == 'left':
        mpo[qubit] = xp.einsum('lpkr,ka->lpar', mpo[qubit], G)
    else:
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")


def absorb_two_qubit_gate(mpo, q1, q2, gate_matrix, side='right',
                          max_bond=256, svd_cutoff=1e-10):
    assert abs(q1 - q2) == 1, f"Qubits must be adjacent: {q1}, {q2}"
    G = xp.array(gate_matrix, dtype=mpo[q1].dtype)

    if q1 > q2:
        q1, q2 = q2, q1
        SWAP = xp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=G.dtype)
        G = SWAP @ G @ SWAP

    T1 = mpo[q1]
    T2 = mpo[q2]
    bL = T1.shape[0]
    bR = T2.shape[3]

    merged = xp.einsum('abcd,defg->abcefg', T1, T2)
    G4 = G.reshape(2, 2, 2, 2)

    if side == 'right':
        result = xp.einsum('PQbe,abcefg->aPcQfg', G4, merged)
    elif side == 'left':
        result = xp.einsum('abcefg,cfPQ->abPeQg', merged, G4)
    else:
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    mat = result.reshape(bL * 2 * 2, 2 * 2 * bR)

    norm = xp.linalg.norm(mat)
    if float(norm) < 1e-30 or not bool(xp.isfinite(norm)):
        mpo[q1] = xp.zeros((bL, 2, 2, 1), dtype=mpo[q1].dtype)
        mpo[q2] = xp.zeros((1, 2, 2, bR), dtype=mpo[q2].dtype)
        return

    mat = mat / norm

    try:
        U, S, Vh = xp.linalg.svd(mat, full_matrices=False)
    except Exception:
        mpo[q1] = xp.zeros((bL, 2, 2, 1), dtype=mpo[q1].dtype)
        mpo[q2] = xp.zeros((1, 2, 2, bR), dtype=mpo[q2].dtype)
        return

    S_trunc = S[S > svd_cutoff]
    keep = min(len(S_trunc), max_bond)
    if keep == 0:
        keep = 1

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    U = U * S[None, :]

    mpo[q1] = U.reshape(bL, 2, 2, keep)
    mpo[q2] = Vh.reshape(keep, 2, 2, bR)


# ══════════════════════════════════════════════════
# MPO INFO
# ══════════════════════════════════════════════════

def get_bond_dims(mpo):
    return [mpo[i].shape[3] for i in range(len(mpo) - 1)]

def mpo_total_elements(mpo):
    return sum(t.size for t in mpo)


# ══════════════════════════════════════════════════
# MPS CONVERSION + CANONICALIZATION
# ══════════════════════════════════════════════════

def mpo_to_mps(mpo):
    mps = []
    for T in mpo:
        zero_in = xp.array([1.0, 0.0], dtype=T.dtype)
        M = xp.einsum('lpir,i->lpr', T, zero_in)
        mps.append(M)
    return canonicalize_mps(mps)


def canonicalize_mps(mps):
    """Right-to-left SVD sweep for left-canonical form."""
    n = len(mps)
    canon = [t.copy() for t in mps]

    for i in range(n - 1, 0, -1):
        T = canon[i]
        bL, d, bR = T.shape
        mat = T.reshape(bL, d * bR)
        U, S, Vh = xp.linalg.svd(mat, full_matrices=False)

        new_bL = len(S)
        canon[i] = Vh.reshape(new_bL, d, bR)

        US = U * S[None, :]
        canon[i - 1] = xp.einsum('abc,ck->abk', canon[i - 1], US)

    return canon


# ══════════════════════════════════════════════════
# PEAK EXTRACTION
# ══════════════════════════════════════════════════

def extract_peak_greedy(mps):
    n = len(mps)
    bitstring = []
    left_env = xp.array([1.0], dtype=mps[0].dtype)

    for i in range(n):
        T = mps[i]
        contracted = xp.einsum('b,bpr->pr', left_env, T)
        probs = xp.array([float(xp.sum(xp.abs(contracted[b, :])**2)) for b in range(2)])
        total = float(xp.sum(probs))
        if total > 0:
            probs = probs / total

        bit = int(xp.argmax(probs))
        bitstring.append(str(bit))
        left_env = contracted[bit, :]

        env_norm = float(xp.linalg.norm(left_env))
        if env_norm > 0:
            left_env = left_env / env_norm

    return ''.join(bitstring)


def sample_mps(mps, n_samples=1000, seed=42):
    """Sample bitstrings from a canonicalized MPS."""
    rng = np_cpu.random.default_rng(seed)  # RNG always on CPU
    n = len(mps)
    samples = []

    for _ in range(n_samples):
        bitstring = []
        left_env = xp.array([1.0], dtype=mps[0].dtype)

        for i in range(n):
            T = mps[i]
            contracted = xp.einsum('b,bpr->pr', left_env, T)
            probs = xp.array([float(xp.sum(xp.abs(contracted[b, :])**2))
                              for b in range(2)])
            total = float(xp.sum(probs))
            if total > 0:
                probs = probs / total

            p0 = float(probs[0])
            bit = 0 if rng.random() < p0 else 1
            bitstring.append(str(bit))
            left_env = contracted[bit, :]

            env_norm = float(xp.linalg.norm(left_env))
            if env_norm > 0:
                left_env = left_env / env_norm

        samples.append(''.join(bitstring))

    return samples
