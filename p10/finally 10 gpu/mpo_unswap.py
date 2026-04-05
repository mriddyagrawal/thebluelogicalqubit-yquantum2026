"""
MPO Unswapping — Greedy SWAP search
=====================================
Reduces MPO bond dimensions by extracting permutation structure.
M = P_L · M_tilde · P_R

Permutation conventions:
  P_L: left-composition — swap VALUES i,i+1 throughout the array
  P_R: right-composition — swap at POSITIONS i,i+1

GPU: All tensor operations through xp (CuPy or NumPy).
"""

from gpu_backend import xp, to_numpy
from mpo_core import get_bond_dims, mpo_total_elements


# ══════════════════════════════════════════════════
# SWAP PRIMITIVES
# ══════════════════════════════════════════════════

SWAP_GATE = xp.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=xp.complex128).reshape(2, 2, 2, 2)


def try_swap_at(mpo, i, side, svd_cutoff=1e-10, max_bond=None):
    """Try applying a SWAP at position (i, i+1) on the given side(s).
    Returns (new_bond, new_T1, new_T2) without modifying the original."""
    T1 = mpo[i]
    T2 = mpo[i + 1]
    bL = T1.shape[0]
    bR = T2.shape[3]

    merged = xp.einsum('abcd,defg->abcefg', T1, T2)

    if side == 'left' or side == 'both':
        merged = xp.einsum('abcefg,cfPQ->abPeQg', merged, SWAP_GATE)
    if side == 'right' or side == 'both':
        merged = xp.einsum('PQbe,abcefg->aPcQfg', SWAP_GATE, merged)

    mat = merged.reshape(bL * 2 * 2, 2 * 2 * bR)

    norm = xp.linalg.norm(mat)
    if float(norm) < 1e-30 or not bool(xp.isfinite(norm)):
        return 1, T1.copy(), T2.copy()

    mat_normed = mat / norm
    U, S, Vh = xp.linalg.svd(mat_normed, full_matrices=False)
    S = S * norm

    mask = S > svd_cutoff
    if max_bond and int(xp.sum(mask)) > max_bond:
        mask[max_bond:] = False
    bond = int(xp.sum(mask))
    if bond == 0:
        bond = 1
        mask[0] = True

    U = U[:, mask]
    S = S[mask]
    Vh = Vh[mask, :]
    U = U * S[None, :]

    new_T1 = U.reshape(bL, 2, 2, bond)
    new_T2 = Vh.reshape(bond, 2, 2, bR)

    return bond, new_T1, new_T2


# ══════════════════════════════════════════════════
# GREEDY UNSWAPPING
# ══════════════════════════════════════════════════

def unswap_mpo(mpo, svd_cutoff=1e-10, max_bond=None, verbose=False):
    """Greedy SWAP search to reduce MPO bond dimensions.

    Returns (mpo, P_L, P_R, stats) where:
      M_original = P_R_mat @ M_new @ P_L_mat
    """
    n = len(mpo)
    P_L = list(range(n))
    P_R = list(range(n))

    total_swaps = 0
    rounds = 0

    if verbose:
        bonds = get_bond_dims(mpo)
        elems = mpo_total_elements(mpo)
        print(f"Unswap start: bonds={bonds}, max={max(bonds)}, elems={elems}")

    while True:
        improved = False
        bonds = get_bond_dims(mpo)

        pairs = sorted(range(n - 1), key=lambda i: bonds[i], reverse=True)

        for i in pairs:
            current_bond = bonds[i]
            if current_bond <= 1:
                continue

            best_bond = current_bond
            best_side = None
            best_T1 = None
            best_T2 = None

            for side in ['left', 'right', 'both']:
                new_bond, new_T1, new_T2 = try_swap_at(
                    mpo, i, side, svd_cutoff=svd_cutoff, max_bond=max_bond)
                if new_bond < best_bond:
                    best_bond = new_bond
                    best_side = side
                    best_T1 = new_T1
                    best_T2 = new_T2

            if best_side is not None:
                mpo[i] = best_T1
                mpo[i + 1] = best_T2

                # P_L: left-composition (swap VALUES i,i+1)
                if best_side in ('left', 'both'):
                    for j in range(n):
                        if P_L[j] == i:
                            P_L[j] = i + 1
                        elif P_L[j] == i + 1:
                            P_L[j] = i
                # P_R: right-composition (swap POSITIONS i,i+1)
                if best_side in ('right', 'both'):
                    P_R[i], P_R[i + 1] = P_R[i + 1], P_R[i]

                total_swaps += 1
                improved = True

                if verbose:
                    bonds_new = get_bond_dims(mpo)
                    print(f"  swap {total_swaps}: pos=({i},{i+1}), "
                          f"side={best_side}, bond {current_bond}→{best_bond}, "
                          f"bonds={bonds_new}")

                # Update bonds for remaining pairs in this sweep
                bonds = get_bond_dims(mpo)

        rounds += 1
        if not improved:
            break

    if verbose:
        bonds = get_bond_dims(mpo)
        elems = mpo_total_elements(mpo)
        print(f"Unswap done: {total_swaps} swaps in {rounds} rounds, "
              f"bonds={bonds}, max={max(bonds)}, elems={elems}")

    stats = {'total_swaps': total_swaps, 'rounds': rounds}
    return mpo, P_L, P_R, stats
