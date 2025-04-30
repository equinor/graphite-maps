"""
reverse_maximin.py
==================

Efficient implementation of the Reverse Maximin Ordering and Sparsity Pattern Construction
from Schäfer et al. (2021), "Sparse Cholesky Factorization by Kullback-Leibler Minimization".

This module computes a sparsity-preserving, reverse-maximin ordering of variables
suitable for fast and scalable approximate Cholesky factorization.

The intuition of the algorithm is found in Section 3 of the reference paper.
The implementation follows Algorithm C.1 from Appendix C.
The method is linear in space and near-linear in time (O(nlog^(2d)n)).

References:
- F. Schäfer, M. Katzfuss, and H. Owhadi, 2021.
  "Sparse Cholesky Factorization by Kullback-Leibler Minimization".
  https://arxiv.org/abs/2004.14455
"""

from collections import defaultdict

import numba as nb
import numpy as np
from heapdict import heapdict


# ---------------------------------------------------------------------
# 1.  Numba helpers (L1 / Manhattan metric only for now)
# To-do: Include more metrics.
# ---------------------------------------------------------------------
@nb.njit(inline="always")
def _manhattan(p: np.ndarray, q: np.ndarray) -> float:
    """L1 or Manhattan distance"""
    s = 0.0
    for i in range(len(p)):
        s += abs(p[i] - q[i])
    return s


@nb.njit
def _argsort_by_manhat(
    xk: np.ndarray, idx: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """Return the indices `idx` sorted by L1-distance to `xk`."""
    d = np.empty(len(idx), dtype=np.float64)
    for t, j in enumerate(idx):
        d[t] = _manhattan(xk, coords[j])
    order = np.argsort(d)
    return idx[order]


# ---------------------------------------------------------------------


def reverse_maxmin_ordering(
    grid_points: np.ndarray,
    dist_to_boundary: np.ndarray,
    rho: float = 2.0,
    verbose: bool = False,
) -> tuple[list[int], dict[int, set[int]]]:
    """
    Reverse-maximin ordering from Appendix C.1
    """
    N = len(grid_points)
    coords = grid_points.astype(np.float64, copy=False)

    # ------------------ state ----------------------------------------
    l = dist_to_boundary.copy()
    selected = np.zeros(N, dtype=np.bool_)
    not_selected = set(range(N))
    ordering_idx = []
    children = defaultdict(list)
    parents = defaultdict(list)
    children_sorted = np.zeros(N, dtype=np.bool_)

    heap = heapdict({i: -l_i for i, l_i in enumerate(l)})

    # ------------------ first (seed) node ----------------------------
    i0, neg_li = heap.popitem()
    selected[i0] = True
    not_selected.remove(i0)

    xi = coords[i0]
    li = -neg_li
    ordering_idx.append(i0)

    # everybody is initially child of i0
    for j in range(N):
        parents[j].append(i0)
        children[i0].append(j)
        if j == i0:
            continue
        dij = _manhattan(xi, coords[j])
        if dij < l[j]:
            l[j] = dij
            heap[j] = -dij

    l_trunc = li  # current truncation radius

    # ------------------ main loop -----------------------------------
    while heap:
        i, neg_li = heap.popitem()
        if selected[i]:
            continue
        li = -neg_li
        xi = coords[i]
        selected[i] = True
        not_selected.remove(i)
        ordering_idx.append(i)

        # ---- choose parent k (Alg. C.1 lines 23-32) ----------------
        k = i0
        distik = np.inf
        for j in parents[i]:
            dij = _manhattan(xi, coords[j])
            if ((j == i0) or (dij + rho * li <= rho * min(l[j], l_trunc))) and (
                dij < distik
            ):
                distik = dij
                k = j

        xk = coords[k]

        # ---- ensure children[k] sorted once ------------------------
        # Slightly different form paper implementation
        # We sort children of k (once) only if the parent is selected as k
        if not children_sorted[k]:
            arr = np.asarray(children[k], dtype=np.int64)
            children[k] = _argsort_by_manhat(xk, arr, coords).tolist()
            children_sorted[k] = True

        r_li = rho * li  # only once
        for j in children[k]:
            if j == i:
                continue
            djk = _manhattan(coords[j], xk)
            if djk > distik + r_li:  # triangle inequality pruning
                break

            dij = _manhattan(xi, coords[j])

            if (not selected[j]) and dij < l[j]:
                l[j] = dij
                heap[j] = -dij

            if dij <= r_li:
                children[i].append(j)
                parents[j].append(i)

        # ---- truncation test (lines 43-49) ----
        # This ensures linear space complexity
        # Modification of reference algorithm Schäfer 2020 Algorithm 4.1
        cond = True
        half_ltr = 0.5 * l_trunc
        for j in not_selected:
            if _manhattan(xi, coords[j]) >= half_ltr:
                cond = False
                break

        if cond:
            l_trunc *= 0.5
            rho_ltr = rho * l_trunc
            keep = []
            for j in children[i]:
                if (j not in not_selected) or (_manhattan(xi, coords[j]) <= rho_ltr):
                    keep.append(j)
                else:
                    parents[j].remove(i)
            children[i] = keep

    # --------------- build sparsity sets ----------------------------
    sparsity = defaultdict(set)
    for i, kids in children.items():
        sparsity[i].update(kids)
        sparsity[i].add(i)

    return ordering_idx, sparsity
