"""Benchmark iterative vs closed-form sparse affine KR map solvers.

Example output:

Benchmarking optimize_sparse_affine_kr_map
Settings: n_samples=100
       p |   iterative (s) | closed-form (s) | speedup (x)
----------------------------------------------------------
    1000 |        0.414917 |        0.052610 |       7.89x
    9702 |        3.612242 |        0.498136 |       7.25x
   99452 |       39.885433 |        5.351585 |       7.45x

"""

from __future__ import annotations

import itertools
import time
from typing import Literal

import gstools as gs
import networkx as nx
import numpy as np
from graphite_maps.precision_estimation import optimize_sparse_affine_kr_map


def sample_grf_cube(n_samples: int, px: int, py: int, pz: int) -> np.ndarray:
    """Sample Gaussian random fields on a 3D structured grid."""
    model = gs.Matern(dim=3, var=1.0, len_scale=min(px, py, pz) / 4.0, nu=1.5)
    srf = gs.SRF(model)
    grid = [np.arange(px), np.arange(py), np.arange(pz)]

    U = np.empty((n_samples, px * py * pz), dtype=float)
    for i in range(n_samples):
        field = srf.structured(grid)
        U[i] = np.asarray(field, dtype=float).ravel()
    return U


def create_flattened_cube_graph(px: int, py: int, pz: int) -> nx.Graph:
    """Coped from ert.
    graph created with nodes numbered from 0 to px*py*pz
    corresponds to the "vectorization" or flattening of
    a 3D cube with shape (px,py,pz) in the same way as
    reshaping such a cube into a one-dimensional array.
    The indexing scheme used to create the graph reflects
    this flattening process"""

    G = nx.Graph()
    for x, y, z in itertools.product(range(px), range(py), range(pz)):
        # Flatten the 3D index to a single index
        index = x * py * pz + y * pz + z

        # Connect to the right neighbor (y-direction)
        if y < py - 1:
            G.add_edge(index, index + pz)

        # Connect to the bottom neighbor (x-direction)
        if x < px - 1:
            G.add_edge(index, index + py * pz)

        # Connect to the neighbor in front (z-direction)
        if z < pz - 1:
            G.add_edge(index, index + 1)

    return G


def run_single_solver(
    U: np.ndarray,
    graph: nx.Graph,
    solver: Literal["iterative", "closed_form"],
) -> float:
    """Run one solver and return seconds."""
    start = time.perf_counter()
    _ = optimize_sparse_affine_kr_map(
        U,
        graph,
        solver=solver,
        use_tqdm=False,
    )
    return time.perf_counter() - start


def main() -> None:
    grid_shapes = ((10, 10, 10), (21, 21, 22), (46, 46, 47))
    n_samples = 100

    print("Benchmarking optimize_sparse_affine_kr_map")
    print(f"Settings: n_samples={n_samples}")

    headers = ["p", "iterative (s)", "closed-form (s)", "speedup (x)"]
    print(f"{headers[0]:>8} | {headers[1]:>15} | {headers[2]:>15} | {headers[3]:>11}")
    print("-" * 58)

    speedups = []
    for px, py, pz in grid_shapes:
        p = px * py * pz
        U = sample_grf_cube(n_samples, px, py, pz)
        graph = create_flattened_cube_graph(px, py, pz)

        iter_time = run_single_solver(U, graph, "iterative")
        closed_time = run_single_solver(U, graph, "closed_form")
        speedup = iter_time / closed_time
        speedups.append(speedup)

        print(f"{p:8d} | {iter_time:15.6f} | {closed_time:15.6f} | {speedup:10.2f}x")

    print(
        "\nSummary: "
        f"closed-form is {min(speedups):.2f}x to {max(speedups):.2f}x faster "
        "than iterative on these cases."
    )


if __name__ == "__main__":
    main()
