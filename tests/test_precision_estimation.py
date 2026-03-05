import networkx as nx
import numpy as np
from graphite_maps import precision_estimation as precest
from scipy.linalg import det
from scipy.sparse import diags


def test_precision_graph_conversion():
    # Create a sparse symmetric matrix filled with ones
    p = 4
    A = diags(
        [np.repeat(1, p - 1), np.repeat(1, p), np.repeat(1, p - 1)],
        [-1, 0, 1],
        shape=(p, p),
        format="csc",
    )

    # Convert to graph and back to matrix
    G = precest.precision_to_graph(A)
    A_converted = precest.graph_to_precision_matrix(G)

    # Check if the original and final matrices are the same
    assert np.array_equal(A.toarray(), A_converted.toarray())


def test_gershgorin_spd_adjustment():
    # SND: -AR-1(phi) precision, p=odd
    p = 5  # Important to be odd, relating to determinant properties
    phi = 0.3
    A = diags(
        [
            np.repeat(-phi, p - 1),
            np.concatenate(([1.0], np.repeat(1.0 + phi**2, p - 2), [1.0])),
            np.repeat(-phi, p - 1),
        ],
        [-1, 0, 1],
        shape=(p, p),
        format="csc",
    )
    A = -A  # For p odd det(-A) = -det(A).

    # Check A is SND (det(A) should be < 0)
    assert det(A.toarray()) < 0, "A is negative definite"

    # Apply the Gershgorin adjustment
    A_adjusted = precest.gershgorin_spd_adjustment(A)

    # Check if A_adjusted is SPD
    assert det(A_adjusted.toarray()) > 0, "A_adjusted is positive definite"


def test_closed_form_matches_iterative_and_is_faster():
    import time

    rng = np.random.default_rng(42)
    n, p = 200, 50
    U = rng.standard_normal((n, p))

    # Banded sparsity pattern (bandwidth 3)
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(1, p):
        for j in range(max(0, i - 3), i):
            G.add_edge(i, j)

    start = time.perf_counter()
    C_iter = precest.optimize_sparse_affine_kr_map(
        U,
        G,
        solver="iterative",
        use_tqdm=False,
    )
    t_iter = time.perf_counter() - start

    start = time.perf_counter()
    C_closed = precest.optimize_sparse_affine_kr_map(
        U,
        G,
        solver="closed_form",
        use_tqdm=False,
    )
    t_closed = time.perf_counter() - start

    assert np.allclose(C_iter.toarray(), C_closed.toarray(), atol=1e-6, rtol=1e-6)
    assert t_closed < t_iter, (
        f"closed_form ({t_closed:.3f}s) should be faster than iterative ({t_iter:.3f}s)"
    )
