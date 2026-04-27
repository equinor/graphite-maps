import numpy as np
import pytest
import scipy as sp
from graphite_maps import precision_estimation as precest
from scipy.sparse import diags


def test_objective_twice():
    # A regression test: ensure that two calls return the same result.
    rng = np.random.default_rng(42)

    C_k = np.exp(rng.normal(0, 0.1, size=5))
    U = rng.normal(size=(5, 5))

    value1 = precest.objective_function(C_k, U)
    value2 = precest.objective_function(C_k, U)
    np.testing.assert_allclose(value1, value2)

    # Check gradient
    rmse = sp.optimize.check_grad(
        precest.objective_function,
        precest.gradient,
        np.array([1, 2, 3, 4, 4.5]),
        U,
        rng=rng,
    )
    assert rmse <= 0.002


def test_precision_graph_conversion():
    # Create a sparse symmetric matrix filled with ones
    p = 4
    A = diags(
        [np.repeat(1, p - 1), np.repeat(1, p), np.repeat(1, p - 1)],
        [-1, 0, 1],
        shape=(p, p),
        format="csc",
        dtype=np.float64,
    )

    # Convert to graph and back to matrix
    G = precest.precision_to_graph(A)
    A_converted = precest.graph_to_precision_matrix(G)

    # Check if the original and final matrices are the same
    assert np.array_equal(A.toarray(), A_converted.toarray())


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
