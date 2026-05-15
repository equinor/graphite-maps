import networkx as nx
import numpy as np
import pytest
import scipy as sp
from graphite_maps import precision_estimation as precest
from scipy.optimize import minimize


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


def test_closed_form_matches_iterative_solver():
    """Closed-form row solver agrees with the iterative (L-BFGS-B) solution."""
    rng = np.random.default_rng(0)
    n, n_cols = 200, 5
    U_reduced = rng.normal(size=(n, n_cols))
    lambda_l2 = 2.0 * n_cols

    # Closed-form solution introduced in commit 864f430
    off_diag_cf, diag_cf = precest.solve_row_closed_form(U_reduced, lambda_l2)

    # Iterative reference solution (L-BFGS-B on the log-diagonal parametrisation)
    x0 = np.zeros(n_cols)
    res = minimize(
        fun=precest.objective_function,
        x0=x0,
        args=(U_reduced, lambda_l2),
        method="L-BFGS-B",
        jac=precest.gradient,
        tol=1e-12,
        options={"gtol": 1e-9},
    )
    off_diag_iter = res.x[:-1]
    diag_iter = np.exp(res.x[-1])

    np.testing.assert_allclose(off_diag_cf, off_diag_iter, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(diag_cf, diag_iter, rtol=1e-4)


def test_fit_precision_cholesky_approximate_two_hop_recovers_fill_in():
    rng = np.random.default_rng(42)
    n = 50000

    # The true precision is sparse on the 4-cycle, but its Cholesky factor in
    # the natural ordering contains fill-in: entries that become nonzero in
    # the factor even though the corresponding precision entry is zero. In
    # this example, C[2, 0] is such a fill-in term, needed so that C.T @ C
    # matches the true precision. A 1-hop neighborhood cannot represent that
    # term, while a 2-hop expansion can, so the 2-hop fit should be much more
    # accurate.
    prec_true = np.array(
        [
            [2.0, -0.4, 0.0, -0.3],
            [-0.4, 2.0, -0.5, 0.0],
            [0.0, -0.5, 2.0, -0.6],
            [-0.3, 0.0, -0.6, 2.0],
        ]
    )
    cov_true = np.linalg.inv(prec_true)
    U = rng.multivariate_normal(np.zeros(4), cov_true, size=n)
    G = nx.cycle_graph(4)

    prec_1_hop = precest.fit_precision_cholesky_approximate(
        U=U, G=G, neighbourhood_expansion=1, use_tqdm=False
    ).toarray()
    prec_2_hop = precest.fit_precision_cholesky_approximate(
        U=U, G=G, neighbourhood_expansion=2, use_tqdm=False
    ).toarray()

    # Without the 2-hop expansion, the solver cannot represent the fill-in and
    # leaks a spurious (0, 2) precision entry.
    assert abs(prec_1_hop[0, 2]) > 0.07
    assert abs(prec_2_hop[0, 2]) < 0.02

    err_1_hop = np.max(np.abs(prec_1_hop - prec_true))
    err_2_hop = np.max(np.abs(prec_2_hop - prec_true))
    assert err_1_hop > err_2_hop
    assert err_1_hop > 0.07
    assert err_2_hop < 0.03


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
