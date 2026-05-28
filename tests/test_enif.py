import logging
import sys

import networkx as nx
import numpy as np
import pytest
import scipy as sp
from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import linear_boost_ic_regression
from graphite_maps.precision_estimation import (
    fit_precision_cholesky_approximate,
)


def rar1(T, phi, rng=None):
    """simulate auto-regressive-1.
    The first element is simulated from stationary distribution.
    """
    if phi <= -1 or phi >= 1:
        raise ValueError(
            "rar1 requires a stationary AR(1): expected phi in range (-1, 1)"
        )
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty([T])
    x[0] = rng.normal(0, 1 / np.sqrt(1 - phi**2))
    for i in range(1, T):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    return x


def nrar1(n, p, phi):
    """Samples n realizations of Ar-1(phi) of size p"""
    rng = np.random.default_rng(42)
    U = np.array([rar1(T=p, phi=phi, rng=rng) for _ in range(n)])
    return U


def create_ar1_precision(p, phi):
    Prec_u = sp.sparse.diags_array(
        [
            np.repeat(-phi, p - 1),
            np.concatenate(([1.0], np.repeat(1.0 + phi**2, p - 2), [1.0])),
            np.repeat(-phi, p - 1),
        ],
        offsets=[-1, 0, 1],
        shape=(p, p),
        format="csc",
    )
    return Prec_u


def test_snapshot_highlevel():
    """This snapshot test is meant to altert us if behavoir changes.
    If this is intended, then simply update the values below."""

    # Smoketest that logging works
    log = logging.getLogger("graphite_maps")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(stream=sys.stdout))

    # Size of the problem
    rng = np.random.default_rng(42)
    n_params = 10
    n_responses = 5
    n_ensemble = 25

    # Create random data
    Graph_u = nx.binomial_graph(n_params, p=0.5, seed=42)
    Prec_eps = sp.sparse.csc_array(np.diag(np.logspace(-3, 3, num=n_responses)))
    H = sp.sparse.csc_array(rng.normal(size=(n_responses, n_params)))
    U = rng.normal(size=(n_ensemble, n_params))
    Y = U @ H.T
    d = np.mean(Y, axis=0)

    # Call the high-level API, using "Graph_u" instead of "Prec_u"
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, ordering_method="natural")
    U_posterior = gtmap.transport(U, Y, d, seed=42)

    # Check result
    desired = np.array([0.01311213, -0.40208624, -0.49277274, -0.29639235, -1.84713641])
    np.testing.assert_allclose(np.diag(U_posterior)[:5], desired, rtol=1e-6)

    # Call the high-level API, using "Prec_u"
    Prec_u = rng.normal(size=(n_params, n_params))
    Prec_u = Prec_u.T + Prec_u
    shift = np.abs(np.linalg.eigvalsh(Prec_u).min()) + 1.0
    Prec_u = Prec_u + shift * np.eye(n_params)
    assert np.linalg.eigvalsh(Prec_u).min() > 0

    gtmap = EnIF(Prec_u=sp.sparse.csc_array(Prec_u), Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, ordering_method="natural")
    U_posterior = gtmap.transport(U, Y, d, seed=42)

    desired = np.array(
        [-0.03747837, -0.35764201, -0.78659938, -0.26586081, -1.53651656]
    )
    np.testing.assert_allclose(np.diag(U_posterior)[:5], desired, rtol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_affine_invariance(seed):
    rng = np.random.default_rng(seed)
    n_params, n_responses, n_ensemble = 50, 25, 10

    # Create data
    Graph_u = nx.binomial_graph(n_params, p=0.2, seed=42)
    Prec_eps = sp.sparse.csc_array(np.diag(np.logspace(-2, 2, num=n_responses)))
    H_true = sp.sparse.csc_array(rng.normal(size=(n_responses, n_params)))
    U = rng.normal(size=(n_ensemble, n_params)) * 3 + 7
    Y = U @ H_true.T
    d = np.mean(Y, axis=0)

    def run(U_in):
        # Estimate H
        H = linear_boost_ic_regression(U=U_in, Y=Y)
        # Estimate precision matrix
        Prec_u = fit_precision_cholesky_approximate(
            U=U_in,
            Graph_u=Graph_u,
            use_tqdm=False,
        )
        enif = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
        idx = enif.get_update_indices(neighbor_propagation_order=15)
        return enif.transport(U=U_in, Y=Y, d=d, update_indices=idx, seed=seed)

    # Posterior parameters with no transformation
    X_raw = run(U)

    # Standardize parameters, run EnIF, then transform back
    mu = U.mean(axis=0, keepdims=True)
    sigma = U.std(axis=0, keepdims=True)
    X_std = run((U - mu) / sigma) * sigma + mu

    # Verify that we get the same answer
    np.testing.assert_allclose(X_raw, X_std)


def test_snapshot_lowlevel():
    """Test calling the API using some low level functions.
    This mimics the usage in:
    https://github.com/equinor/ert/blob/da04bf5924c4f0cebee0bf5d941f0e5c2acc8876/src/ert/analysis/_enif_update.py
    """

    # Size of the problem
    rng = np.random.default_rng(42)
    n_params = 10
    n_responses = 5
    n_ensemble = 25

    # Create random data
    Graph_u = nx.binomial_graph(n_params, p=0.5, seed=42)
    Prec_eps = sp.sparse.csc_array(np.diag(np.logspace(-3, 3, num=n_responses)))
    H = sp.sparse.csc_array(rng.normal(size=(n_responses, n_params)))
    U = rng.normal(size=(n_ensemble, n_params))
    Y = U @ H.T
    d = np.mean(Y, axis=0)

    # Call fit: Learn sparse linear map only
    H = linear_boost_ic_regression(
        U=U,
        Y=Y,
    )

    desired_H = np.array([0.0, 0.74667254, 1.21359438, 0.0, 0.0])
    np.testing.assert_allclose(np.diag(H.todense())[:5], desired_H, rtol=1e-6)

    # Estimate precision matrix
    Prec_u = fit_precision_cholesky_approximate(
        U=U,
        Graph_u=Graph_u,
        neighbourhood_expansion=2,
        use_tqdm=True,
    )

    desired_Prec_u = np.array([1.707219, 1.779207, 1.171189, 1.042243, 1.224324])
    np.testing.assert_allclose(np.diag(Prec_u.todense())[:5], desired_Prec_u, rtol=1e-6)

    # Initialize EnIF object with full precision matrices
    gtmap = EnIF(
        Prec_u=Prec_u,
        Prec_eps=Prec_eps,
        H=H,
    )

    update_indices = gtmap.get_update_indices(
        neighbor_propagation_order=15,
    )

    X_updated = gtmap.transport(
        U=U,
        Y=Y,
        d=d,
        update_indices=update_indices,
        iterative=False,
        seed=13,
    )

    desired_X_updated = np.array(
        [-0.19630398, -0.27700102, -0.87434721, 0.03307433, -1.60668824]
    )
    np.testing.assert_allclose(np.diag(X_updated)[:5], desired_X_updated, rtol=1e-6)


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_posterior_low_level_api_equals_high_level_api(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # AR(1) conditional-independence graph: u[i] only depends on its immediate neighbors u[i-1], u[i+1]
    Graph_u = nx.path_graph(p)

    # Specify observations and associate uncertainty, and a linear map H
    d = np.array([30.0])
    sd_eps = 1
    H = np.zeros((1, p))
    H[0, p // 2] = 1  # observe the middle component of the state
    H = sp.sparse.csc_array(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_array(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U)
    U_posterior_highlevel = gtmap.transport(U, Y, d, seed=42)

    # EnIF low-level API
    gtmap_lowlevel = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap_lowlevel.fit_precision(U)
    canonical = gtmap_lowlevel.pushforward_to_canonical(U)
    # Work out residuals and associate unexplained variance
    residual = gtmap_lowlevel.response_residual(U, Y)
    eps = gtmap_lowlevel.generate_observation_noise(n, seed=42)
    residual_noisy = residual + eps
    canonical_posterior = gtmap_lowlevel.update_canonical(canonical, residual_noisy, d)
    U_posterior_lowlevel = gtmap_lowlevel.pullback_from_canonical(
        canonical_posterior, U_prior=U
    )

    assert np.allclose(U_posterior_lowlevel, U_posterior_highlevel, atol=1e-6)


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_enif_equals_kalman_under_exact_precision_and_H(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the precision
    Prec_u = create_ar1_precision(p, phi)

    # Specify observations and associate uncertainty, and a linear map H
    d = np.array([30.0])
    sd_eps = 1
    H = np.zeros((1, p))
    H[0, p // 2] = 1  # observe the middle component of the state
    H = sp.sparse.csc_array(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_array(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API with known precision
    gtmap = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U)
    U_posterior_enif = gtmap.transport(U, Y, d, seed=42)

    # Create Kalman update -- use same noise
    eps = gtmap.generate_observation_noise(n, seed=42)
    Sigma_u = np.linalg.inv(Prec_u.toarray())
    Sigma_d = H @ Sigma_u @ H.T + np.linalg.inv(Prec_eps.toarray())
    K = Sigma_u @ H.T @ np.linalg.inv(Sigma_d)
    U_posterior_enkf = np.empty_like(U)
    for i in range(n):
        innovation = d - Y[i, 0] - eps[i, 0]  # scalar
        U_posterior_enkf[i] = U[i] + (K @ innovation).ravel()

    assert np.allclose(U_posterior_enif, U_posterior_enkf, atol=1e-12)


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_pullback_of_pushforward_equals_input(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the precision
    Prec_u = create_ar1_precision(p, phi)

    # Specify observations and associate uncertainty, and a linear map H
    sd_eps = 1
    H = np.zeros((1, p))
    H[0, p // 2] = 1  # observe the middle component of the state
    H = sp.sparse.csc_array(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_array(Prec_eps)

    # Notice: No update in canonical space
    gtmap_pullpush = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    canonical = gtmap_pullpush.pushforward_to_canonical(U)
    U_posterior = gtmap_pullpush.pullback_from_canonical(canonical)

    # Due to no update, we should have equality
    assert np.allclose(U, U_posterior, atol=1e-12)


def test_that_iterative_pullback_matches_cholesky():
    n = 100
    p = 500
    phi = 0.5
    U = nrar1(n, p, phi)
    Prec_u = create_ar1_precision(p, phi)

    d = np.array([30.0])
    sd_eps = 1
    H = np.zeros((1, p))
    H[0, p // 2] = 1
    H = sp.sparse.csc_array(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_array(Prec_eps)
    Y = U @ H.T

    # Pushforward and update so the pullback solves a non-trivial posterior system
    gtmap = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    canonical = gtmap.pushforward_to_canonical(U)
    residual = gtmap.response_residual(U, Y)
    eps = gtmap.generate_observation_noise(n, seed=42)
    canonical_posterior = gtmap.update_canonical(canonical, residual + eps, d)

    # Full solve: both paths should agree
    U_chol = gtmap.pullback_from_canonical(canonical_posterior, iterative=False)
    U_iter = gtmap.pullback_from_canonical(canonical_posterior, iterative=True)
    assert np.allclose(U_chol, U_iter, atol=1e-3)

    # Partial solve: only update the middle half of the state
    update_indices = np.arange(p // 4, 3 * p // 4)
    U_chol_partial = gtmap.pullback_from_canonical(
        canonical_posterior, update_indices=update_indices, U_prior=U, iterative=False
    )
    U_iter_partial = gtmap.pullback_from_canonical(
        canonical_posterior, update_indices=update_indices, U_prior=U, iterative=True
    )
    assert np.allclose(U_chol_partial, U_iter_partial, atol=1e-3)


@pytest.mark.parametrize("algo", ["LASSO", "influence-boost"])
def test_that_fit_attributes_round_trip_through_reconstruction(algo):
    n, p, m, phi = 50, 80, 4, 0.5
    U = nrar1(n, p, phi)

    rng = np.random.default_rng(0)
    H_true = np.zeros((m, p))
    for j in range(m):
        H_true[j, rng.integers(0, p)] = 1.0
    Y = U @ H_true.T

    Prec_eps = sp.sparse.csc_array(np.eye(m))
    Graph_u = nx.path_graph(p)

    trainer = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)
    trainer.fit(U, Y=Y, learning_algorithm=algo)
    assert trainer.H.nnz > 0, f"fit({algo}) produced an all-zero H"

    fresh = EnIF(
        Prec_u=trainer.Prec_u.copy(),
        Prec_eps=trainer.Prec_eps.copy(),
        H=trainer.H.copy(),
    )
    fresh.unexplained_variance = trainer.unexplained_variance.copy()

    d = np.zeros(m)
    trainer_post = trainer.transport(U, Y, d, seed=42)
    fresh_post = fresh.transport(U, Y, d, seed=42)

    assert np.allclose(fresh_post, trainer_post, atol=1e-12), (
        "Reconstructed EnIF produced a different posterior than the trainer"
    )
    assert not np.allclose(trainer_post, U), (
        "transport() returned the prior unchanged, H had no effect"
    )


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
