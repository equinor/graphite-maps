import numpy as np
import pytest
import scipy as sp
from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import (
    linear_boost_ic_regression,
    residual_variance,
)
from graphite_maps.precision_estimation import (
    fit_precision_cholesky_approximate,
    precision_to_graph,
)


# Simulate data
def rar1(T, phi, rng=None):
    """simulate auto-regressive-1.
    The first element is simulated from stationary distribution.
    """
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
    Prec_u = sp.sparse.diags(
        [
            np.repeat(-phi, p - 1),
            np.concatenate(([1.0], np.repeat(1.0 + phi**2, p - 2), [1.0])),
            np.repeat(-phi, p - 1),
        ],
        [-1, 0, 1],
        shape=(p, p),
        format="csc",
    )
    return Prec_u


def create_ar1_graph(p):
    # Graph created through precision matrix
    # Could be created directly
    phi = 0.3
    Prec_u = create_ar1_precision(p, phi)
    Graph_u = precision_to_graph(Prec_u)
    return Graph_u


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_posterior_low_level_api_equals_high_level_api(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the graph in a very inelegant way (potential for improvement)
    Graph_u = create_ar1_graph(p)

    # Specify observations and associate uncertainty, and a linear map H
    d = np.array([30.0])
    sd_eps = 1
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, verbose_level=4)
    U_posterior_highlevel = gtmap.transport(U, Y, d, seed=42, verbose_level=10)

    # EnIF low-level API
    gtmap_lowlevel = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap_lowlevel.fit_precision(U)
    if gtmap_lowlevel.H is None:
        gtmap_lowlevel.fit_H(U, U @ H.T)  # simulations Y = U@H.T
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
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API with known precision
    gtmap = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, verbose_level=4)
    U_posterior_enif = gtmap.transport(U, Y, d, seed=42, verbose_level=10)

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
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Notice: No update in canonical space
    gtmap_pullpush = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    canonical = gtmap_pullpush.pushforward_to_canonical(U)
    U_posterior = gtmap_pullpush.pullback_from_canonical(canonical)

    # Due to no update, we should have equality
    assert np.allclose(U, U_posterior, atol=1e-12)


def test_enif_mda_equals_single_update():
    """EnIF-MDA with alphas summing to 1 equals a single EnIF update (Gauss-linear).

    Uses exact, known parameters (Prec_u from the analytical AR(1) formula,
    H as a hand-picked row of the identity) with zero unexplained variance.
    This is a pure algebraic verification of the MDA identity under ideal
    conditions.
    """
    n, p, phi = 50, 30, 0.5
    U = nrar1(n, p, phi)
    Prec_u = create_ar1_precision(p, phi)

    d = np.array([30.0])
    H = sp.sparse.csc_matrix(np.eye(1, p, p // 2))
    Prec_eps = sp.sparse.csc_matrix(np.array([[1.0]]))

    alphas = [0.3, 0.5, 0.2]  # sum = 1

    # --- Noise control ---
    rng = np.random.default_rng(42)
    sd = np.sqrt(1.0 / Prec_eps.toarray()[0, 0])  # measurement std dev
    z_list = [rng.normal(0, sd, size=(n, 1)) for _ in alphas]  # base noise
    eps_mda = [z / np.sqrt(a) for z, a in zip(z_list, alphas, strict=True)]  # inflated
    eps_single = sum(a * e for a, e in zip(alphas, eps_mda, strict=True))  # equivalent

    # --- Single update ---
    enif = EnIF(Prec_u=Prec_u.copy(), Prec_eps=Prec_eps, H=H)
    enif.unexplained_variance = np.array([0.0])
    canonical = enif.pushforward_to_canonical(U)
    canonical = enif.update_canonical(canonical, eps_single, d)
    U_single = enif.pullback_from_canonical(canonical, U_prior=U)

    # --- MDA: multiple partial updates ---
    U_current = U.copy()
    Prec_u_current = Prec_u.copy()
    for alpha, eps_k in zip(alphas, eps_mda, strict=True):
        enif_k = EnIF(Prec_u=Prec_u_current, Prec_eps=alpha * Prec_eps, H=H)
        enif_k.unexplained_variance = np.array([0.0])
        canonical_k = enif_k.pushforward_to_canonical(U_current)
        canonical_k = enif_k.update_canonical(canonical_k, eps_k, d)
        U_current = enif_k.pullback_from_canonical(canonical_k, U_prior=U_current)
        Prec_u_current = enif_k.Prec_u

    assert np.allclose(U_single, U_current, atol=1e-10)


def test_enif_mda_with_fitted_H_and_approximate_precision():
    """EnIF-MDA equals single update using fitted H and approximate precision.

    Uses parameters estimated from data (H via linear_boost_ic_regression,
    Prec_u via fit_precision_cholesky_approximate) with nonzero
    unexplained_variance from the regression residuals. Verifies the MDA
    identity holds when using imperfect, data-driven estimates, and that
    unexplained_variance must also be scaled by 1/alpha in each MDA step.
    """
    n, p, phi = 200, 30, 0.5
    rng = np.random.default_rng(123)

    # --- Generate synthetic data ---
    U = nrar1(n, p, phi)
    H_true = sp.sparse.csc_matrix(np.eye(1, p, p // 2))
    obs_sd = 1.0
    Y = U @ H_true.T + rng.normal(0, obs_sd, size=(n, 1))
    d = np.array([30.0])
    Prec_eps = sp.sparse.csc_matrix(np.array([[1.0 / obs_sd**2]]))

    # --- Fit H using linear_boost_ic_regression ---
    H_fitted = linear_boost_ic_regression(U, Y)
    unexplained_var = residual_variance(U, Y, H_fitted)

    # --- Fit Prec_u using fit_precision_cholesky_approximate ---
    Graph_u = precision_to_graph(create_ar1_precision(p, phi))
    Prec_u_fitted = fit_precision_cholesky_approximate(U, Graph_u)

    # --- Noise control ---
    alphas = [0.3, 0.5, 0.2]  # sum = 1
    noise_rng = np.random.default_rng(99)
    sd_eps = np.sqrt(1.0 / Prec_eps.toarray()[0, 0])  # measurement std dev
    z_list = [noise_rng.normal(0, sd_eps, size=(n, 1)) for _ in alphas]  # base noise
    eps_mda = [z / np.sqrt(a) for z, a in zip(z_list, alphas, strict=True)]  # inflated
    eps_single = sum(a * e for a, e in zip(alphas, eps_mda, strict=True))  # equivalent

    # Residuals from fitted H
    residuals = Y - U @ H_fitted.T

    # --- Single update using transport-level steps ---
    enif = EnIF(Prec_u=Prec_u_fitted.copy(), Prec_eps=Prec_eps, H=H_fitted)
    enif.unexplained_variance = unexplained_var
    canonical = enif.pushforward_to_canonical(U)
    residual_noisy_single = residuals + eps_single
    canonical = enif.update_canonical(canonical, residual_noisy_single, d)
    U_single = enif.pullback_from_canonical(canonical, U_prior=U)

    # --- MDA: multiple partial updates ---
    # Scale both Prec_eps and unexplained_variance so that the total residual
    # covariance inflates by 1/alpha_k:  (unexp_var + eps_var) / alpha_k
    U_current = U.copy()
    Prec_u_current = Prec_u_fitted.copy()
    for alpha, eps_k in zip(alphas, eps_mda, strict=True):
        enif_k = EnIF(Prec_u=Prec_u_current, Prec_eps=alpha * Prec_eps, H=H_fitted)
        enif_k.unexplained_variance = unexplained_var / alpha
        # enif_k.unexplained_variance = unexplained_var  # No scaling of unexp var
        canonical_k = enif_k.pushforward_to_canonical(U_current)
        residual_noisy_k = residuals + eps_k
        canonical_k = enif_k.update_canonical(canonical_k, residual_noisy_k, d)
        U_current = enif_k.pullback_from_canonical(canonical_k, U_prior=U_current)
        Prec_u_current = enif_k.Prec_u

    assert np.allclose(U_single, U_current, atol=1e-10)
