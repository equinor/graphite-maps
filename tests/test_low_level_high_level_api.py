import numpy as np

np.set_printoptions(precision=3, suppress=True)
import pytest
import scipy as sp
from graphite_maps.enif import EnIF
from graphite_maps.precision_estimation import precision_to_graph


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


def create_ar1_graph(p):
    phi = 0.3

    # create AR-1 precision matrix
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

    # create corresponding graph -- often we only know this
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
