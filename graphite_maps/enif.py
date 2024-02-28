from typing import Optional
import numpy as np
from scipy.sparse import spmatrix, diags
from sksparse.cholmod import cholesky
import networkx as nx

from .precision_estimation import fit_precision_cholesky
from .linear_regression import (
    linear_l1_regression,
    linear_boost_ic_regression,
    response_residual,
    residual_variance,
)


def generate_gaussian_noise(n: int, Prec_eps: spmatrix) -> np.ndarray:
    """
    Generates 'n' samples of Gaussian noise with precision 'Prec_eps'.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    Prec_eps : scipy.sparse.spmatrix
        The precision matrix for the Gaussian noise, assumed to be sparse.

    Returns
    -------
    np.ndarray
        The Gaussian noise array of shape (n, m), where Prec_eps has shape (m, m).
    """

    m = Prec_eps.shape[0]
    standard_normal_samples = np.random.normal(size=(n, m))
    cholesky_factor = cholesky(Prec_eps)

    # Transform the samples using the inverse Cholesky factor
    # This transformation results in samples from N(0, Prec_eps^-1)
    eps = cholesky_factor.solve_A(standard_normal_samples.T).T

    assert eps.shape == (n, m), "Sampling returns wrong size"

    return eps


class EnIF:
    def __init__(
        self,
        *,
        Prec_u: Optional[spmatrix] = None,
        Graph_u: Optional[nx.Graph] = None,
        Prec_eps: Optional[spmatrix],
        H: Optional[spmatrix] = None,
    ) -> None:
        assert (
            Prec_u is not None or Graph_u is not None
        ), "Provide either Prec_u or Graph_u"

        self.Prec_u = Prec_u
        self.Graph_u = Graph_u
        self.Prec_eps = Prec_eps
        self.H = H
        self.unexplained_variance = None

    def fit(
        self,
        U: np.ndarray,
        Y: Optional[np.ndarray] = None,
        learning_algorithm: Optional[str] = "LASSO",
        verbose: bool = True,
    ) -> None:
        """
        Fit precision of u and (sparse) mapping H:u->y.
        """
        if self.Prec_u is None:
            self.fit_precision(U)
        elif verbose:
            print(
                "Precision u exists. Use `fit_precision` to refit if necessary"
            )
        if Y is not None:
            assert self.H is None, "Y should not be provided if H exists"
            self.fit_H(U=U, Y=Y, learning_algorithm=learning_algorithm)
        elif verbose:
            print("H mapping exists. Use `fit_H` to refit if necessary")

    def transport(
        self, U: np.ndarray, Y: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """
        Transport U from a sample from the prior to the posterior
        """
        n, _ = U.shape
        n_y, m = Y.shape
        assert n == n_y, "Number of ensembles must be the same"
        assert d.shape == (m,), "Observations must match responses"

        # Map parameters to canonical parametrization
        canonical = self.pushforward_to_canonical(U)

        # Work out residuals and associate unexplained variance
        residual = response_residual(U, Y, self.H)
        eps = generate_gaussian_noise(n, self.Prec_eps)
        residual_noisy = residual + eps
        if self.unexplained_variance is None:
            self.unexplained_variance = residual_variance(U, Y, self.H)

        # Update in canonical parametrization
        canonical_updated = self.update_canonical(canonical, residual_noisy, d)

        # Bring realizations back
        return self.pullback_from_canonical(canonical_updated)

    # Low-level API methods
    def fit_precision(self, U: np.ndarray) -> None:
        """
        Estimate self.Prec_u from data U w.r.t. graph self.Graph_u
        """
        self.Prec_u = fit_precision_cholesky(U, self.Graph_u)

    def fit_H(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        learning_algorithm: Optional[str] = "LASSO",
    ) -> None:
        """
        Estimate H from data U using (sparse) linear regression
        """
        if learning_algorithm == "LASSO":
            self.H = linear_l1_regression(U, Y)
        elif learning_algorithm == "influence-boost":
            self.H = linear_boost_ic_regression(U, Y)
        else:
            raise ValueError(
                f"Argument `learning_algorithm` must be a valid type. "
                f"Got: {learning_algorithm}"
            )
        self.unexplained_variance = residual_variance(U, Y, self.H)

    def pushforward_to_canonical(self, U: np.ndarray) -> np.ndarray:
        """
        Map each realization u in U to canonical space eta = Prec * u
        """
        assert self.Prec_u is not None, "Precision must exist to pushforward"
        Eta = (self.Prec_u @ U.T).T
        assert Eta.shape == U.shape, "Eta preserves the shape of U"
        return Eta

    @property
    def Prec_residual_noisy(self) -> spmatrix:

        if self.Prec_eps is None:
            raise ValueError("Prec_eps is not set.")

        eps_prec_diag = self.Prec_eps.diagonal()
        eps_variances = 1.0 / eps_prec_diag
        residual_noisy_variances = self.unexplained_variance + eps_variances
        Prec_r = diags(1.0 / residual_noisy_variances, 0)
        assert (
            Prec_r.shape == self.Prec_eps.shape
        ), "Residuals and noise precision should have same shape"
        return Prec_r

    def update_canonical(
        self, canonical: np.ndarray, residual_noisy: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """
        Use information-filter equations to update (eta, Prec) using perturbed d
        """
        assert self.H is not None, "H must be provided of fitted"
        assert self.Prec_u is not None, "Precision must be provided of fitted"

        n, p = canonical.shape
        n_r, m = residual_noisy.shape
        assert n == n_r, "canonical and residual_noisy must have equal samples"
        assert d.shape == (
            m,
        ), "d and residual_noisy must have matching dimension"

        updated_canonical = np.empty((n, p))
        Prec_r = self.Prec_residual_noisy
        for i in range(n):
            d_adjusted = d - residual_noisy[i, :]
            updated_canonical[i, :] = (
                canonical[i, :] + self.H.T @ Prec_r @ d_adjusted
            )

        # posterior precision
        self.Prec_u = self.Prec_u + self.H.T @ self.Prec_eps @ self.H

        return updated_canonical

    def pullback_from_canonical(
        self, updated_canonical: np.ndarray
    ) -> np.ndarray:
        """
        Solve u = Prec * eta
        """
        # Compute sparse Cholesky factorization
        factor = cholesky(self.Prec_u)
        # Use the Cholesky factor to solve u = Prec * eta
        updated_moment = factor.solve_A(updated_canonical.T).T

        return updated_moment
