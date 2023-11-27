from typing import Optional
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve
import networkx as nx  # type: ignore

from .precision_estimation import fit_precision_cholesky
from .linear_regression import linear_l1_regression

from scipy.sparse.linalg import inv
from numpy.random import multivariate_normal


def perturb_d(d, Prec_eps):
    """
    Perturbs the array 'd' by adding Gaussian noise with precision 'Prec_eps'.

    Parameters
    ----------
    d : np.ndarray
        The original array to be perturbed.
    Prec_eps : scipy.sparse.csc_matrix
        The precision matrix for the Gaussian noise.

    Returns
    -------
    np.ndarray
        The perturbed array.
    """
    # Compute the covariance matrix (inverse of the precision matrix)
    p, _ = Prec_eps.shape
    Cov_eps = inv(Prec_eps).reshape(p, p)

    # The length of the noise vector
    length = d.shape[0]

    # Sample noise from a multivariate normal distribution
    eps = multivariate_normal(mean=np.zeros(length), cov=Cov_eps)

    # Add the noise to 'd'
    d_perturbed = d + eps

    return d_perturbed


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

        # Initialize your attributes here
        self.Prec_u = Prec_u
        self.Graph_u = Graph_u
        self.Prec_eps = Prec_eps
        self.H = H

    def fit(
        self,
        U: np.ndarray,
        Y: Optional[np.ndarray] = None,
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
            self.H = self.fit_H(U, Y)
        elif verbose:
            print("H mapping exists. Use `fit_H` to refit if necessary")

    def transport(self, U: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Transport U from a sample from the prior to the posterior
        """
        canonical = self.pushforward_to_canonical(U)
        updated = self.update_canonical(canonical, d)
        return self.pullback_from_canonical(updated)

    # Low-level API methods
    def fit_precision(self, U: np.ndarray) -> None:
        self.Prec_u = fit_precision_cholesky(U, self.Graph_u)

    def fit_H(self, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        H_fitted = linear_l1_regression(U, Y)
        return H_fitted

    def pushforward_to_canonical(self, U: np.ndarray) -> np.ndarray:
        # TODO: Replace with actual pushforward logic
        assert self.Prec_u is not None, "Precision must exist to pushforward"

        n, p = U.shape
        Nu = np.empty((n, p))
        for i in range(n):
            Nu[i, :] = self.Prec_u @ U[i, :]
        return Nu

    def update_canonical(
        self, canonical: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        assert self.H is not None, "H must be provided of fitted"
        assert self.Prec_u is not None, "Precision must be provided of fitted"

        # posterior nu
        n, p = canonical.shape
        updated_canonical = np.empty((n, p))
        for i in range(n):
            d_perturbed = perturb_d(d, self.Prec_eps)
            updated_canonical[i:,] = (
                canonical[i:,] + self.H.T @ self.Prec_eps @ d_perturbed
            )

        # posterior precision
        self.Prec_u = self.Prec_u + self.H.T @ self.Prec_eps @ self.H

        return updated_canonical

    def pullback_from_canonical(
        self, updated_canonical: np.ndarray
    ) -> np.ndarray:
        n, p = updated_canonical.shape
        # This implementation can be improved using pre-computed (sparse)
        # AMD-optimized cholesky factor of precision matrix
        updated_moment = np.empty((n, p))
        for i in range(n):
            # here we can likely use chol-solve repeatedly!!!
            updated_moment[i, :] = spsolve(
                self.Prec_u, updated_canonical[i, :]
            )

        return updated_moment
