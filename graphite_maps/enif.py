from typing import Literal

import networkx as nx
import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import bicgstab
from sksparse.cholmod import cholesky
from tqdm import tqdm

from graphite_maps import linear_regression as lr
from graphite_maps.precision_estimation import (
    find_sparsity_structure_from_chol,
    fit_precision_cholesky,
)
from graphite_maps.utils import generate_gaussian_noise


class EnIF:
    """Initialize an Ensemble Information Filter (EnIF).

    The filter is parametrized by the prior precision of the state `u` (or a
    graph specifying its sparsity), the precision of the observation noise,
    and optionally the linear map `H`. Anything left as `None` at initialization
    can be learned from data via `fit`.

    Parameters
    ----------
    Prec_u : spmatrix, optional
        Prior precision matrix of the state, shape (params, params). If omitted,
        it is estimated from data using `Graph_u` as the sparsity pattern.
    Graph_u : nx.Graph, optional
        Conditional-independence graph on the `params` state components,
        defining the sparsity of `Prec_u`. Required when `Prec_u` is not
        provided.
    Prec_eps : spmatrix
        Precision matrix of the observation noise, shape (responses, responses).
    H : spmatrix, optional
        Linear observation operator mapping state to responses, shape
        (responses, params). If omitted, it is estimated from data by `fit`.
    """

    def __init__(
        self,
        *,
        Prec_u: spmatrix | None = None,
        Graph_u: nx.Graph | None = None,
        Prec_eps: spmatrix | None,
        H: spmatrix | None = None,
    ) -> None:
        assert Prec_u is not None or Graph_u is not None, (
            "Provide either Prec_u or Graph_u"
        )

        if Prec_u is not None and (
            not isinstance(Prec_u, sp.sparse.sparray) and Prec_u.ndim == 2
        ):
            raise TypeError("`Prec_u` must be a 2D sparse array")
        if Prec_u is not None and Prec_u.shape[0] != Prec_u.shape[1]:
            raise ValueError("`Prec_u` must be a square 2D sparse array")

        if H is not None and (
            not isinstance(H, sp.sparse.sparray) and Prec_u.ndim == 2
        ):
            raise TypeError("`H` must be a 2D sparse array")

        self.Prec_u = Prec_u
        self.Graph_u = Graph_u
        self.Prec_eps = Prec_eps
        self.H = H
        self.unexplained_variance: NDArray[np.floating] | None = None

        # Convenience for re-use of cholesky and ordering
        self.Graph_C: nx.Graph | None = None
        self.perm_compose: NDArray[np.integer] | None = None
        self.P_rev: spmatrix | None = None
        self.P_order: spmatrix | None = None

    def fit(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
        learning_algorithm: Literal["LASSO", "influence-boost"] = "LASSO",
        ordering_method: str = "metis",
        verbose_level: int = 0,
    ) -> None:
        """Fit the prior precision of `u` and, optionally the mapping `H`.

        If `Prec_u` was not supplied at construction, it is estimated from `U`
        using the sparsity pattern of `Graph_u`. If `Y` is supplied and `H` was
        not set at construction, a sparse linear map `H` is learned from `U`
        to `Y` and the per-response residual variance is stored on the
        instance. Already-provided quantities are kept as-is.

        Parameters
        ----------
        U : ndarray of shape (realizations, parameters)
            Prior ensemble: `n` realizations of the `p`-dimensional state.
        Y : ndarray of shape (realizations, responses), optional
            Response ensemble used to learn `H`. Must be omitted if `H` was
            provided at construction.
        learning_algorithm : {"LASSO", "influence-boost"}, default="LASSO"
            Estimator used to fit `H`. Ignored when `Y` is not provided.
        ordering_method : str, default="metis"
            Fill-reducing ordering passed to the Cholesky factorization when
            estimating `Prec_u`.
        verbose_level : int, default=0
        """

        if self.Prec_u is None:
            self.fit_precision(
                U,
                verbose_level=verbose_level - 1,
                ordering_method=ordering_method,
            )
        elif verbose_level > 0:
            print("Precision u exists. Use `fit_precision` to refit if necessary")
        if Y is not None:
            assert self.H is None, "Y should not be provided if H exists"
            self.fit_H(
                U=U,
                Y=Y,
                learning_algorithm=learning_algorithm,
                verbose_level=verbose_level - 1,
            )
        elif verbose_level > 0:
            print("H mapping exists. Use `fit_H` to refit if necessary")

    def transport(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating],
        d: NDArray[np.floating],
        update_indices: NDArray[np.integer] | None = None,
        seed: int | None = None,
        iterative: bool = False,
        verbose_level: int = 0,
    ) -> NDArray[np.floating]:
        """Transport a prior ensemble to the posterior given observations `d`.

        Each realization is mapped to the canonical (information) parametrization
        `eta = Prec_u @ u`, updated with a perturbed-observation information
        filter step using the current `H`, `Prec_u`, and `Prec_eps`, and then
        mapped back to the state space. When `update_indices` is given, only
        those components are solved for and the rest are copied from the
        prior, which is the usual speed-up for localized updates.

        Parameters
        ----------
        U : ndarray of shape (realizations, parameters)
            Prior ensemble.
        Y : ndarray of shape (realizations, responses)
            Response ensemble evaluated on `U`.
        d : ndarray of shape (responses,)
            Observed data vector.
        update_indices : ndarray of int, optional
            Indices of state components to update. Defaults to all `parameter`
            components.
        seed : int, optional
        iterative : bool, default=False
        verbose_level : int, default=0

        Returns
        -------
        U_post : ndarray of shape (realizations, parameters)
            Posterior ensemble. Components not listed in `update_indices` are
            equal to those in `U`.
        """
        n, _ = U.shape
        n_y, m = Y.shape
        assert n == n_y, "Number of ensembles must be the same"
        assert d.shape == (m,), "Observations must match responses"

        # Map parameters to canonical parametrization
        canonical = self.pushforward_to_canonical(U, verbose_level=verbose_level - 1)

        # Work out residuals and associate unexplained variance
        residuals = self.response_residual(U, Y, verbose_level=verbose_level - 1)
        # Due to observation error
        eps = self.generate_observation_noise(
            n, seed=seed, verbose_level=verbose_level - 1
        )
        residual_noisy = residuals + eps

        # Update in canonical parametrization
        canonical_updated = self.update_canonical(
            canonical, residual_noisy, d, verbose_level=verbose_level - 1
        )

        # Bring realizations back
        return self.pullback_from_canonical(
            updated_canonical=canonical_updated,
            update_indices=update_indices,
            U_prior=U,
            iterative=iterative,
            verbose_level=verbose_level - 1,
        )

    # Low-level API methods
    def fit_precision(
        self,
        U: NDArray[np.floating],
        ordering_method: str = "metis",
        verbose_level: int = 0,
    ) -> None:
        """
        Estimate self.Prec_u from data U w.r.t. graph self.Graph_u
        """
        assert self.Graph_u is not None, "Graph_u must be set to fit precision"
        (
            self.Prec_u,
            self.Graph_C,
            self.perm_compose,
            self.P_rev,
            self.P_order,
        ) = fit_precision_cholesky(
            U=U,
            Graph_u=self.Graph_u,
            verbose_level=verbose_level - 1,
            ordering_method=ordering_method,
        )

    def fit_H(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating],
        learning_algorithm: Literal["LASSO", "influence-boost"] = "LASSO",
        verbose_level: int = 0,
    ) -> None:
        """
        Estimate H from data U using (sparse) linear regression
        """
        if learning_algorithm not in ("LASSO", "influence-boost"):
            raise ValueError(
                f"Argument `learning_algorithm` must be 'LASSO' or 'influence-boost'. "
                f"Got: {learning_algorithm}"
            )

        if learning_algorithm == "LASSO":
            self.H = lr.linear_l1_regression(U, Y, verbose_level=verbose_level - 1)
        else:
            self.H = lr.linear_boost_ic_regression(
                U, Y, verbose_level=verbose_level - 1
            )

        self.unexplained_variance = lr.residual_variance(
            U, Y, self.H, verbose_level=verbose_level - 1
        )

    def pushforward_to_canonical(
        self, U: NDArray[np.floating], verbose_level: int = 0
    ) -> NDArray[np.floating]:
        """
        Map each realization u in U to canonical space eta = Prec * u
        """
        if verbose_level > 0:
            print("Mapping realizations to canonical space")

        assert self.Prec_u is not None, "Precision must exist to pushforward"
        Eta = U @ self.Prec_u
        assert Eta.shape == U.shape, "Eta preserves the shape of U"
        return Eta

    def Prec_residual_noisy(self, verbose_level: int = 0) -> spmatrix:
        if self.Prec_eps is None:
            raise ValueError("Prec_eps is not set.")
        elif self.unexplained_variance is None:
            raise ValueError("`unexplained_variance` is not set.")

        eps_variances = 1.0 / self.Prec_eps.diagonal()
        residual_noisy_var = self.unexplained_variance + eps_variances
        Prec_r = diags(1.0 / residual_noisy_var, 0, format="csc")
        assert Prec_r.shape == self.Prec_eps.shape, (
            "Residuals and noise precision should have same shape"
        )
        if verbose_level > 0:
            print(
                f"Total residual variance: {np.sum(residual_noisy_var)}\n"
                f"Unexplained variance: {np.sum(self.unexplained_variance)}\n"
                f"Measurement variance: {np.sum(eps_variances)}"
            )
        return Prec_r

    def response_residual(
        self, U: NDArray[np.floating], Y: NDArray[np.floating], verbose_level: int = 0
    ) -> NDArray[np.floating]:
        """Residual from regression self.H for Y on U"""
        if self.H is None:
            raise ValueError("H is not set.")

        self.unexplained_variance = lr.residual_variance(
            U, Y, self.H, verbose_level=verbose_level - 1
        )

        return lr.response_residual(U, Y, self.H, verbose_level=verbose_level - 1)

    def generate_observation_noise(
        self, n: int, seed: int | None = None, verbose_level: int = 0
    ) -> NDArray[np.floating]:
        """Sample n realizations of observation noise."""

        return generate_gaussian_noise(
            n, self.Prec_eps, seed=seed, verbose_level=verbose_level - 1
        )

    def update_canonical(
        self,
        canonical: NDArray[np.floating],
        residual_noisy: NDArray[np.floating],
        d: NDArray[np.floating],
        verbose_level: int = 0,
    ) -> NDArray[np.floating]:
        """
        Use information-filter equations to update (eta, Prec) using perturbed
        d
        """
        assert self.H is not None, "H must be provided of fitted"
        assert self.Prec_u is not None, "Precision must be provided of fitted"

        n, _p = canonical.shape
        n_r, m = residual_noisy.shape
        assert n == n_r, "canonical and residual_noisy must have equal samples"
        assert d.shape == (m,), "d and residual_noisy must have matching dimension"

        if verbose_level > 5:
            chol_LLT = cholesky(self.Prec_u, ordering_method="metis")
            prior_logdet = 2.0 * np.sum(np.log(chol_LLT.L().diagonal()))
            print(f"Prior precision log-determinant: {prior_logdet}")

        updated_canonical = canonical.copy()
        Prec_r = self.Prec_residual_noisy(verbose_level=verbose_level - 1)
        for i in range(n):
            d_adjusted = d - residual_noisy[i, :]
            # Eqn (46) from the paper
            updated_canonical[i, :] += self.H.T @ Prec_r @ d_adjusted

        # posterior precision
        self.Prec_u = self.Prec_u + self.H.T @ Prec_r @ self.H  # Eqn (47)

        # Only print this if one really wants it. The cholesky can be heavy
        if verbose_level > 5:
            chol_LLT = cholesky(self.Prec_u, ordering_method="metis")
            posterior_logdet = 2.0 * np.sum(np.log(chol_LLT.L().diagonal()))
            print(f"Posterior precision log-determinant: {posterior_logdet}")

            # Update the ordering knowledge
            self.Graph_C, self.perm_compose, self.P_rev, self.P_order = (
                find_sparsity_structure_from_chol(chol_LLT=chol_LLT)
            )

        return updated_canonical

    def pullback_from_canonical(
        self,
        updated_canonical: NDArray[np.floating],
        update_indices: NDArray[np.integer] | None = None,
        U_prior: NDArray[np.floating] | None = None,
        iterative: bool = False,
        verbose_level: int = 0,
    ) -> NDArray[np.floating]:
        """
        Solve u = Prec * eta using selective updates for specified indices,
        taking into account previously calculated values of U_prior where
        appropriate.
        """
        assert self.Prec_u is not None, "Prec_u must exist"

        if verbose_level > 0:
            print("Mapping canonical-scaled realizations to moment realization")

        p = updated_canonical.shape[1]  # Number of columns in the matrix
        all_indices = np.arange(p, dtype=int)
        if update_indices is None:
            update_indices = all_indices
        else:
            update_indices = update_indices.astype(int)

        assert update_indices is not None
        unchanged_indices = np.setdiff1d(all_indices, update_indices)

        updated_moment: NDArray[np.floating]
        if U_prior is None:
            updated_moment = np.zeros(updated_canonical.shape)
        else:
            updated_moment = U_prior.copy()

        if update_indices.size == 0:
            return updated_moment

        A33 = self.Prec_u[update_indices, :][:, update_indices]

        if iterative:
            for i in tqdm(
                range(updated_moment.shape[0]),
                desc="Mapping data to moment parametrisation realization-by-realization",
            ):
                if unchanged_indices.size > 0:
                    A32 = self.Prec_u[update_indices, :][:, unchanged_indices]
                    Y32 = updated_canonical[i, update_indices] - A32.dot(
                        updated_moment[i, unchanged_indices]
                    )
                else:
                    Y32 = updated_canonical[i, update_indices]

                x_updated, _ = bicgstab(A33, Y32)
                updated_moment[i, update_indices] = x_updated
        else:
            chol_LLT = cholesky(A33, ordering_method="metis")
            if unchanged_indices.size > 0:
                A32 = self.Prec_u[update_indices, :][:, unchanged_indices]
                Y32 = updated_canonical[:, update_indices].T - A32.dot(
                    updated_moment[:, unchanged_indices].T
                )
            else:
                Y32 = updated_canonical[:, update_indices].T

            X32 = chol_LLT.solve_A(Y32).T
            updated_moment[:, update_indices] = X32

        return updated_moment

    def get_update_indices(
        self,
        neighbor_propagation_order: int = 10,
        verbose_level: int = 0,
    ) -> NDArray[np.integer]:
        """
        Determine indices to update based on the order of neighbor propagation.

        Parameters:
        - neighbor_propagation_order: Levels of neighbors to include.

        Returns:
        - update_indices: Array of indices that includes the initial
            predictors and their neighbors up to the specified order.
        """
        assert self.H is not None, "H must exist"
        assert self.Prec_u is not None, "Prec_u must exist"

        _, cols = self.H.nonzero()
        predictors = set(cols)
        adjacency = self.Prec_u.copy()
        all_nodes = set(predictors)  # Start with predictors

        # Initialize sets to manage nodes
        current_nodes = predictors.copy()
        new_nodes = set()

        # Iteratively find neighbors up to the specified order
        for _ in range(neighbor_propagation_order):
            temp_nodes = set()
            for col in current_nodes:
                neighbors = adjacency[:, col].nonzero()[0]
                temp_nodes.update(neighbors)

            # Update new_nodes with newly discovered nodes
            new_nodes = temp_nodes.difference(all_nodes)
            all_nodes.update(new_nodes)
            current_nodes = new_nodes.copy()

        if verbose_level > 0:
            print(
                f"Retrieving {len(all_nodes)} parameters out of a total "
                f"{adjacency.shape[0]}"
            )

        return np.array(list(all_nodes), dtype=int)

    @property
    def C_structure_exists(self) -> bool:
        """Check if information from permuted Cholesky decomposition exists"""
        return not (
            self.Graph_C is None
            or self.perm_compose is None
            or self.P_rev is None
            or self.P_order is None
        )


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
