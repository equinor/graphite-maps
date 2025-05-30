import networkx as nx
import numpy as np
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import bicgstab
from sksparse.cholmod import cholesky
from tqdm import tqdm

from . import linear_regression as lr
from .precision_estimation import (
    find_sparsity_structure_from_chol,
    fit_precision_cholesky,
)


def generate_gaussian_noise(
    n: int, Prec: spmatrix, seed: int | None = None, verbose_level: int = 0
) -> np.ndarray:
    """
    Generates 'n' samples of Gaussian noise with precision 'Prec'.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    Prec : scipy.sparse.spmatrix
        The precision matrix for the Gaussian noise, assumed to be sparse.

    Returns
    -------
    np.ndarray
        The Gaussian noise array of shape (n, m), where Prec has shape (m, m).
    """

    m = Prec.shape[0]
    rng = np.random.default_rng(seed)
    eps = rng.normal(
        loc=0,
        scale=np.sqrt(np.linalg.inv(Prec.toarray()).diagonal()),
        size=(n, m),
    )

    # standard_normal_samples = rng.normal(size=(n, m))
    # cholesky_factor = cholesky(Prec)

    # Transform the samples using the inverse Cholesky factor
    # This transformation results in samples from N(0, Prec^-1)
    # eps = cholesky_factor.solve_A(standard_normal_samples.T).T

    assert eps.shape == (n, m), "Sampling returns wrong size"

    if verbose_level > 0:
        print(
            f"Sampling with seed={seed}\nSampled Gaussian noise with shape {eps.shape}"
        )

    return eps


class EnIF:
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

        self.Prec_u = Prec_u
        self.Graph_u = Graph_u
        self.Prec_eps = Prec_eps
        self.H = H
        self.unexplained_variance: np.ndarray | None = None

        # Convenience for re-use of cholesky and ordering
        self.Graph_C: nx.Graph | None = None
        self.perm_compose: np.ndarray | None = None
        self.P_rev: spmatrix | None = None
        self.P_order: spmatrix | None = None

    def fit(
        self,
        U: np.ndarray,
        Y: np.ndarray | None = None,
        learning_algorithm: str | None = "LASSO",
        lambda_l2_precision: float = 1.0,
        ordering_method: str = "metis",
        verbose_level: int = 0,
    ) -> None:
        """
        Fit precision of u and (sparse) mapping H:u->y.
        """

        if self.Prec_u is None:
            self.fit_precision(
                U,
                lambda_l2_precision,
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
        U: np.ndarray,
        Y: np.ndarray,
        d: np.ndarray,
        update_indices: np.ndarray | None = None,
        seed: int | None = None,
        iterative: bool = False,
        verbose_level: int = 0,
    ) -> np.ndarray:
        """
        Transport U from a sample from the prior to the posterior
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
        U: np.ndarray,
        lambda_l2: float = 1.0,
        ordering_method: str = "metis",
        verbose_level: int = 0,
    ) -> None:
        """
        Estimate self.Prec_u from data U w.r.t. graph self.Graph_u
        """
        (
            self.Prec_u,
            self.Graph_C,
            self.perm_compose,
            self.P_rev,
            self.P_order,
        ) = fit_precision_cholesky(
            U,
            self.Graph_u,
            lambda_l2=lambda_l2,
            verbose_level=verbose_level - 1,
            ordering_method=ordering_method,
        )

    def fit_H(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        learning_algorithm: str | None = "LASSO",
        verbose_level: int = 0,
    ) -> None:
        """
        Estimate H from data U using (sparse) linear regression
        """
        if learning_algorithm == "LASSO":
            self.H = lr.linear_l1_regression(U, Y, verbose_level=verbose_level - 1)
        elif learning_algorithm == "influence-boost":
            self.H = lr.linear_boost_ic_regression(
                U, Y, verbose_level=verbose_level - 1
            )
        else:
            raise ValueError(
                f"Argument `learning_algorithm` must be a valid type. "
                f"Got: {learning_algorithm}"
            )
        self.residual_variance(U, Y, verbose_level=verbose_level - 1)

    def pushforward_to_canonical(
        self, U: np.ndarray, verbose_level: int = 0
    ) -> np.ndarray:
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
        self, U: np.ndarray, Y: np.ndarray, verbose_level: int = 0
    ) -> np.ndarray:
        """Residual from regression self.H for Y on U"""
        if self.H is None:
            raise ValueError("H is not set.")

        self.residual_variance(U, Y, verbose_level=verbose_level - 1)

        return lr.response_residual(U, Y, self.H, verbose_level=verbose_level - 1)

    def residual_variance(
        self, U: np.ndarray, Y: np.ndarray, verbose_level: int = 0
    ) -> None:
        """Sets self.unexplained_variance from variance on residuals"""
        if self.H is None:
            raise ValueError("H is not set.")

        self.unexplained_variance = lr.residual_variance(
            U, Y, self.H, verbose_level=verbose_level - 1
        )

    def generate_observation_noise(
        self, n: int, seed: int | None = None, verbose_level: int = 0
    ) -> np.ndarray:
        """Sample n realizations of observation noise."""
        if n < 1:
            raise ValueError(f"`n` should be g.e. 1, got {n}")

        return generate_gaussian_noise(
            n, self.Prec_eps, seed=seed, verbose_level=verbose_level - 1
        )

    def update_canonical(
        self,
        canonical: np.ndarray,
        residual_noisy: np.ndarray,
        d: np.ndarray,
        verbose_level: int = 0,
    ) -> np.ndarray:
        """
        Use information-filter equations to update (eta, Prec) using perturbed
        d
        """
        assert self.H is not None, "H must be provided of fitted"
        assert self.Prec_u is not None, "Precision must be provided of fitted"

        n, p = canonical.shape
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
            updated_canonical[i, :] += self.H.T @ Prec_r @ d_adjusted

        # posterior precision
        self.Prec_u = self.Prec_u + self.H.T @ Prec_r @ self.H

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
        updated_canonical: np.ndarray,
        update_indices: np.ndarray | None = None,
        U_prior: np.ndarray | None = None,
        iterative: bool = False,
        verbose_level: int = 0,
    ) -> np.ndarray:
        """
        Solve u = Prec * eta using selective updates for specified indices,
        taking into account previously calculated values of X where
        appropriate.
        """
        assert self.Prec_u is not None, "Prec_u must exist"

        if verbose_level > 0:
            print("Mapping canonical-scaled realizations to moment realization")

        p = updated_canonical.shape[1]  # Number of columns in the matrix
        all_indices = np.arange(p)
        if update_indices is None:
            update_indices = np.arange(p, dtype=int)

        update_indices = (
            update_indices.astype(int)
            if update_indices.size > 0
            else np.array([], dtype=int)
        )
        unchanged_indices = np.setdiff1d(all_indices, update_indices)

        if U_prior is None:
            updated_moment = np.zeros(updated_canonical.shape)
        else:
            updated_moment = U_prior.copy()

        if update_indices.size > 0:
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
        neighbor_propagation_order=10,
        verbose_level: int = 0,
    ):
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

        return np.array(list(all_nodes))

    @property
    def C_structure_exists(self):
        """Check if information from permuted Cholesky decomposition exists"""
        return not (
            self.Graph_C is None
            or self.perm_compose is None
            or self.P_rev is None
            or self.P_order is None
        )
