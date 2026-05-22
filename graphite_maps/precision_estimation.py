import logging
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csc_array, tril
from sksparse.cholmod import Factor, cholesky
from tqdm import tqdm

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def find_sparsity_structure_from_chol(
    chol_LLT: Factor,
) -> tuple[nx.Graph, NDArray[np.integer], csc_array, csc_array]:
    L = chol_LLT.L()
    p = L.shape[0]

    # Get the in-fill reducing permutation vector
    perm_order = chol_LLT.P()

    # Create the permutation matrix P
    P_order = sp.csc_array(
        (np.ones(len(perm_order)), (perm_order, np.arange(len(perm_order)))),
        shape=(p, p),
    )

    # Create the reverse order permutation array
    perm_reverse = np.arange(p - 1, -1, -1)
    # Create the reverse permutation matrix
    P_rev = sp.csc_array((np.ones(p), (perm_reverse, np.arange(p))), shape=(p, p))

    # Apply in-fill reducing ordering permutation to reverse permutation
    perm_compose = perm_order[perm_reverse]

    # Extract the lower triangular Cholesky factor
    L = chol_LLT.L()

    # Create matrix of non-zeroes equalling C: L @ L.T=C.T @ C unique when SPD
    C_pattern = (P_rev @ L @ P_rev).T

    # Extract structure into a graph for C
    Graph_C = nx.from_scipy_sparse_array(C_pattern)
    Graph_C.remove_edges_from(nx.selfloop_edges(Graph_C))

    # Return the results
    return Graph_C, perm_compose, P_rev, P_order


def find_sparsity_structure_from_graph(
    Graph_u: nx.Graph,
    ordering_method: str = "metis",
) -> tuple[nx.Graph, NDArray[np.integer], csc_array, csc_array]:
    """
    Finds sparsity structure for lower triangular C so that
      C.T @ C = L @ L.T = P.T @ A @ P.

    The permutation P is optimized and returned, so is the non-zero structure
    of C. For convenience the permutation so that data for A can be arranged
    according to C is also returned.

    Parameters
    ----------
    Graph_u : nx.Graph
        The graph representing the non-zero structure in the precision matrix.

    Returns
    -------
    Graph_C : nx.Graph
        The graph representing the non-zero structure in C.
    perm_compose : NDArray[np.integer]
        The composed permutation array.
    P_rev : scipy.sparse.csc_array
        The reverse permutation matrix.
    P_order : scipy.sparse.csc_array
        The in-fill reducing ordering permutation matrix.

    Examples
    --------
    >>> import networkx as nx
    >>> Graph_u = nx.Graph([(0, 1), (0, 3), (0, 4), (1, 4), (3, 4)])

    The "metis" ordering method is not deterministic, so we use "natural" here:

    >>> result = find_sparsity_structure_from_graph(Graph_u, ordering_method="natural")
    >>> Graph_C, perm_compose, P_rev, P_order = result
    >>> nx.to_scipy_sparse_array(Graph_C).todense().round(1)
    array([[ 0. ,  0.4,  0.4,  0.5],
           [ 0.4,  0. , -0.1,  0.5],
           [ 0.4, -0.1,  0. ,  0.5],
           [ 0.5,  0.5,  0.5,  0. ]])
    >>> perm_compose
    array([3, 2, 1, 0])
    >>> P_rev.todense()
    array([[0., 0., 0., 1.],
           [0., 0., 1., 0.],
           [0., 1., 0., 0.],
           [1., 0., 0., 0.]])
    >>> P_order.todense()
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """

    # Create SPD matrix with same sparsity structure as Prec
    SPD_Prec = nx.to_scipy_sparse_array(
        Graph_u, weight=None, dtype=np.float64, format="csc"
    )
    # Use Gershgorin circle theorem to ensure SP
    # This ensures all eigenvalues are in a circle centered at max_degree+1.0
    # and radius < (max_degree+1.0), so guaranteed > 0
    max_degree = max(dict(Graph_u.degree()).values())
    log.info("max degree of graph is: %s", max_degree)
    SPD_Prec.setdiag(max_degree + 1.0)

    # PT prec P = LLT
    start = time.perf_counter()
    chol_LLT = cholesky(SPD_Prec, ordering_method=ordering_method)
    end = time.perf_counter()
    log.info("Permutation optimization took %.2f seconds", end - start)

    Graph_C, perm_compose, P_rev, P_order = find_sparsity_structure_from_chol(
        chol_LLT=chol_LLT
    )

    log.info("Parameters in precision: %s\n", tril(SPD_Prec).nnz)
    log.info("Parameters in Cholesky factor: %s", Graph_C.number_of_edges())

    # Return the results
    return Graph_C, perm_compose, P_rev, P_order


def objective_function(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> float:
    """
    Objective function for optimizing the affine KR map with standard Gaussian
    reference and l2 regularized dependence.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    float
        The value of `objective_function`.
    """
    C_k = C_k.copy()

    C_k[-1] = np.exp(C_k[-1])
    Su = U.dot(C_k)
    n, _ = U.shape
    regularization_l2 = 0.5 * lambda_l2 * np.sum(C_k[:-1] ** 2)
    return 0.5 * np.sum(Su**2) - n * np.log(abs(C_k[-1])) + regularization_l2


def gradient(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> NDArray[np.floating]:
    """
    Gradient of the objective function.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    np.ndarray
        The gradient of the objective function.
    """
    C_k = C_k.copy()

    n, _ = U.shape
    C_k[-1] = np.exp(C_k[-1])
    prediction = U.dot(C_k)
    grad = U.T.dot(prediction)
    grad[:-1] += lambda_l2 * C_k[:-1]  # Adjust for L2 regularization
    grad[-1] -= n / C_k[-1]  # Adjust for the -log|C_k,k| term
    grad[-1] *= C_k[-1]  # Adjust for log-transform
    return grad


def hessian(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> NDArray[np.floating]:
    """
    Hessian `objective_function`.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    np.ndarray
        The Hessian of the objective function.
    """
    C_k = C_k.copy()

    n, _ = U.shape
    H = U.T.dot(U)
    np.fill_diagonal(H[:-1, :-1], H.diagonal()[:-1] + lambda_l2)  # L2-term
    C_k[-1] = np.exp(C_k[-1])  # log-transform
    H[-1, -1] += n / (C_k[-1] ** 2)  # Adjust for the -log|C_k,k| term
    H[-1, -1] *= 2.0 * C_k[-1]  # log-transform adjustment
    return H


def solve_row_closed_form(
    U_reduced: np.ndarray,
    lambda_l2: float,
) -> tuple[np.ndarray, float]:
    """
    Closed-form minimizer for one row objective.

    This function computes the exact analytic solution for the row-wise objective
    function, avoiding the need for iterative numerical optimization.
    For the full mathematical derivation of this closed-form solution,
    see ``docs/source/ClosedFormRowSolver.md``.

    Parameters
    ----------
    U_reduced : np.ndarray
        Reduced data matrix with columns corresponding to non-zero entries
        in row ``k`` ordered as ``[..., k]``.
    lambda_l2 : float
        L2 regularization weight on off-diagonal coefficients.

    Returns
    -------
    tuple[np.ndarray, float]
        Off-diagonal coefficients and positive diagonal coefficient.

    Raises
    ------
    ValueError
        If numerical degeneracy prevents a valid positive diagonal estimate.
    """
    n, n_cols = U_reduced.shape
    z = U_reduced[:, -1]

    if n_cols == 1:
        alpha = float(np.dot(z, z))
        if alpha <= 0.0:
            raise ValueError("Degenerate row: non-positive alpha in closed-form solve")
        diag_value = np.sqrt(n / alpha)
        return np.empty(0, dtype=U_reduced.dtype), float(diag_value)

    X = U_reduced[:, :-1]
    gram = X.T @ X
    np.fill_diagonal(gram, np.diag(gram) + lambda_l2)
    rhs = X.T @ z

    beta_tilde = np.linalg.solve(gram, rhs)
    alpha = float(np.dot(z, z) - np.dot(rhs, beta_tilde))
    if alpha <= 0.0:
        raise ValueError("Degenerate row: non-positive alpha in closed-form solve")

    diag_value = np.sqrt(n / alpha)
    off_diag = -diag_value * beta_tilde
    return off_diag, float(diag_value)


def optimize_sparse_affine_kr_map(
    U: NDArray[np.floating],
    G: nx.Graph,
    use_tqdm: bool = True,
) -> csc_array:
    """Optimize the affine Knothe-Rosenblatt (KR) map with standard Gaussian
    reference and l2-regularized dependence using the closed-form row solve.

    Parameters
    ----------
    U : np.ndarray
        The data matrix with shape (samples, parameters)
    G : networkx.Graph
        The graph representing the non-zero structure in C.

    Returns
    -------
    scipy.sparse.csc_array
        The optimized sparse Cholesky factor C of the precision matrix,
        such that C.T @ C = Prec.

    Examples
    --------

    In this example we start with a known precision matrix, generate data,
    then use that data to try to infer the precision matrix again.

    Create precision matrix Prec:

    >>> import numpy as np
    >>> import scipy as sp
    >>> import networkx as nx
    >>> rng = np.random.default_rng(42)
    >>> diagonal = np.array([1, 2, 3, 4], dtype=float)
    >>> Prec = np.diag(diagonal) + np.diag(diagonal[:-1] / 2, k=1)
    >>> Prec += np.diag(diagonal[:-1] / 2, k=-1)
    >>> Prec
    array([[1. , 0.5, 0. , 0. ],
           [0.5, 2. , 1. , 0. ],
           [0. , 1. , 3. , 1.5],
           [0. , 0. , 1.5, 4. ]])

    Sample data U and create corresponding graph G:

    >>> mean = np.ones(Prec.shape[0])
    >>> cov = np.linalg.inv(Prec)
    >>> U = rng.multivariate_normal(mean=mean, cov=cov, size=999)

    >>> G_mat = sp.sparse.csc_array((Prec != 0).astype(int))
    >>> G = nx.from_scipy_sparse_array(G_mat)
    >>> G.edges
    EdgeView([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])

    Estimate the precision matrix from U and G:

    >>> C = optimize_sparse_affine_kr_map(U=U, G=G, use_tqdm=False).todense()
    >>> Prec_est = (C.T @ C)
    >>> Prec_est.round(2) # Estimated precision matrix
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])

    Demonstrate shift invariance. Notice that we get the same result as above:

    >>> U_shift = U +  np.array([1, 10, 100, 1000])
    >>> C = optimize_sparse_affine_kr_map(U=U_shift, G=G, use_tqdm=False).todense()
    >>> (C.T @ C).round(2)
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])

    Demonstrate affine invariance (shift and scale):

    >>> mu = np.array([5, 2, 3, 1])
    >>> sigma = np.array([1, 2, 4, 8])
    >>> U_scaled = (U + mu) * sigma
    >>> C_scaled = optimize_sparse_affine_kr_map(U=U_scaled, G=G, use_tqdm=False).todense()
    >>> Prec_est_scaled = (C_scaled.T @ C_scaled)
    >>> Prec_est_scaled.round(2)
    array([[0.99, 0.23, 0.  , 0.  ],
           [0.23, 0.5 , 0.13, 0.  ],
           [0.  , 0.13, 0.2 , 0.05],
           [0.  , 0.  , 0.05, 0.06]])

    Notice that this is the same as above:

    >>> Prec_est2 = np.diag(sigma) @ Prec_est_scaled @ np.diag(sigma)
    >>> (Prec_est2).round(2)
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])
    >>> np.allclose(Prec_est, Prec_est2)
    True
    """
    log.info("Starting statistical fitting of precision")

    n, p = U.shape
    assert n > 1

    # Standardize data U to have zero mean, unit variance
    mu = U.mean(axis=0, keepdims=True)
    sigma = U.std(axis=0)
    U_std = (U - mu) / sigma[None, :]

    C_full = sp.lil_array((p, p))  # lil_array for efficient row operations
    loop_function = (
        tqdm(range(p), desc="Learning precision Cholesky factor row-by-row")
        if use_tqdm
        else range(p)
    )

    for k in loop_function:
        non_zero_indices = [j for j in G.neighbors(k) if j < k] + [k]

        # Extract the reduced version of U
        U_reduced = U_std[:, non_zero_indices]

        # Optimization for reduced C_k
        lambda_l2_aic = 2.0 * len(non_zero_indices)
        off_diag_std, diag_value_std = solve_row_closed_form(
            U_reduced=U_reduced,
            lambda_l2=lambda_l2_aic,
        )

        # Unscale the row back to original coordinates
        if len(non_zero_indices) > 1:
            cols_off = non_zero_indices[:-1]
            C_full[k, cols_off] = off_diag_std / sigma[cols_off]
        C_full[k, k] = diag_value_std / sigma[k]

    # Convert to csc_array for efficient storage and arithmetic operations
    return C_full.tocsc()


def fit_precision_cholesky(
    U: NDArray[np.floating],
    Graph_u: nx.Graph,
    ordering_method: str = "metis",
    use_tqdm: bool = True,
    Graph_C: nx.Graph | None = None,
    perm_compose: NDArray[np.integer] | None = None,
    P_rev: csc_array | None = None,
    P_order: csc_array | None = None,
) -> tuple[csc_array, nx.Graph, NDArray[np.integer], csc_array, csc_array]:
    """
    Estimate the precision matrix using Cholesky decomposition.
    An l2-regularized negative log-likelihood is minimized.

    Parameters
    ----------
    U : The data matrix.
    Graph_u : The graph representing the non-zero structure in the precision
    matrix.

    Returns
    -------
    scipy.sparse.csc_array
        Estimated precision matrix.
    """
    _, p = U.shape
    assert len(Graph_u.nodes) == p, "nodes in graph equals columns of data"

    if Graph_C is None or perm_compose is None or P_rev is None or P_order is None:
        # 1. Find in-fill reducing ordering for C
        Graph_C, perm_compose, P_rev, P_order = find_sparsity_structure_from_graph(
            Graph_u,
            ordering_method=ordering_method,
        )

    # 2. Estimate non-zeroes of C
    U_perm = U[:, perm_compose]
    C = optimize_sparse_affine_kr_map(
        U=U_perm,
        G=Graph_C,
        use_tqdm=use_tqdm,
    )

    # 2.b Compute log-determinant of estimate, for logging
    L_r = P_rev @ C.T @ P_rev  # Factor of reverse precision
    prec_logdet = 2.0 * np.sum(np.log(L_r.diagonal()))
    log.info("Precision has log-determinant: %.3f", prec_logdet)

    # 3. Unwrap C to yield precision (Eqn 73 in paper)
    Prec = P_order @ P_rev @ (C.T @ C) @ P_rev @ P_order.T
    return Prec, Graph_C, perm_compose, P_rev, P_order


def fit_precision_cholesky_approximate(
    U: NDArray[np.floating],
    G: nx.Graph,
    neighbourhood_expansion: int = 2,
    use_tqdm: bool = True,
) -> csc_array:
    """
    Estimate the precision matrix using approximate Cholesky.
    The Cholesky is assumed as sparse as the corresponding precision, with
    sparsity pattern from G, but with increased neighbourhood. This is
    akin to a Vecchia approximation, which is alleviated by the neighbourhood
    expansion.

    No permutation optimisation is performed. It is beneficial if U and G have
    a "sensible" ordering. This may e.g. be that neighbours defined by G are
    close in U.

    Parameters
    ----------
    U : np.ndarray
        The data matrix.
    G : networkx.Graph
        The graph representing the non-zero structure in C.
    neighbourhood_expansion: int, optional
        The number of hops to the new neighbourhood set

    Returns
    -------
    scipy.sparse.csc_array
        The optimized sparse Cholesky factor of the precision matrix.
    """
    # The k'th power is a "neighborhood expansion", see docs of nx.power
    G = nx.power(G, k=neighbourhood_expansion)
    G.add_edges_from((n, n) for n in G.nodes())  # Add edges (1, 1), (2, 2), ...

    C = optimize_sparse_affine_kr_map(
        U=U,
        G=G,
        use_tqdm=use_tqdm,
    )
    Prec_approx = C.T @ C
    return Prec_approx.tocsc()


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
