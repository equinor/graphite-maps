import time
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, tril
from sksparse.cholmod import cholesky
from tqdm import tqdm


def graph_to_precision_matrix(graph: nx.Graph) -> csc_matrix:
    """
    Convert a NetworkX graph to a sparse CSC precision matrix.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph representing connections between variables.

    Returns
    -------
    scipy.sparse.csc_matrix
        CSC sparse matrix representing the precision matrix.
    """
    prec = nx.to_scipy_sparse_array(graph)
    prec.tolil()
    prec.setdiag(1)
    return prec.tocsc()


def precision_to_graph(precision_matrix: csc_matrix) -> nx.Graph:
    """
    Convert a sparse symmetric precision matrix to a NetworkX graph.

    Parameters
    ----------
    precision_matrix : scipy.sparse.csc_matrix
        CSC sparse matrix representing the precision matrix.

    Returns
    -------
    networkx.Graph
        A NetworkX graph where each edge corresponds to a non-zero element
        the precision matrix.
    """
    return nx.from_scipy_sparse_array(precision_matrix)


def gershgorin_spd_adjustment(prec):
    """
    Performs Gershgorin-style diagonal adjustment on the input symmetric
    sparse matrix `prec` and returns the adjusted matrix. The adjustment is
    performed to ensure that the matrix is symmetric positive definite (SPD).

    Parameters
    ----------
    prec : scipy.sparse.csc_matrix
        The input sparse matrix to adjust.

    Returns
    -------
    scipy.sparse.csc_matrix
        The SPD matrix after Gershgorin-style diagonal adjustment.
    """
    prec = prec.copy().tocsc()
    eps = 1e-1
    offdiag_abs_sum = np.asarray(np.abs(prec).sum(axis=1)).ravel() - prec.diagonal()
    for i in range(prec.shape[0]):
        if offdiag_abs_sum[i] >= prec[i, i]:
            prec[i, i] = offdiag_abs_sum[i] + eps
    return prec


def find_sparsity_structure_from_chol(
    chol_LLT: csc_matrix,
) -> tuple[nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
    L = chol_LLT.L()
    p = L.shape[0]

    # Get the in-fill reducing permutation vector
    perm_order = chol_LLT.P()

    # Create the permutation matrix P
    P_order = sp.csc_matrix(
        (np.ones(len(perm_order)), (perm_order, np.arange(len(perm_order)))),
        shape=(p, p),
    )

    # Create the reverse order permutation array
    perm_reverse = np.arange(p - 1, -1, -1)
    # Create the reverse permutation matrix
    P_rev = sp.csc_matrix((np.ones(p), (perm_reverse, np.arange(p))), shape=(p, p))

    # Apply in-fill reducing ordering permutation to reverse permutation
    perm_compose = perm_order[perm_reverse]

    # Extract the lower triangular Cholesky factor
    L = chol_LLT.L()

    # Create matrix of non-zeroes equalling C: LLT=CTC unique when SPD
    C_pattern = (P_rev @ L @ P_rev).T

    # Extract structure into a graph for C
    Graph_C = nx.from_scipy_sparse_array(C_pattern)
    Graph_C.remove_edges_from(nx.selfloop_edges(Graph_C))

    # Return the results
    return Graph_C, perm_compose, P_rev, P_order


def find_sparsity_structure_from_graph(
    Graph_u: nx.Graph,
    ordering_method: str = "metis",
    verbose_level: int = 0,
) -> tuple[nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
    """
    Finds sparsity structure for lower triangular C so that CTC = LLT = PTAP.
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
    perm_compose : NDArray[signedinteger[Any]]
        The composed permutation array.
    P_rev : scipy.sparse.csc_matrix
        The reverse permutation matrix.
    P_order : scipy.sparse.csc_matrix
        The in-fill reducing ordering permutation matrix.
    """

    # Create SPD matrix with same sparsity structure as Prec
    SPD_Prec = nx.to_scipy_sparse_array(Graph_u, weight=None)
    SPD_Prec = SPD_Prec.astype(np.float64)
    # Use Gershgorin circle theorem to ensure SP
    # This ensures all eigenvalues are in a circle centered at max_degree+1.0
    # and radius < (max_degree+1.0), so guaranteed > 0
    max_degree = max(dict(Graph_u.degree()).values())
    if verbose_level > 0:
        print(f"max degree of graph is: {max_degree}")
    SPD_Prec.tolil()
    SPD_Prec.setdiag(max_degree + 1.0)
    SPD_Prec = sp.csc_matrix(SPD_Prec)

    # PT prec P = LLT
    start = time.time()
    chol_LLT = cholesky(SPD_Prec, ordering_method=ordering_method)
    end = time.time()
    if verbose_level > 0:
        print(f"Permutation optimization took {end - start} seconds")

    Graph_C, perm_compose, P_rev, P_order = find_sparsity_structure_from_chol(
        chol_LLT=chol_LLT
    )

    if verbose_level > 0:
        print(
            f"Parameters in precision: {tril(SPD_Prec).nnz}\n"
            f"Parameters in Cholesky factor: {Graph_C.number_of_edges()}"
        )

    # Return the results
    return Graph_C, perm_compose, P_rev, P_order


def objective_function(C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0) -> float:
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
    C_k[-1] = np.exp(C_k[-1])
    Su = U.dot(C_k)
    n, _ = U.shape
    regularization_l2 = 0.5 * lambda_l2 * np.sum(C_k[:-1] ** 2)
    return 0.5 * np.sum(Su**2) - n * np.log(abs(C_k[-1])) + regularization_l2


def gradient(C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0) -> np.ndarray:
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
    n, _ = U.shape
    C_k[-1] = np.exp(C_k[-1])
    prediction = U.dot(C_k)
    grad = U.T.dot(prediction)
    grad[:-1] += lambda_l2 * C_k[:-1]  # Adjust for L2 regularization
    grad[-1] -= n / C_k[-1]  # Adjust for the -log|C_k,k| term
    grad[-1] *= C_k[-1]  # Adjust for log-transform
    return grad


def hessian(C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0) -> np.ndarray:
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
    n, _ = U.shape
    H = U.T.dot(U)
    np.fill_diagonal(H[:-1, :-1], H.diagonal()[:-1] + lambda_l2)  # L2-term
    C_k[-1] = np.exp(C_k[-1])  # log-transform
    H[-1, -1] += n / (C_k[-1] ** 2)  # Adjust for the -log|C_k,k| term
    H[-1, -1] *= 2.0 * C_k[-1]  # log-transform adjustment
    return H


def optimize_sparse_affine_kr_map(
    U: np.ndarray,
    G: nx.Graph,
    lambda_l2: float = 1.0,
    optimization_method: str = "L-BFGS-B",
    verbose_level: int = 0,
    use_tqdm=True,
) -> csc_matrix:
    """
    Optimizing the affine KR map with standard Gaussian reference  and l2
    regularized dependence.

    Parameters
    ----------
    U : np.ndarray
        The data matrix.
    G : networkx.Graph
        The graph representing the non-zero structure in C.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    scipy.sparse.csc_matrix
        The optimized sparse Cholesky factor of the precision matrix.
    """

    if verbose_level > 0:
        print("Starting statistical fitting of precision")

    _, p = U.shape

    # Initialize a sparse matrix for C_full
    C_full = sp.lil_matrix((p, p))  # lil_matrix for efficient row operations

    loop_function = (
        tqdm(range(p), desc="Learning precision Cholesky factor row-by-row")
        if use_tqdm
        else range(p)
    )

    for k in loop_function:
        non_zero_indices = [j for j in G.neighbors(k) if j < k] + [k]

        # Extract the non-zero elements of C_k
        C_k_reduced = C_full[k, non_zero_indices].toarray().ravel()

        # Extract the reduced version of U
        U_reduced = U[:, non_zero_indices]

        # Optimization for reduced C_k
        lambda_l2_aic = 2.0 * len(non_zero_indices)
        res = minimize(
            fun=objective_function,
            x0=C_k_reduced,
            args=(U_reduced, lambda_l2_aic),
            method=optimization_method,
            jac=gradient,
            # hess=hessian,
            tol=1e-12,
            options={"gtol": 1e-9},
        )

        # Update the full C_k with optimized values
        C_full[k, non_zero_indices] = res.x
        C_full[k, k] = np.exp(C_full[k, k])  # res.x learns log diag

    # Convert to csc_matrix for efficient storage and arithmetic operations
    return C_full.tocsc()


def fit_precision_cholesky(
    U: np.ndarray,
    Graph_u: nx.Graph,
    lambda_l2: float = 1.0,
    ordering_method: str = "metis",
    verbose_level: int = 0,
    use_tqdm=True,
    Graph_C: nx.Graph | None = None,
    perm_compose: np.ndarray | None = None,
    P_rev: csc_matrix | None = None,
    P_order: csc_matrix | None = None,
) -> tuple[csc_matrix, nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
    """
    Estimate the precision matrix using Cholesky decomposition.
    An l2-regularized negative log-likelihood is minimized.

    Parameters
    ----------
    U : The data matrix.
    Graph_u : The graph representing the non-zero structure in the precision
    matrix.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    scipy.sparse.csc_matrix
        Estimated precision matrix.
    """
    _, p = U.shape
    assert len(Graph_u.nodes) == p, "nodes in graph equals columns of data"

    if Graph_C is None or perm_compose is None or P_rev is None or P_order is None:
        # 1. Find in-fill reducing ordering for C
        Graph_C, perm_compose, P_rev, P_order = find_sparsity_structure_from_graph(
            Graph_u,
            verbose_level=verbose_level - 1,
            ordering_method=ordering_method,
        )

    # 2. Estimate non-zeroes of C
    U_perm = U[:, perm_compose]
    C = optimize_sparse_affine_kr_map(
        U_perm,
        Graph_C,
        lambda_l2=lambda_l2,
        verbose_level=verbose_level - 1,
        use_tqdm=use_tqdm,
    )

    # 2.b Compute log-determinant of estimate, for logging
    if verbose_level > 0:
        L_r = P_rev @ C.T @ P_rev  # Factor of reverse precision
        prec_logdet = 2.0 * np.sum(np.log(L_r.diagonal()))
        print(f"Precision has log-determinant: {prec_logdet}")

    # 3. Unwrap C to yield precision
    Prec = P_order @ P_rev @ (C.T @ C) @ P_rev @ P_order.T
    return Prec, Graph_C, perm_compose, P_rev, P_order


def fit_precision_cholesky_approximate(
    U: np.ndarray,
    G: nx.Graph,
    neighbourhood_expansion: int = 2,
    optimization_method: str = "L-BFGS-B",
    verbose_level: int = 0,
    use_tqdm=True,
) -> csc_matrix:
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
    scipy.sparse.csc_matrix
        The optimized sparse Cholesky factor of the precision matrix.
    """

    # Assuming Graph_u already exists
    G_expanded = nx.Graph()

    for node in G.nodes():
        hops = nx.single_source_shortest_path_length(
            G, node, cutoff=neighbourhood_expansion
        )

        for neighbor in hops:
            # Add the nodes and edges between them
            G_expanded.add_node(node)
            G_expanded.add_node(neighbor)
            G_expanded.add_edge(node, neighbor)

    C = optimize_sparse_affine_kr_map(
        U,
        G_expanded,
        optimization_method=optimization_method,
        verbose_level=verbose_level - 1,
        use_tqdm=use_tqdm,
    )

    Prec_approx = C.T @ C

    return Prec_approx
