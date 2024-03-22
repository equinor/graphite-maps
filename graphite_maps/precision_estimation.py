import numpy as np
import networkx as nx
import scipy.sparse as sp

from scipy.optimize import minimize
from sksparse.cholmod import cholesky
from tqdm import tqdm
from scipy.sparse import lil_matrix, csc_matrix, tril

from typing import Tuple, Any
from numpy.typing import NDArray


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
    # Determine size, p, from the number of nodes in the graph
    p = len(graph.nodes)

    # Initialize a LIL sparse matrix for easy value assignment
    prec_matrix = lil_matrix((p, p))

    # Iterate over the edges and set the corresponding elements to 1
    for i, j in graph.edges():
        prec_matrix[i, j] = 1
        prec_matrix[j, i] = 1  # Ensure the matrix is symmetric

    # Diagonal elements should be set to 1 (self-connection)
    for i in range(p):
        prec_matrix[i, i] = 1

    # Convert the matrix to CSC format for efficient arithmetic and storage
    return prec_matrix.tocsc()


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
    graph = nx.Graph()

    # Iterate over non-zero elements of the matrix
    rows, cols = precision_matrix.nonzero()
    for i, j in zip(rows, cols):
        if i <= j:  # Avoid double counting in symmetric matrix
            graph.add_edge(i, j)

    return graph


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
    offdiag_abs_sum = np.abs(prec).sum(axis=1).A.ravel() - prec.diagonal()
    for i in range(prec.shape[0]):
        if offdiag_abs_sum[i] > prec[i, i]:
            prec[i, i] = offdiag_abs_sum[i] + eps
    return prec


def find_sparsity_structure(
    Graph_u: nx.Graph,
    ordering_method: str = "best",
    verbose_level: int = 0,
) -> Tuple[nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
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
    P_amd : scipy.sparse.csc_matrix
        The AMD permutation matrix.
    """

    # Matrices are pxp
    p = len(Graph_u.nodes)

    # Create SPD matrix with same sparsity structure as Prec
    SPD_Prec = graph_to_precision_matrix(Graph_u)
    SPD_Prec = gershgorin_spd_adjustment(SPD_Prec)

    # Create the reverse order permutation array
    perm_reverse = np.arange(p - 1, -1, -1)
    # Create the reverse permutation matrix
    P_rev = sp.csc_matrix(
        (np.ones(p), (perm_reverse, np.arange(p))), shape=(p, p)
    )

    # PT prec P = LLT
    chol_LLT = cholesky(SPD_Prec, ordering_method=ordering_method)

    # Get the AMD permutation vector
    perm_amd = chol_LLT.P()

    # Create the permutation matrix P
    P_amd = sp.csc_matrix(
        (np.ones(len(perm_amd)), (perm_amd, np.arange(len(perm_amd)))),
        shape=SPD_Prec.shape,
    )

    # Extract the lower triangular Cholesky factor
    L = chol_LLT.L()

    # Apply the AMD permutation to the reverse permutation
    perm_compose = perm_amd[perm_reverse]

    # Create matrix of non-zeroes equalling C: LLT=CTC unique when SPD
    C_pattern = (P_rev @ L @ P_rev).T

    # Extract structure into a graph for C
    G_C = nx.Graph()
    G_C.add_nodes_from(range(p))
    # Add edges exploiting the sparsity and lower triangular structure
    rows, cols = C_pattern.nonzero()
    for i, j in zip(rows, cols):
        if i > j:  # Ensure lower triangular structure
            G_C.add_edge(i, j)

    if verbose_level > 0:
        print(
            f"Parameters in precision: {tril(SPD_Prec).nnz}\n"
            f"Parameters in Cholesky factor: {L.nnz}"
        )

    # Return the results
    return G_C, perm_compose, P_rev, P_amd


def objective_function(
    C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0
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
    Su = U.dot(C_k)
    n, _ = U.shape
    regularization_l2 = 0.5 * lambda_l2 * np.sum(C_k[:-1] ** 2)
    return 0.5 * np.sum(Su**2) - n * np.log(abs(C_k[-1])) + regularization_l2


def gradient(
    C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0
) -> np.ndarray:
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
    prediction = U.dot(C_k)
    grad = U.T.dot(prediction)
    grad[:-1] += lambda_l2 * C_k[:-1]  # Adjust for L2 regularization
    grad[-1] -= n / C_k[-1]  # Adjust for the -log|C_k,k| term
    return grad


def hessian(
    C_k: np.ndarray, U: np.ndarray, lambda_l2: float = 1.0
) -> np.ndarray:
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
    H[-1, -1] += n / (C_k[-1] ** 2)  # Adjust for the -log|C_k,k| term
    return H


def optimize_sparse_affine_kr_map(
    U: np.ndarray,
    G: nx.Graph,
    lambda_l2: float = 1.0,
    optimization_method: str = "L-BFGS-B",
    verbose_level: int = 0,
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
    C_full.setdiag(1)  # Set diagonal to 1

    for k in tqdm(
        range(p), desc="Learning precision Cholesky factor row-by-row"
    ):
        non_zero_indices = [j for j in G.neighbors(k) if j < k] + [k]
        # print(non_zero_indices)

        # Extract the non-zero elements of C_k
        C_k_reduced = C_full[k, non_zero_indices].toarray().ravel()

        # Extract the reduced version of U
        U_reduced = U[:, non_zero_indices]

        # Optimization for reduced C_k
        lambda_l2_aic = len(non_zero_indices)
        res = minimize(
            fun=objective_function,
            x0=C_k_reduced,
            args=(U_reduced, lambda_l2_aic),
            method=optimization_method,
            jac=gradient,
            hess=hessian,
            options={"gtol": 1e-4, "xtol": 1e-4, "barrier_tol": 1e-4},
        )

        # Update the full C_k with optimized values
        C_full[k, non_zero_indices] = res.x

    # Convert to csc_matrix for efficient storage and arithmetic operations
    return C_full.tocsc()


def fit_precision_cholesky(
    U: np.ndarray,
    Graph_u: nx.Graph,
    lambda_l2: float = 1.0,
    verbose_level: int = 0,
) -> np.ndarray:
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

    # 1. Find permutation yielding AMD ordering for C
    Graph_C, perm_compose, P_rev, P_amd = find_sparsity_structure(
        Graph_u, verbose_level=verbose_level - 1
    )

    # 2. Estimate non-zeroes of C
    U_perm = U[:, perm_compose]
    C = optimize_sparse_affine_kr_map(
        U_perm, Graph_C, lambda_l2=lambda_l2, verbose_level=verbose_level - 1
    )

    # 2.b Compute log-determinant of estimate, for logging
    if verbose_level > 0:
        L_r = P_rev @ C.T @ P_rev  # Factor of reverse precision
        prec_logdet = 2.0 * np.sum(np.log(L_r.diagonal()))
        print(f"Precision has log-determinant: {prec_logdet}")

    # 3. Unwrap C to yield precision
    return P_amd @ P_rev @ (C.T @ C) @ P_rev @ P_amd.T
