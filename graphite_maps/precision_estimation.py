import numpy as np
import networkx as nx
import scipy.sparse as sp
import time

from scipy.optimize import minimize
from sksparse.cholmod import cholesky
from tqdm import tqdm
from scipy.sparse import csc_matrix, tril

from typing import Tuple, Any, Optional
from numpy.typing import NDArray


def graph_of_permuted_matrix(
    Graph_original: nx.Graph, P_rev: csc_matrix, P_order: csc_matrix
) -> nx.Graph:
    A = nx.to_scipy_sparse_array(Graph_original, weight=None)
    A.data[:] = 1
    A = P_rev @ P_order.T @ A @ P_order @ P_rev
    A = A.tolil()
    G_u_perm = nx.from_scipy_sparse_array(A)
    G_u_perm.remove_edges_from(nx.selfloop_edges(G_u_perm))
    return G_u_perm


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
    offdiag_abs_sum = np.abs(prec).sum(axis=1).A.ravel() - prec.diagonal()
    for i in range(prec.shape[0]):
        if offdiag_abs_sum[i] >= prec[i, i]:
            prec[i, i] = offdiag_abs_sum[i] + eps
    return prec


def find_sparsity_structure_from_chol(
    chol_LLT: csc_matrix,
) -> Tuple[nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
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
    P_rev = sp.csc_matrix(
        (np.ones(p), (perm_reverse, np.arange(p))), shape=(p, p)
    )

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
        print(f"Permutation optimization took {end-start} seconds")

    Graph_C, perm_compose, P_rev, P_order = find_sparsity_structure_from_chol(
        chol_LLT=chol_LLT
    )

    if verbose_level > 0:
        print(
            f"Parameters in precision: {tril(SPD_Prec).nnz}\n"
            f"Parameters in Cholesky factor: {Graph_C.number_of_edges() + Graph_C.number_of_nodes()}"
        )

    # Return the results
    return Graph_C, perm_compose, P_rev, P_order


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
    # C_k[-1] = np.exp(C_k[-1])
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
    # C_k[-1] = np.exp(C_k[-1])
    prediction = U.dot(C_k)
    grad = U.T.dot(prediction)
    grad[:-1] += lambda_l2 * C_k[:-1]  # Adjust for L2 regularization
    grad[-1] -= n / C_k[-1]  # Adjust for the -log|C_k,k| term
    # grad[-1] *= C_k[-1]  # Adjust for log-transform
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
    # C_k[-1] = np.exp(C_k[-1])  # log-transform
    H[-1, -1] += n / (C_k[-1] ** 2)  # Adjust for the -log|C_k,k| term
    # H[-1, -1] *= 2.0 * C_k[-1]  # log-transform adjustment
    return H


def objective_full(C_vals, U, rows, cols, lambda_l2):
    n, p = U.shape
    C = sp.csc_matrix((C_vals, (rows, cols)), shape=(p, p))
    C = C + sp.diags(np.exp(C.diagonal()) - C.diagonal(), 0)
    # C.setdiag( np.exp(C.diagonal()))

    regularization = 0.0  # lambda_l2 * np.sum(C_vals**2)
    Su = C.dot(U.T)
    log_det = n * np.sum(np.log(C.diagonal()))
    return 0.5 * np.sum(Su**2) + regularization - log_det


def gradient_full(C_vals, U, rows, cols, lambda_l2):
    n, p = U.shape
    C = sp.csc_matrix((C_vals, (rows, cols)), shape=(p, p))
    C = C + sp.diags(np.exp(C.diagonal()) - C.diagonal(), 0)

    grad_mat = U.T.dot(U.dot(C.T.A))  # this becomes pxp
    grad = np.zeros(len(C_vals))
    for i, (r, c) in enumerate(zip(rows, cols)):
        grad[i] = grad_mat[c, r]
        if r == c:
            grad[i] -= n / C[c, r]  # -log diag
            grad[i] *= C[r, r]  # adjust for log-transform
    return grad


def orthogonality_constraint(C_vals, rows, cols, p, k, j):
    C = sp.csc_matrix((C_vals, (rows, cols)), shape=(p, p))
    C = C + sp.diags(np.exp(C.diagonal()) - C.diagonal(), 0)
    return C[:, k].T.dot(C[:, j])[0, 0]


def precision_constraints(C_vals, rows, cols, p, k, j, prec_value_kj):
    # print("rows", rows, "cols", cols, "p", p, "vals", C_vals)
    C = sp.csc_matrix((C_vals, (rows, cols)), shape=(p, p))
    C = C + sp.diags(np.exp(C.diagonal()) - C.diagonal(), 0)
    return C[:, k].T.dot(C[:, j])[0, 0] - prec_value_kj


def generate_ortho_constraints(G_C, G_u_perm, rows, cols):
    constraints = []
    p = G_C.number_of_nodes()
    for k in range(p):
        diff_neighbors = set(G_C.neighbors(k)).difference(
            set(G_u_perm.neighbors(k))
        )
        for j in diff_neighbors:
            if j < k:  # To ensure each pair is only considered once
                constraint = {
                    "type": "eq",
                    "fun": orthogonality_constraint,
                    "args": (rows, cols, p, k, j),
                }
                constraints.append(constraint)
    return constraints


def generate_prec_constraints(constraints_info, rows, cols):
    constraints = []
    p = int(np.max(rows)) + 1  # could also just fetch k
    for info in constraints_info:
        j, k, prec_value_kj = info
        constraint = {
            "type": "eq",
            "fun": precision_constraints,
            "args": (rows, cols, p, k, j, prec_value_kj),
        }
        constraints.append(constraint)
    return constraints


def optimize_sparse_affine_kr_map(
    U: np.ndarray,
    G_C: nx.Graph,
    G_u_perm: nx.Graph,
    constraints_info: list = [],
    lambda_l2: float = 1.0,
    optimization_method: str = "L-BFGS-B",
    verbose_level: int = 0,
    use_tqdm=True,
    rowbyrow: bool = True,
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

    if rowbyrow:

        # Initialize a sparse matrix for C_full
        C_full = sp.lil_matrix(
            (p, p)
        )  # lil_matrix for efficient row operations
        C_full.setdiag(0.5)

        loop_function = (
            tqdm(
                range(p), desc="Learning precision Cholesky factor row-by-row"
            )
            if use_tqdm
            else range(p)
        )

        for k in loop_function:
            non_zero_indices = [j for j in G_C.neighbors(k) if j < k] + [k]

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
            # C_full[k, k] = np.exp(C_full[k, k])  # res.x learns log diag

    else:
        C = nx.to_scipy_sparse_array(G_C)
        C = sp.tril(C)
        C.data[:] = 0.001
        C.setdiag(0.1)
        C_vals = C.data
        rows, cols = C.nonzero()

        # constraints = generate_constraints(C_vals, rows, cols, p, G_C, G_u_perm)
        constraints_chol = generate_ortho_constraints(
            G_C, G_u_perm, rows, cols
        )
        if verbose_level > 0:
            print(f"Number of chol constraints: {len(constraints_chol)}")

        constraints_prec = generate_prec_constraints(
            constraints_info, rows, cols
        )
        if verbose_level > 0:
            print(f"Number of prec constraints: {len(constraints_prec)}")

        lambda_l2 = 0.0

        res = minimize(
            fun=objective_full,
            x0=C_vals,
            args=(U, rows, cols, lambda_l2),
            jac=gradient_full,
            constraints=constraints_chol + constraints_prec,
            options={"disp": False},
        )
        # print("optim: ", res)
        C_full = sp.csc_matrix((res.x, (rows, cols)), shape=(p, p))
        C_full = C_full + sp.diags(
            np.exp(C_full.diagonal()) - C_full.diagonal(), 0
        )

    # Convert to csc_matrix for efficient storage and arithmetic operations
    return C_full.tocsc()


def fit_precision_cholesky(
    U: np.ndarray,
    Graph_u: nx.Graph,
    constraints_info: list = [],
    lambda_l2: float = 1.0,
    ordering_method: str = "metis",
    verbose_level: int = 0,
    use_tqdm=True,
    Graph_C: Optional[nx.Graph] = None,
    perm_compose: Optional[np.ndarray] = None,
    P_rev: Optional[csc_matrix] = None,
    P_order: Optional[csc_matrix] = None,
    rowbyrow: bool = True,
) -> Tuple[csc_matrix, nx.Graph, NDArray[Any], csc_matrix, csc_matrix]:
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

    if (
        Graph_C is None
        or perm_compose is None
        or P_rev is None
        or P_order is None
    ):
        # 1. Find in-fill reducing ordering for C
        Graph_C, perm_compose, P_rev, P_order = (
            find_sparsity_structure_from_graph(
                Graph_u,
                verbose_level=verbose_level - 1,
                ordering_method=ordering_method,
            )
        )

    # 2. Estimate non-zeroes of C
    U_perm = U[:, perm_compose]
    G_u_perm = graph_of_permuted_matrix(Graph_u, P_rev, P_order)
    C = optimize_sparse_affine_kr_map(
        U=U_perm,
        G_C=Graph_C,
        G_u_perm=G_u_perm,
        constraints_info=constraints_info,
        lambda_l2=lambda_l2,
        verbose_level=verbose_level - 1,
        use_tqdm=use_tqdm,
        rowbyrow=rowbyrow,
    )

    # 2.b Compute log-determinant of estimate, for logging
    if verbose_level > 0:
        L_r = P_rev @ C.T @ P_rev  # Factor of reverse precision
        prec_logdet = 2.0 * np.sum(np.log(L_r.diagonal()))
        print(f"Precision has log-determinant: {prec_logdet}")

    # 3. Unwrap C to yield precision
    Prec = P_order @ P_rev @ (C.T @ C) @ P_rev @ P_order.T
    return Prec, Graph_C, perm_compose, P_rev, P_order
