import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def linear_l1_regression(U, Y):
    """
    Performs LASSO regression for each response in Y against predictors in U,
    constructing a sparse matrix of regression coefficients.

    The function scales features in U using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of U. This
    extracts the effect of each feature in U on each response in Y, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (n_responses, n_features) with re-scaled LASSO
        regression coefficients for each response in Y.

    Raises
    ------
    AssertionError
        If the number of samples in U and Y do not match, or if the shape of
        H_sparse is not (m, p).
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    assert n == n_y, "Number of samples in U and Y must be the same"

    scaler = StandardScaler()
    U_scaled = scaler.fit_transform(U)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(
        range(m), desc="Learning sparse linear map for each response"
    ):
        y_j = Y[:, j]

        # Learn individual regularization and fit
        eps = 1e-3
        max_iter = 10000
        model_cv = LassoCV(
            cv=10, fit_intercept=False, max_iter=max_iter, eps=eps
        )
        model_cv.fit(U_scaled, y_j)

        # Extract coefficients
        for non_zero_ind in model_cv.coef_.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                model_cv.coef_[non_zero_ind] / scaler.scale_[non_zero_ind]
            )

    values_H, i_H, j_H = np.array(values_H), np.array(i_H), np.array(j_H)
    H_sparse = sp.csc_matrix((values_H, (i_H, j_H)), shape=(m, p))

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    return H_sparse
