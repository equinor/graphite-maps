import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.sparse import spmatrix
from scipy.stats import chi2
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def linear_l1_regression(U, Y, verbose_level: int = 0):
    """Performs LASSO regression for each response in Y against predictors in
    U, constructing a sparse matrix of regression coefficients.

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
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.

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

    if verbose_level > 0:
        print(f"Learning sparse linear map of shape {(m, p)}")

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        y_j = Y_scaled[:, j]

        # Learn individual regularization and fit
        eps = 1e-3
        max_iter = 10000
        model_cv = LassoCV(cv=10, fit_intercept=False, max_iter=max_iter, eps=eps)
        model_cv.fit(U_scaled, y_j)

        # Extract coefficients
        for non_zero_ind in model_cv.coef_.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                scaler_y.scale_[j]
                * model_cv.coef_[non_zero_ind]
                / scaler_u.scale_[non_zero_ind]
            )

    H_sparse = sp.csc_matrix(
        (np.array(values_H), (np.array(i_H), np.array(j_H))), shape=(m, p)
    )

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    if verbose_level > 0:
        print(
            f"Total elements: {m * p}\n"
            f"Non-zero elements: {H_sparse.nnz}\n"
            f"Fraction of non-zeros: {H_sparse.nnz / (m * p)}"
        )

    return H_sparse


def expected_max_chisq(p):
    """Expected maximum of p central chi-square(1) random variables."""

    def dmaxchisq(x):
        return 1.0 - np.exp(p * chi2.logcdf(x, df=1))

    expectation, _ = quad(dmaxchisq, 0, np.inf)
    return expectation


def mse(residuals):
    return 0.5 * np.mean(residuals**2)


def calculate_psi_M(x, y, beta_estimate):
    """The psi/score function for mse: 0.5*residual**2."""
    residuals = y - beta_estimate * x
    psi = -residuals * x
    M = -np.mean(x**2)
    return psi, M


def calculate_influence(x, y, beta_estimate):
    """The influence of (x, y) on beta_estimate as an mse M-estimator."""
    psi, M = calculate_psi_M(x, y, beta_estimate)
    return psi / M


def boost_linear_regression(
    X, y, learning_rate=0.5, tol=1e-6, max_iter=10000, effective_dimension=None
):
    """Boost coefficients of linearly regressing y on standardized X.

    The coefficient selection utilizes information theoretic weighting. The
    stopping criterion utilizes information theoretic loss-reduction.
    """
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    residuals = y.copy()
    residuals_loo = y.copy()

    # A stricter criterion is the loo-adjustment: mse(residuals_loo)-mse
    # (residuals). This converges to TIC. Under certain conditions this is AIC.
    # At worst, we are maximizing squares. See Lunde 2020 Appendix A. This
    #  needs to be adjusted for.
    # The mse_factor adjusts for this.
    # if effective_dimension is None:
    #    effective_dimension = n_features
    # mse_factor = expected_max_chisq(np.ceil(effective_dimension))

    for _ in range(max_iter):
        coef_changes = np.dot(X.T, residuals) / n_samples

        # Could be adjusted for IC -- some features already included
        # The IC would build in additional motivation for sparsity
        feature_evaluation = np.abs(coef_changes)

        # Select feature based on loss criterion
        best_feature = np.argmax(feature_evaluation)
        beta_estimate = coef_changes[best_feature]

        # adjust to loo estimates for coef_change
        influence = calculate_influence(X[:, best_feature], residuals, beta_estimate)
        beta_estimate_loo = beta_estimate - influence / n_samples

        # residuals_full = residuals - beta_estimate * X[:, best_feature]
        residuals_full_loo = (
            residuals_loo - learning_rate * beta_estimate_loo * X[:, best_feature]
        )

        if mse(residuals_loo) < mse(residuals_full_loo):
            break

        # Check if adding the full weight of the feature would decrease loss
        # if mse(residuals) < mse(residuals_full) + mse_factor * (
        #    mse(residuals_full_loo) - mse(residuals_full)
        # ):
        #    break

        coef_change = beta_estimate * learning_rate
        coef_change_loo = beta_estimate_loo * learning_rate

        # Check for convergence
        if np.abs(coef_change) < tol:
            break
        else:
            # Update
            residuals -= coef_change * X[:, best_feature]
            coefficients[best_feature] += coef_change

            # loo update
            residuals_loo -= coef_change_loo * X[:, best_feature]

    # ensure cutoff values -- very small if data standardized
    # prefer sparsity
    cutoff = 2.0 * learning_rate / np.sqrt(n_samples)  # 2.0: 95% ci-ish
    coefficients[np.abs(coefficients) < cutoff] = 0

    return coefficients


def linear_boost_ic_regression(
    U, Y, learning_rate=0.5, effective_dimension=None, verbose_level: int = 0
):
    """Performs boosted linear regression for each response in Y against
    predictors in U, constructing a sparse matrix of regression coefficients.
    The complexity is tuned with an information theoretic approach.

    The function scales features in U using standard scaling before learning
    the coefficients, then re-scales the coefficients to the original scale of
    U. This extracts the effect of each feature in U on each response in Y,
    ignoring intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.

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

    if verbose_level > 0:
        print(f"Learning sparse linear map of shape {(m, p)}")

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        y_j = Y_scaled[:, j]

        # Learn individual fit
        coefficients_j = boost_linear_regression(
            U_scaled,
            y_j,
            learning_rate=learning_rate,
            effective_dimension=effective_dimension,
        )

        # Extract coefficients
        for non_zero_ind in coefficients_j.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                scaler_y.scale_[j]
                * coefficients_j[non_zero_ind]
                / scaler_u.scale_[non_zero_ind]
            )

    H_sparse = sp.csc_matrix(
        (np.array(values_H), (np.array(i_H), np.array(j_H))), shape=(m, p)
    )

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    if verbose_level > 0:
        print(
            f"Total elements: {m * p}\n"
            f"Non-zero elements: {H_sparse.nnz}\n"
            f"Fraction of non-zeros: {H_sparse.nnz / (m * p)}"
        )

    return H_sparse


def response_residual(
    U: np.ndarray, Y: np.ndarray, H: spmatrix, verbose_level: int = 0
) -> np.ndarray:
    """Residual from regression H for Y on U"""
    n_u, p = U.shape
    n_y, m = Y.shape
    assert n_u == n_y, "Number of ensembles must be the same"
    assert (m, p) == H.shape, "Coefficients in H must match U and Y dimensions"

    if verbose_level > 0:
        print("Calculating response residuals")

    return Y - U @ H.T


def residual_variance(
    U: np.ndarray, Y: np.ndarray, H: spmatrix, verbose_level: int = 0
) -> np.ndarray:
    """Variance in Y not explained by U through H"""

    n_u, p = U.shape
    n_y, m = Y.shape
    assert n_u == n_y, "Number of ensembles must be the same"
    assert (m, p) == H.shape, "Coefficients in H must match U and Y dimensions"

    R = response_residual(U, Y, H)
    unexplained_variance = np.var(R, axis=0)

    assert unexplained_variance.shape == (m,), (
        "Number of variance components must match number of observations"
    )

    if verbose_level > 0:
        print("Calculating unexplained variance")

    return unexplained_variance
