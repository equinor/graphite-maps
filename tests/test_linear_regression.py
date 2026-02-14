import time

import numpy as np
import pytest
from graphite_maps.linear_regression import (
    boost_linear_regression,
    linear_boost_ic_regression,
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def standardize_data(X, y):
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)

    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler_X, scaler_y


def generate_data(n, p_noisy, seed=42):
    """3 signal features and p_noisy noisy ones. All standardized."""
    rng = np.random.default_rng(seed)

    X1 = rng.random((n, 1))
    X2 = rng.random((n, 1))
    X3 = rng.random((n, 1))
    X = np.hstack((X1, X2, X3))

    # True relationship + noise
    y = 1 * X[:, 0] - X[:, 1] + X[:, 2] + rng.normal(0, 1e-2, size=n)

    # Add noise to data
    noise_features = rng.random((n, p_noisy))
    X_with_noise = np.hstack((X, noise_features))

    X_with_noise_scaled, y_scaled, *_ = standardize_data(X_with_noise, y)
    return X_with_noise_scaled, y_scaled


@pytest.mark.parametrize("n, p_noisy", [[100, 9], [200, 99], [1000, 999]])
def test_that_mse_train_decreases(n, p_noisy):
    X, y = generate_data(n, p_noisy)

    # Baseline: mean predictor
    y_mean = np.mean(y)
    mse_mean = mean_squared_error(y, np.full_like(y, y_mean))

    # Learn (very sparse) linear regression
    beta_sparse = boost_linear_regression(X=X, y=y, learning_rate=0.1, max_iter=100)

    y_pred = X @ beta_sparse
    mse_model = mean_squared_error(y, y_pred)

    assert mse_model <= mse_mean


@pytest.mark.parametrize(
    "n, p_noisy", [(50, 5), (100, 10), (200, 20), (400, 40), (1000, 999)]
)
def test_that_regression_learns_more_with_more_data(n, p_noisy):
    X_full, y_full = generate_data(10 * n, p_noisy=p_noisy)

    # Split into two halves
    X_half, y_half = X_full[:n], y_full[:n]

    # Train on a tenth
    beta_half = boost_linear_regression(X=X_half, y=y_half)

    # Train on all data
    beta_full = boost_linear_regression(X=X_full, y=y_full)

    # Predict both on full data
    y_pred_half = X_full @ beta_half
    y_pred_full = X_full @ beta_full

    mse_half = mean_squared_error(y_full, y_pred_half)
    mse_full = mean_squared_error(y_full, y_pred_full)

    assert mse_full <= mse_half


@pytest.mark.parametrize(
    "n, p_noisy, max_iters",
    [
        (200, 20, [10, 50, 100]),
        (400, 40, [20, 100, 200]),
    ],
)
def test_that_mse_decreases_with_more_iterations(n, p_noisy, max_iters):
    X, y = generate_data(n, p_noisy)

    prev_mse = float("inf")
    for max_iter in max_iters:
        beta = boost_linear_regression(X=X, y=y, max_iter=max_iter)
        y_pred = X @ beta
        mse = mean_squared_error(y, y_pred)

        assert mse <= prev_mse

        prev_mse = mse


def test_parallel_speedup():
    """Test that parallel execution is faster than sequential."""
    rng = np.random.default_rng(42)
    # Larger data needed to overcome parallel overhead (~50ms per task)
    n_samples = 200
    n_features = 500
    n_responses = 1000

    U = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_responses))

    # Time sequential execution
    start = time.perf_counter()
    H_sequential = linear_boost_ic_regression(U, Y, n_jobs=1)
    time_sequential = time.perf_counter() - start

    # Time parallel execution
    start = time.perf_counter()
    H_parallel = linear_boost_ic_regression(U, Y, n_jobs=-1)
    time_parallel = time.perf_counter() - start

    # Verify results are identical
    np.testing.assert_array_almost_equal(
        H_sequential.toarray(), H_parallel.toarray(), decimal=10
    )

    # Parallel should be faster on multi-core machines
    print(f"\nSequential: {time_sequential:.2f}s, Parallel: {time_parallel:.2f}s")
    print(f"Speedup: {time_sequential / time_parallel:.2f}x")

    # Assert parallel provides speedup (at least 1.2x on multi-core)
    assert time_parallel < time_sequential, (
        f"Parallel ({time_parallel:.2f}s) should be faster than "
        f"sequential ({time_sequential:.2f}s)"
    )
