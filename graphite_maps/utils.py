import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix


def generate_gaussian_noise(
    n: int, Prec: spmatrix, seed: int | None = None, verbose_level: int = 0
) -> NDArray[np.floating]:
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
    if n < 1:
        raise ValueError(f"`n` should be g.e. 1, got {n}")

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
