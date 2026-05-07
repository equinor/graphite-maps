import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import sparray

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def generate_gaussian_noise(
    n: int, Prec: sparray, seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generates 'n' samples of Gaussian noise with precision 'Prec'.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    Prec : scipy.sparse.sparray
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

    log.info("Sampling with seed=%s", seed)
    log.info("Sampled Gaussian noise with shape=%s", eps.shape)

    return eps
