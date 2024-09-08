import numpy as np
import numpy.typing as npt
from bonner.caching import cache
from scipy.stats import ortho_group


def simulate_data_with_singular_value_spectrum(
    shape: tuple[int, int],
    *,
    singular_values: npt.NDArray[np.floating],
    seed: int = 0,
) -> npt.NDArray[np.floating]:
    n, p = shape
    d = min(n, p)

    if not np.all(singular_values >= 0):
        error = "Expected all singular values to be >= 0"
        raise ValueError(error)

    if not len(singular_values) <= d:
        error = "Expected len(singular values) <= min(shape)"
        raise ValueError(error)

    singular_values = np.pad(
        singular_values,
        (0, d - len(singular_values)),
        mode="constant",
    )

    cacher = cache(
        "orthonormal_matrices/dim={dim}.random_state={random_state}.npy",
    )
    u = cacher(ortho_group.rvs)(dim=n, random_state=seed)[:, :d]
    v = cacher(ortho_group.rvs)(dim=p, random_state=seed + 1)[:, :d]

    return u @ np.diag(singular_values) @ v.T


def simulate_data_with_covariance_spectrum(
    shape: tuple[int, int],
    *,
    covariance_spectrum: npt.NDArray[np.number],
    seed: int = 0,
    ddof: int = 1,
) -> npt.NDArray[np.floating]:
    return simulate_data_with_singular_value_spectrum(
        shape=shape,
        singular_values=np.sqrt((shape[0] - ddof) * covariance_spectrum),
        seed=seed,
    )


def create_power_law_spectrum(
    d: int,
    *,
    exponent: float,
    scale: float = 1,
) -> npt.NDArray[np.floating]:
    ranks = 1 + np.arange(d)
    return scale * np.exp(exponent * np.log(ranks))
