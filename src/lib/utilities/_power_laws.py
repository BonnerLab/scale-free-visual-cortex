from collections.abc import Callable

import numpy as np
from sklearn.decomposition import PCA


def compute_best_fit(
    x: np.ndarray,
    y: np.ndarray,
    /,
) -> tuple[Callable[[float], float], float]:
    good_data = (y.flatten() > 0) & (x.flatten() > 0)
    x = np.log(x.flatten()[good_data])
    y = np.log(y.flatten()[good_data])

    pca = PCA()
    pca.fit(np.stack([x, y], axis=-1))
    m = float(pca.components_[0][1] / pca.components_[0][0])
    c = y.mean() - m * x.mean()

    return lambda x: np.exp(m * np.log(x) + c), m
