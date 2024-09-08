from collections.abc import Sequence

import numpy as np


def assign_data_to_geometrically_spaced_bins(
    data: Sequence[float],
    /,
    *,
    n: int | None = None,
    base: int = 10,
    density: float | None = None,
    start: float | None = None,
    stop: float | None = None,
) -> np.ndarray[float]:
    """Assign each data point to a bin, where bins are spaced geometrically.

    The bins are geometrically spaced from `start` to `stop` (inclusive). The number
    of bins is determined by `n`, if provided, or `density` and `base` otherwise.
    For example, if `base == 10` and `density == 3`, there will be about 3 bins
    every decade.

    Args:
    ----
        data: Data points to be assigned to bins.
        n: The number of bins to use. Defaults to None.
        density: The approximate number of bins within each level.
        base: The base used for logarithmic binning. Defaults to 10.
        start: The start of the logarithmic scale. Defaults to `min(data)`.
        stop: The end of the logarithmic scale. Defaults to `max(data)`.

    Returns:
    -------
        The bin centers corresponding to each data point, where each bin center is the
        geometric mean of the edges corresponding to the bin.

    """
    start = min(data) if start is None else start
    stop = max(data) if stop is None else stop

    if n is None:
        if density is None:
            error = "At least one of `n` and `density` must be provided."
            raise ValueError(error)

        n_levels = (np.log(stop) - np.log(start)) / np.log(base)
        n = int(density * n_levels)

    bin_edges = np.geomspace(start, stop, num=n)
    bin_centers = np.exp(np.log(bin_edges)[:-1] + np.diff(np.log(bin_edges)) / 2)

    bin_edges[-1] = np.inf
    return bin_centers[np.digitize(data, bin_edges) - 1]
