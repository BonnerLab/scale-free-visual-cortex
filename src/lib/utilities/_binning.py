import typing

import numpy as np
import numpy.typing as npt
import xarray as xr
from flox.xarray import xarray_reduce

T = typing.TypeVar("T", xr.DataArray, xr.Dataset)


def bin_data[T](
    data: T,
    *,
    bin_edges: dict[str, npt.NDArray[np.number]],
    bin_centers: dict[str, npt.NDArray[np.number]],
    dim: str,
) -> T:
    dims = list(bin_edges.keys())
    binned = (
        xarray_reduce(
            data,
            *dims,
            func="nanmean",
            expected_groups=tuple(bin_edges.values()),
            isbin=True,
        )
        .assign_coords(
            {
                f"{dim_}_bins": (
                    f"{dim_}_bins",
                    bin_centers[dim_],
                )
                for dim_ in dims
            },
        )
        .stack({dim: [f"{dim_}_bins" for dim_ in dims]})
        .rename({f"{dim_}_bins": dim_ for dim_ in dims})
        .dropna(dim=dim, how="all")
    )
    if len(dims) == 1:
        binned = binned.reset_index(dim).rename({dims[0]: dim}).set_index({dim: [dim]})
    return binned


def extract_uniformly_spaced_bins(
    start: float,
    stop: float,
    *,
    n: int | None = None,
    spacing: float | None = None,
    adjust_endpoints: bool = True,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    if n is None:
        if spacing is None:
            error = "exactly one of `spacings` and `n_bins` must be provided"
            raise ValueError(error)
        n = int(np.ceil((stop - start) / spacing))

    bin_edges = np.linspace(start, stop, n + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    if adjust_endpoints:
        bin_edges = expand_endpoints(bin_edges)

    return bin_edges, bin_centers


def extract_geometrically_spaced_bins(
    start: float,
    stop: float,
    n: int | None = None,
    *,
    base: int = 10,
    density: float | None = None,
    adjust_endpoints: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if n is None:
        if density is None:
            error = "exactly one of `n` and `density` must be provided."
            raise ValueError(error)

        n_levels = (np.log(stop) - np.log(start)) / np.log(base)
        n = int(density * n_levels)
    else:
        n += 1

    bin_edges = np.geomspace(start, stop, num=n)
    bin_centers = np.exp(np.log(bin_edges)[:-1] + np.diff(np.log(bin_edges)) / 2)

    if adjust_endpoints:
        bin_edges = expand_endpoints(bin_edges)

    return bin_edges, bin_centers


def expand_endpoints(
    bin_edges: npt.NDArray[np.floating],
    *,
    factor: float = 0.001,
) -> npt.NDArray[np.floating]:
    bin_edges_expanded = bin_edges
    bin_edges_expanded[0] *= 1 - factor
    bin_edges_expanded[-1] *= 1 + factor
    return bin_edges_expanded


def assign_bins(
    data: npt.NDArray[np.floating],
    *,
    bin_edges: npt.NDArray[np.floating],
    bin_centers: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return bin_centers[np.digitize(data, bin_edges) - 1]
