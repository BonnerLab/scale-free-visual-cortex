r"""
Efficiently applying a kernel to data not on a grid.

I want to filter my data $X \\in \\mathbb{R}^N$ using a kernel $K(x_i, x_j)$

But my data has holes.

So I need to compute, for all $x_i$:

$$
\begin{align}
    \frac{\\sum_j K(x_i, x_j) x_i}{\\sum_j K(x_i, x_j)}
    = \\sum_j \\left( \frac{K(x_i, x_j)}{\\sum_k K(x_i, x_k)} \right) x_i
\\end{align}
$$

1. Compute the pairwise distance matrix $\\Delta$ where $\\Delta_{ij} = ||x_i - x_j||$.
2. Compute the kernel matrix $K$ where $K_{ij} = K(x_i, x_j)$. This can be done efficiently since $K(x_i, x_j)$ is a function of $\\Delta_{ij}$.
3. Normalize the kernel matrix by dividing by its row-/column-sum: $K_{\text{normalized}} = K / $
"""

import itertools

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from bonner.computation.cuda import try_devices
from tqdm.auto import tqdm


def _gaussian(
    x: npt.NDArray[np.number],
    *,
    sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    result = np.exp(-(x**2) / (2 * sigma**2))
    if normalize:
        result /= np.sqrt(2 * np.pi * sigma**2)
    return result


def _difference_of_gaussians(
    x: npt.NDArray[np.number],
    *,
    sigma_1: float,
    sigma_2: float,
    normalize: bool = True,
) -> np.ndarray:
    return _gaussian(
        x,
        sigma=sigma_1,
        normalize=normalize,
    ) - _gaussian(
        x,
        sigma=sigma_2,
        normalize=normalize,
    )


def spatially_smooth_data(
    data: xr.DataArray,
    *,
    sigma: float,
) -> xr.DataArray:
    pairwise_distances = compute_pairwise_distances(data)

    kernel = _gaussian(pairwise_distances, sigma=sigma, normalize=False)
    kernel = _normalize_kernel(kernel)
    kernel = torch.from_numpy(kernel)

    return xr.DataArray(
        name=f"{data.name}.smoothed_sigma={sigma}",
        data=apply_kernel(torch.from_numpy(data.data), kernel=kernel.T).cpu().numpy(),
        dims=data.dims,
        coords=data.coords,
    )


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    return kernel / kernel.sum(axis=-1, keepdims=True)


def compute_pairwise_distances(
    data: xr.DataArray,
    *,
    dims: tuple[str, ...] = ("x", "y", "z"),
) -> npt.NDArray[np.floating]:
    return np.sqrt(
        np.stack([
            np.subtract.outer(
                data[dim].data.astype(np.int64),
                data[dim].data.astype(np.int64),
            )
            ** 2
            for dim in dims
        ]).sum(axis=0),
    )


@try_devices
def apply_kernel(
    data: torch.Tensor,
    *,
    kernel: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return kernel.to(dtype=dtype) @ data.to(dtype=dtype)


def _compute_variance_decay_with_spatial_smoothing(
    data: torch.Tensor,
    *,
    sigma: float,
    pairwise_distances: npt.NDArray[np.number],
    original_variance: torch.Tensor,
) -> xr.DataArray:
    kernel = _gaussian(pairwise_distances, sigma=sigma, normalize=False)
    kernel = _normalize_kernel(kernel)
    kernel = torch.from_numpy(kernel)

    return xr.DataArray(
        name="proportion of remaining variance",
        data=(apply_kernel(data, kernel=kernel).var(dim=0) / original_variance)
        .cpu()
        .numpy(),
        dims=("component",),
        coords={"component": ("component", 1 + np.arange(data.shape[-1]))},
    )


@try_devices
def _helper(
    data: torch.Tensor,
    *,
    pairwise_distances: npt.NDArray[np.floating],
    sigmas: npt.NDArray[np.number],
) -> xr.DataArray:
    original_variance = data.var(dim=0)

    return xr.concat(
        [
            _compute_variance_decay_with_spatial_smoothing(
                data=data,
                sigma=sigma,
                pairwise_distances=pairwise_distances,
                original_variance=original_variance,
            ).expand_dims({"sigma": [sigma]})
            for sigma in tqdm(sigmas, desc="sigma", leave=False)
        ],
        dim="sigma",
    )


def compute_variance_decay_with_spatial_smoothing(
    data: xr.DataArray,
    *,
    sigmas: npt.NDArray[np.number],
) -> xr.DataArray:
    return _helper(
        data=torch.from_numpy(data.to_numpy()),
        pairwise_distances=compute_pairwise_distances(data),
        sigmas=sigmas,
    )


def _compute_variance_decay_with_bandpass_filtering(
    data: torch.Tensor,
    *,
    sigma_1: float,
    sigma_2: float,
    pairwise_distances: npt.NDArray[np.floating],
    original_variance: torch.Tensor,
) -> xr.DataArray:
    kernel = _normalize_kernel(
        _difference_of_gaussians(
            pairwise_distances,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            normalize=False,
        ),
    )
    kernel = torch.from_numpy(kernel)

    return xr.DataArray(
        name="proportion of variance",
        data=(apply_kernel(data, kernel=kernel).var(dim=0) / original_variance)
        .cpu()
        .numpy(),
        dims=("component",),
        coords={"component": ("component", 1 + np.arange(data.shape[-1]))},
    )


def compute_variance_decay_with_bandpass_filtering(
    data: xr.DataArray,
    *,
    sigmas: npt.NDArray[np.number],
) -> xr.DataArray:
    pairwise_distances = compute_pairwise_distances(data)
    data_ = torch.from_numpy(data.data).to("cuda")
    original_variance = data_.var(dim=0)

    variances = []

    for sigma_1, sigma_2 in tqdm(itertools.pairwise(sigmas), desc="sigma", leave=False):
        variances.append(
            _compute_variance_decay_with_bandpass_filtering(
                data=data_,
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                pairwise_distances=pairwise_distances,
                original_variance=original_variance,
            )
            .expand_dims({"sigma": [np.exp(0.5 * (np.log(sigma_1) + np.log(sigma_2)))]})
            .assign_coords(
                {
                    "sigma_low": ("sigma", [sigma_1]),
                    "sigma_high": ("sigma", [sigma_2]),
                },
            ),
        )
    return xr.concat(variances, dim="sigma")
