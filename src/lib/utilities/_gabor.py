from collections.abc import Mapping, Sequence
from hashlib import blake2b

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from bonner.caching import cache
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib.datasets import StimulusSet


def create_gabor_filter(
    *,
    size: int,
    x: float,  # in range [-0.5, 0.5]
    y: float,  # in range [-0.5, 0.5]
    aspect_ratio: float,
    orientation: float,
    frequency: float,
    scale: float,
    phase: float = 0,
) -> npt.NDArray[np.floating]:
    grid_sampling = np.linspace(-0.5, 0.5, num=size)
    x_grid, y_grid = np.meshgrid(grid_sampling - x, grid_sampling - y)

    cos, sin = np.cos(orientation), np.sin(orientation)

    x_prime = x_grid * cos + y_grid * sin
    y_prime = -x_grid * sin + y_grid * cos

    return np.exp(
        -(x_prime**2 + aspect_ratio**2 * y_prime**2) / (2 * scale**2),
    ) * np.cos(
        2 * np.pi * frequency * x_prime + phase,
    )


def standardize_filter(filter_: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return (filter_ - filter_.mean()) / filter_.std()


def create_gabor_filter_bank(
    *,
    size: int,
    parameters: Sequence[Mapping[str, float]],
) -> xr.DataArray:
    parameters_dict_of_lists = {
        parameter: np.array([float(dict_[parameter]) for dict_ in parameters])
        for parameter in parameters[0]
    }
    parameters_dict_of_lists = dict(sorted(parameters_dict_of_lists.items()))

    hash_ = blake2b(
        np.stack(
            list(parameters_dict_of_lists.values()),
            axis=0,
        ).tobytes(),
        digest_size=4,
    ).hexdigest()

    name = f"size={size}.parameters={hash_}"
    cacher = cache(f"gabor_filter_bank/filter_banks/{name}.nc")

    filter_bank = cacher(_create_gabor_filter_bank)(size=size, parameters=parameters)
    return filter_bank.rename(f"{filter_bank.name}.size={size}.parameters={hash_}")


def _create_gabor_filter_bank(
    *,
    size: int,
    parameters: Sequence[Mapping[str, float]],
) -> xr.DataArray:
    parameters_dict_of_lists = {
        parameter: np.array([float(dict_[parameter]) for dict_ in parameters])
        for parameter in parameters[0]
    }

    filter_bank = xr.DataArray(
        name="gabor_filter_bank",
        data=np.empty(shape=(len(parameters), size, size), dtype=np.float32),
        dims=("filter", "height", "width"),
        coords={
            parameter: ("filter", values)
            for parameter, values in parameters_dict_of_lists.items()
        },
    )

    for i_filter, parameters_ in enumerate(parameters):
        filter_bank.data[i_filter, ...] = create_gabor_filter(
            size=size,
            **parameters_,
        )

    return filter_bank


@cache(
    "gabor_filter_bank/activations/{filter_bank_identifier}/{stimulus_set_identifier}.nc",
    helper=lambda kwargs: {
        "filter_bank_identifier": kwargs["filter_bank"].name,
        "stimulus_set_identifier": kwargs["stimulus_set"].identifier,
    },
)
def apply_gabor_filter_bank(
    filter_bank: xr.DataArray,
    *,
    stimulus_set: StimulusSet,
    dataloader: DataLoader,
) -> xr.DataArray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filter_bank_torch = torch.from_numpy(filter_bank.to_numpy()).to(device)

    output = xr.DataArray(
        name=f"{filter_bank.name}.{stimulus_set.identifier}",
        data=np.empty(
            shape=(len(stimulus_set), filter_bank.sizes["filter"]),
            dtype=np.float32,
        ),
        dims=("stimulus", "filter"),
        coords={
            coord: ("filter", filter_bank[coord].to_numpy())
            for coord in filter_bank.coords
        },
    )

    i_stimulus = 0
    for batch in tqdm(dataloader, desc="batch", leave=False):
        output.data[i_stimulus : i_stimulus + len(batch), ...] = (
            torch.einsum(
                "bchw,fhw->bcf",
                batch.to(device),
                filter_bank_torch,
            )
            .cpu()
            .numpy()
            .squeeze()
        )
        i_stimulus += len(batch)

    return output
