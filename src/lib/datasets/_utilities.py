"""Utility functions."""

import hashlib
import itertools

import numpy as np
import xarray as xr


def filter_by_stimulus(
    data: xr.DataArray,
    /,
    *,
    stimuli: set[int],
) -> xr.DataArray:
    stimuli_sorted = np.array(sorted(stimuli))
    hash_ = hashlib.blake2b(stimuli_sorted.tobytes(), digest_size=4).hexdigest()
    return (
        data.load()
        .isel({"presentation": np.isin(data["stimulus"], stimuli_sorted)})
        .sortby(["stimulus", "repetition"])
        .assign_attrs({"stimuli": hash_})
        .rename(f"{data.name}.stimuli={hash_}")
    )


def filter_by_repetition(data: xr.DataArray, /, *, repetition: int) -> xr.DataArray:
    return (
        data.load()
        .isel(presentation=data["repetition"] == repetition)
        .assign_attrs({"repetition": repetition})
        .rename(f"{data.name}.repetition={repetition}")
    )


def split_by_repetition(
    data: xr.DataArray,
    /,
    *,
    n_repetitions: int,
) -> dict[int, xr.DataArray]:
    return {
        repetition: filter_by_repetition(data, repetition=repetition)
        for repetition in range(n_repetitions)
    }


def sample_neuroids(
    data: xr.DataArray,
    *,
    n_neuroids: int,
    random_state: int = 0,
) -> xr.DataArray:
    rng = np.random.default_rng(seed=random_state)
    selection = rng.permutation(data.sizes["neuroid"])[:n_neuroids]
    return (
        data.load()
        .isel({"neuroid": selection})
        .rename(f"{data.name}.n_neuroids={n_neuroids}.random_state={random_state}")
    )


def shuffle_neuroids(dataset: xr.DataArray, *, seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(dataset.sizes["neuroid"])
    dataset = dataset.reset_index("neuroid")
    for coord in ("x", "y", "z"):
        dataset[coord].data = dataset[coord].data[permutation]
    dataset = dataset.set_index({"neuroid": ["x", "y", "z"]})
    return dataset.rename(f"{dataset.name}.shuffled_neuroids.seed=0")


def sample_neuroids_isotropically(
    data: xr.DataArray,
    *,
    spacing: int,
) -> xr.DataArray:
    if spacing == 1:
        return data

    neuroids = set(
        itertools.product(*[
            range(data[coord].min().data, data[coord].max().data, spacing)
            for coord in ("x", "y", "z")
        ]),
    )
    neuroids &= set(data["neuroid"].data)

    return (
        data.load()
        .sel(neuroid=sorted(neuroids))
        .rename(f"{data.name}.isotropic_sampling.spacing={spacing}")
    )
