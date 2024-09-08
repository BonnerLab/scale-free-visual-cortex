from hashlib import blake2b

import numpy as np
import numpy.typing as npt
import xarray as xr
from bonner.caching import cache
from tqdm.auto import tqdm

from lib.spectra._cross_decomposition import CrossDecomposition

EMPTY_ARRAY = np.array([])


def _compute_filter_label(filter_: npt.NDArray[np.bool_] | None) -> str:
    return (
        "all"
        if filter_ is None
        else blake2b(filter_.tobytes(), digest_size=4).hexdigest()
    )


def filter_datasets(
    *,
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    filter_train: npt.NDArray[np.bool_] | None,
    filter_test: npt.NDArray[np.bool_] | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    label_train = _compute_filter_label(filter_train)
    label_test = _compute_filter_label(filter_test)
    if filter_train is not None:
        x_train = x_train.isel(presentation=filter_train).rename(
            f"{x_train.name}.indices={label_train}",
        )
        y_train = y_train.isel(presentation=filter_train).rename(
            f"{y_train.name}.indices={label_train}",
        )
    if filter_test is not None:
        x_test = x_test.isel(presentation=filter_test).rename(
            f"{x_test.name}.indices={label_test}",
        )
        y_test = y_test.isel(presentation=filter_test).rename(
            f"{y_test.name}.indices={label_test}",
        )
    return x_train, y_train, x_test, y_test


def compute_spectrum(
    *,
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    filter_train: npt.NDArray[np.bool_] | None = None,
    filter_test: npt.NDArray[np.bool_] | None = None,
    metric: str = "covariance",
    randomized: bool = False,
    normalize: bool = True,
    n_bootstraps: int = 0,
    n_permutations: int = 0,
    batch_size: int = 10,
    seed: int = 0,
    sigmas: npt.NDArray[np.number] = EMPTY_ARRAY,
) -> xr.Dataset:
    if str(x_train.name) > str(y_train.name):
        x_train, y_train = y_train, x_train
        x_test, y_test = y_test, x_test

    hash_sigmas = blake2b(sigmas, digest_size=8).hexdigest()
    cacher = cache(
        "spectra"
        f"/randomized={randomized}.metric={metric}.seed={seed}"
        f"/x_train={x_train.name}"
        f"/y_train={y_train.name}"
        f"/x_test={x_test.name}"
        f"/y_test={y_test.name}"
        f"/train_indices={_compute_filter_label(filter_train)}.test_indices={_compute_filter_label(filter_test)}"
        f"/n_bootstraps={n_bootstraps}.n_permutations={n_permutations}.sigmas={hash_sigmas}.nc",
    )
    spectra = cacher(_compute_spectrum)(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        filter_train=filter_train,
        filter_test=filter_test,
        randomized=randomized,
        metric=metric,
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
        batch_size=batch_size,
        seed=seed,
        sigmas=sigmas,
    )
    if isinstance(spectra, xr.DataArray):
        spectra = spectra.to_dataset()

    if metric == "covariance" and normalize:
        for variable in (
            "covariance",
            "covariance (bootstrapped)",
            "covariance (permuted)",
        ):
            if variable in spectra:
                spectra[variable] /= np.sqrt(
                    x_test.sizes["neuroid"] * y_test.sizes["neuroid"],
                )
    return spectra


def _compute_spectrum(
    *,
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    filter_train: npt.NDArray[np.bool_] | None,
    filter_test: npt.NDArray[np.bool_] | None,
    randomized: bool,
    metric: str,
    n_bootstraps: int,
    n_permutations: int,
    batch_size: int,
    seed: int,
    sigmas: npt.NDArray[np.number],
) -> xr.Dataset:
    x_train, y_train, x_test, y_test = filter_datasets(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        filter_train=filter_train,
        filter_test=filter_test,
    )

    cross_decomposition = CrossDecomposition(randomized=randomized)
    cross_decomposition.fit(x_train, y_train)

    spectra: list[xr.DataArray | xr.Dataset] | xr.Dataset = [
        cross_decomposition.compute_spectrum(
            x_test,
            y_test,
            metric=metric,
        ),
    ]
    if n_bootstraps != 0:
        spectra.append(
            cross_decomposition.compute_bootstrapped_spectra(
                x_test,
                y_test,
                metric=metric,
                n_bootstraps=n_bootstraps,
                batch_size=batch_size,
                seed=seed,
            ),
        )
    if n_permutations != 0:
        spectra.append(
            cross_decomposition.compute_permuted_spectra(
                x_test,
                y_test,
                metric=metric,
                n_permutations=n_permutations,
                batch_size=batch_size,
                seed=seed,
            ),
        )

    if len(sigmas) > 0:
        spectra.append(cross_decomposition.smooth_singular_vectors(sigmas=sigmas))

    spectra = xr.merge(spectra)
    if isinstance(spectra, xr.DataArray):
        spectra = spectra.to_dataset()
    return spectra


def _get_random_train_test_split(
    n: int,
    *,
    n_train: int,
    n_test: int | None,
    seed: int = 0,
) -> dict[str, npt.NDArray[np.bool_]]:
    rng = np.random.default_rng(seed=seed)
    if n_test is None:
        n_test = n - n_train
    permutation = rng.permutation(
        np.concatenate(
            [
                np.ones(n_train, dtype=np.int8),
                np.zeros(n_test, dtype=np.int8),
                -np.ones(n - n_train - n_test, dtype=np.int8),
            ],
            axis=0,
        ),
    )
    return {
        "train": permutation == 1,
        "test": permutation == 0,
    }


def compute_spectra_with_independent_train_test_splits(
    *,
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    n_train: int,
    n_test: int,
    n_splits: int,
    **kwargs,
) -> xr.Dataset:
    n_stimuli = x_train.sizes["presentation"]

    spectra: list[xr.Dataset] = []
    for split in tqdm(range(n_splits), desc="split", leave=False):
        filters = _get_random_train_test_split(
            n_stimuli,
            n_train=n_train,
            n_test=n_test,
            seed=split,
        )

        spectra.append(
            compute_spectrum(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                filter_train=filters["train"],
                filter_test=filters["test"],
                seed=split,
                **kwargs,
            ).expand_dims({"split": [split]}),
        )

    return xr.concat(spectra, dim="split")


def _get_folds(*, n_stimuli: int, n_folds: int, seed: int) -> npt.NDArray[np.uint8]:
    n_stimuli_per_fold = n_stimuli // n_folds

    rng = np.random.default_rng(seed=seed)
    return rng.permutation(
        np.concatenate(
            [
                fold * np.ones((n_stimuli_per_fold,), dtype=np.uint8)
                for fold in range(n_folds)
            ]
            + [np.zeros((n_stimuli % n_folds,), dtype=np.uint8)],
            axis=0,
        ),
    )


def compute_spectra_with_n_fold_cross_validation(
    *,
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    n_folds: int,
    seed: int = 0,
    **kwargs,
) -> xr.Dataset:
    folds = _get_folds(
        n_stimuli=x_train.sizes["presentation"],
        n_folds=n_folds,
        seed=seed,
    )
    return xr.concat(
        [
            compute_spectrum(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                filter_train=folds != fold,
                filter_test=folds == fold,
                seed=seed,
                **kwargs,
            ).expand_dims({"fold": [fold]})
            for fold in tqdm(range(n_folds), desc="fold", leave=False)
        ],
        dim="fold",
    )
