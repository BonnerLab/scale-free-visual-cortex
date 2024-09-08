import xarray as xr
from tqdm.auto import tqdm

from lib.spectra import (
    assign_data_to_geometrically_spaced_bins,
    compute_spectra_with_n_fold_cross_validation,
)


def compute_within_individual_spectra(
    datasets: dict[int, dict[int, xr.DataArray]],
    *,
    n_folds: int = 8,
    density: float | None = 3,
    stop: int = 10_000,
    n_permutations: int = 5_000,
    n_bootstraps: int = 5_000,
    **kwargs,
) -> xr.Dataset:
    spectra_within = []

    for individual, dataset in tqdm(datasets.items(), desc="individual", leave=False):
        spectra_ = compute_spectra_with_n_fold_cross_validation(
            x_train=dataset[0],
            y_train=dataset[1],
            x_test=dataset[0],
            y_test=dataset[1],
            n_folds=n_folds,
            n_permutations=n_permutations,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
        spectra_ = (
            spectra_.assign_coords({
                "rank": (
                    "component",
                    assign_data_to_geometrically_spaced_bins(
                        spectra_["component"].data,
                        density=density,
                        start=1,
                        stop=stop,
                    ),
                ),
            })
            .groupby("rank")
            .mean()
            .expand_dims(individual=[f"{1 + individual}"])
        )
        spectra_within.append(spectra_)

    return xr.concat(spectra_within, dim="individual")


def compute_cross_individual_spectra(
    datasets: dict[int, dict[int, xr.DataArray]],
    *,
    reference_individual: int,
    n_folds: int = 8,
    density: float | None = 3,
    stop: int = 10_000,
    n_permutations: int = 5_000,
    n_bootstraps: int = 5_000,
    **kwargs,
) -> xr.Dataset:
    spectra_cross = []
    for individual, dataset in tqdm(datasets.items(), desc="individual", leave=False):
        spectra_ = xr.concat(
            [
                compute_spectra_with_n_fold_cross_validation(
                    x_train=datasets[reference_individual][0],
                    y_train=dataset[1],
                    x_test=datasets[reference_individual][0],
                    y_test=dataset[1],
                    n_folds=n_folds,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    **kwargs,
                ).expand_dims({"comparison": [0]}),
                compute_spectra_with_n_fold_cross_validation(
                    x_train=datasets[reference_individual][1],
                    y_train=dataset[0],
                    x_test=datasets[reference_individual][1],
                    y_test=dataset[0],
                    n_folds=n_folds,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    **kwargs,
                ).expand_dims({"comparison": [1]}),
            ],
            dim="comparison",
        )
        spectra_ = (
            spectra_.assign_coords({
                "rank": (
                    "component",
                    assign_data_to_geometrically_spaced_bins(
                        spectra_["component"].data,
                        density=density,
                        start=1,
                        stop=stop,
                    ),
                ),
            })
            .groupby("rank")
            .mean()
            .mean("comparison")
            .expand_dims(individual=[f"{1 + individual}"])
        )
        spectra_cross.append(spectra_)

    return xr.concat(spectra_cross, dim="individual")
