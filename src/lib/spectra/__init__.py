__all__ = (
    "CrossDecomposition",
    "compute_cross_individual_spectra",
    "compute_cross_roi_spectra",
    "compute_pairwise_distances",
    "compute_spectra_with_independent_train_test_splits",
    "compute_spectra_with_n_fold_cross_validation",
    "compute_spectrum",
    "compute_within_individual_spectra",
    "offset_spectra",
    "plot_spectra",
    "plot_spectrum",
)
import xarray as xr
from tqdm.auto import tqdm

from lib.spectra._computation import (
    compute_spectra_with_independent_train_test_splits,
    compute_spectra_with_n_fold_cross_validation,
    compute_spectrum,
)
from lib.spectra._cross_decomposition import CrossDecomposition
from lib.spectra._plotting import offset_spectra, plot_spectra, plot_spectrum
from lib.spectra._spatial_scale import compute_pairwise_distances
from lib.utilities._binning import bin_data, extract_geometrically_spaced_bins


def compute_within_individual_spectra(
    datasets: dict[int, dict[int, xr.DataArray]],
    *,
    n_folds: int = 8,
    density: float | None = 3,
    stop: int = 10_000,
    n_permutations: int = 0,
    n_bootstraps: int = 0,
    **kwargs,
) -> xr.Dataset:
    bin_edges, bin_centers = extract_geometrically_spaced_bins(
        start=1,
        stop=stop,
        density=density,
    )

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
        spectra_ = bin_data(
            spectra_,
            bin_edges={"component": bin_edges},
            bin_centers={"component": bin_centers},
            dim="rank",
        ).expand_dims(individual=[individual])

        spectra_within.append(spectra_)

    return xr.concat(spectra_within, dim="individual")


def compute_cross_individual_spectra(
    datasets: dict[int, dict[int, xr.DataArray]],
    *,
    reference_individual: int,
    n_folds: int = 8,
    density: float | None = 3,
    stop: int = 10_000,
    n_permutations: int = 0,
    n_bootstraps: int = 0,
    **kwargs,
) -> xr.Dataset:
    bin_edges, bin_centers = extract_geometrically_spaced_bins(
        start=1,
        stop=stop,
        density=density,
    )

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
            bin_data(
                spectra_,
                bin_edges={"component": bin_edges},
                bin_centers={"component": bin_centers},
                dim="rank",
            )
            .mean("comparison")
            .expand_dims(individual=[individual])
        )
        spectra_cross.append(spectra_)

    return xr.concat(spectra_cross, dim="individual")


def compute_cross_roi_spectra(
    datasets: dict[str, dict[int, xr.DataArray]],
    *,
    reference_roi: str,
    density: float | None = 3,
    stop: int = 10_000,
    n_folds: int = 8,
    n_permutations: int = 0,
    n_bootstraps: int = 0,
    wide_bins: bool = False,
    **kwargs,
) -> xr.Dataset:
    if wide_bins:
        bin_edges, bin_centers = extract_geometrically_spaced_bins(
            start=1,
            stop=1_000,
            n=3,
        )
    else:
        bin_edges, bin_centers = extract_geometrically_spaced_bins(
            start=1,
            stop=stop,
            density=density,
        )

    rois = datasets.keys()
    spectra_cross = []
    for roi in rois:
        spectra_ = xr.concat(
            [
                compute_spectra_with_n_fold_cross_validation(
                    x_train=datasets[reference_roi][0],
                    y_train=datasets[roi][1],
                    x_test=datasets[reference_roi][0],
                    y_test=datasets[roi][1],
                    n_folds=n_folds,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    **kwargs,
                ).expand_dims({"comparison": [0]}),
                compute_spectra_with_n_fold_cross_validation(
                    x_train=datasets[reference_roi][1],
                    y_train=datasets[roi][0],
                    x_test=datasets[reference_roi][1],
                    y_test=datasets[roi][0],
                    n_folds=n_folds,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    **kwargs,
                ).expand_dims({"comparison": [1]}),
            ],
            dim="comparison",
        )

        spectra_ = (
            bin_data(
                spectra_,
                bin_edges={"component": bin_edges},
                bin_centers={"component": bin_centers},
                dim="rank",
            )
            .mean("comparison")
            .expand_dims(roi=[roi])
        )

        spectra_cross.append(spectra_)
    return xr.concat(spectra_cross, dim="roi")
