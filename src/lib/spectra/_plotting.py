from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from bonner.plotting import apply_offset
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer


def offset_spectra(
    spectra: pd.DataFrame,
    /,
    *,
    keys: Sequence[str],
    offset_key: str = "rank",
    offset_magnitude: float = 1.025,
) -> pd.DataFrame:
    return apply_offset(
        spectra,
        keys=keys,
        offset_key=offset_key,
        offset_magnitude=offset_magnitude,
        offset_type="multiplicative",
    ).reset_index(drop=True)


def plot_spectrum(
    ax: Axes,
    *,
    spectrum: xr.Dataset,
    metric: str = "covariance",
    errorbar: tuple[str, float] = ("fold-sd", 1),
    hide_insignificant: bool = False,
    hide_insignificant_errorbar: bool = False,
    null_quantile: float = 0.999,
    kwargs_significant: dict = {},
    kwargs_insignificant: dict = {},
) -> ErrorbarContainer:
    means = spectrum[metric].mean(dim="fold")
    if f"{metric} (permuted)" in spectrum:
        spectrum = spectrum.assign_coords({
            "significant": (
                "rank",
                (
                    means
                    > (
                        spectrum[f"{metric} (permuted)"]
                        .mean("fold")
                        .quantile(null_quantile, dim="permutation")
                    )
                ).to_numpy(),
            ),
        })

    match errorbar:
        case ("fold-sd", n):
            fold_sd = spectrum[metric].std("fold", ddof=1)
            errors = n * fold_sd
        case ("fold-se", n):
            fold_se = spectrum[metric].std("fold", ddof=1) / np.sqrt(
                spectrum.sizes["fold"] - 1,
            )
            errors = n * fold_se
        case ("bootstrap", n):
            bootstrap_sd = (
                spectrum[f"{metric} (bootstrapped)"]
                .mean("fold")
                .std("bootstrap", ddof=1)
            )
            errors = n * bootstrap_sd.rename(metric)
        case _:
            raise NotImplementedError

    if hide_insignificant:
        significant = spectrum["significant"].to_numpy()
        errorbar_ = ax.errorbar(
            means["rank"].isel(rank=significant),
            means.isel(rank=significant),
            errors.isel(rank=significant),
            **kwargs_significant,
        )
        if hide_insignificant_errorbar:
            ax.plot(
                means["rank"].isel(rank=~significant),
                means.isel(rank=~significant),
                **kwargs_insignificant,
            )
        else:
            errorbar_ = ax.errorbar(
                means["rank"].isel(rank=~significant),
                means.isel(rank=~significant),
                errors.isel(rank=~significant),
                **kwargs_insignificant,
            )
    else:
        errorbar_ = ax.errorbar(means["rank"], means, errors, **kwargs_significant)

    return errorbar_


def plot_spectra(
    spectra: xr.Dataset,
    *,
    ax: Axes,
    hue: str,
    hue_order: Sequence[int | str] | None = None,
    hue_reference: int | str | None = None,
    hue_labels: Sequence[str] | None = None,
    marker: str | None = "s",
    palette: str = "crest_r",
    metric: str = "covariance",
    errorbar: tuple[str, float] = ("fold-sd", 1),
    null_quantile: float = 0.999,
    hide_insignificant: bool = False,
    hide_insignificant_errorbar: bool = False,
    kwargs_significant: dict = {},
    kwargs_insignificant: dict = {},
    offset_magnitude: float = 1.025,
) -> None:
    hues = np.unique(spectra[hue].to_numpy()) if hue_order is None else hue_order

    if isinstance(palette, str):
        color_palette = sns.color_palette(palette, n_colors=len(hues))
    else:
        color_palette = palette

    if hide_insignificant:
        significant = spectra[metric].mean("fold") > (
            spectra[f"{metric} (permuted)"]
            .mean("fold")
            .quantile(null_quantile, dim="permutation")
        )

    spectra_offset = offset_spectra(
        spectra[metric].to_dataframe().reset_index(),
        keys=[hue],
        offset_magnitude=offset_magnitude,
    )

    for i_hue, hue_ in enumerate(hues):
        spectrum = (
            spectra_offset.loc[spectra_offset[hue] == hue_]
            .to_xarray()
            .drop_vars("index")
            .set_index({"index": ["fold", "rank"]})
            .unstack("index")
        )
        if hide_insignificant:
            spectrum = spectrum.assign_coords({
                "significant": (
                    "rank",
                    significant.sel({hue: hue_}).to_numpy().flatten(),
                ),
            })

        kwargs_significant_ = {
            "ls": "None",
            "c": color_palette[i_hue],
            "marker": ("s" if hue_ == hue_reference else "o")
            if marker is None
            else marker,
            "zorder": 2 if hue_ == hue_reference else 1.99,
            "mew": 0,
            "alpha": 1 if (hue_ == hue_reference) or (hue_reference is None) else 0.75,
            "label": hue_labels[i_hue] if hue_labels is not None else hue_,
        } | kwargs_significant

        kwargs_insignificant_ = (
            kwargs_significant_
            | {
                "mew": 1,
                "alpha": 0.5,
                "mfc": "None",
                "label": "",
            }
            | kwargs_insignificant
        )

        data_line, caplines, barlinecols = plot_spectrum(
            ax,
            spectrum=spectrum,
            metric=metric,
            errorbar=errorbar,
            hide_insignificant=hide_insignificant,
            hide_insignificant_errorbar=hide_insignificant_errorbar,
            kwargs_significant=kwargs_significant_,
            kwargs_insignificant=kwargs_insignificant_,
        )
        zorder = 2 + i_hue * 0.01
        data_line.set_zorder(zorder)
        [artist.set_zorder(zorder) for artist in caplines]
        [artist.set_zorder(zorder) for artist in barlinecols]
