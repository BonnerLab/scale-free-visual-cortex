from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from bonner.plotting import apply_offset
from matplotlib.axes import Axes


def offset_spectra(
    spectra: pd.DataFrame,
    /,
    *,
    keys: Sequence[str],
    offset_magnitude: float = 1.025,
) -> pd.DataFrame:
    return apply_offset(
        spectra,
        keys=keys,
        offset_key="rank",
        offset_magnitude=offset_magnitude,
        offset_type="multiplicative",
    ).reset_index(drop=True)


def plot_spectra(
    spectra: xr.Dataset,
    *,
    ax: Axes,
    hue: str,
    hue_order: Sequence[int | str] | None = None,
    hue_reference: int | str | None = None,
    hue_labels: Sequence[str] | None = None,
    palette: str = "crest_r",
    metric: str = "covariance",
    errorbar: tuple[str, float] = ("fold-sd", 1),
    null_quantile: float = 0.99,
    hide_insignificant: bool = False,
) -> None:
    hues = np.unique(spectra[hue].to_numpy()) if hue_order is None else hue_order

    if isinstance(palette, str):
        color_palette = sns.color_palette(palette, n_colors=len(hues))
    else:
        color_palette = palette

    mean = spectra[metric].mean("fold")

    match errorbar:
        case ("fold-sd", n):
            fold_sd = spectra[metric].std("fold", ddof=1)
            error = n * fold_sd
        case ("fold-se", n):
            fold_se = spectra[metric].std("fold", ddof=1) / np.sqrt(
                spectra.sizes["fold"] - 1,
            )
            error = n * fold_se
        case ("bootstrap", n):
            bootstrap_sd = (
                spectra[f"{metric} (bootstrapped)"]
                .mean("fold")
                .std("bootstrap", ddof=1)
            )
            error = n * bootstrap_sd.rename(metric)
        case _:
            raise NotImplementedError

    mean_offset = offset_spectra(mean.to_dataframe().reset_index(), keys=[hue])
    error_offset = offset_spectra(error.to_dataframe().reset_index(), keys=[hue])

    if f"{metric} (permuted)" in spectra:
        nulls = (
            spectra[f"{metric} (permuted)"]
            .mean("fold")
            .quantile(null_quantile, dim="permutation")
        )
        nulls_offset = offset_spectra(nulls.to_dataframe().reset_index(), keys=[hue])

    lines = []
    for i_hue, hue_ in enumerate(hues):
        mean_offset_ = mean_offset.loc[mean_offset[hue] == hue_]
        error_offset_ = error_offset.loc[error_offset[hue] == hue_]

        rank = mean_offset_["rank"].to_numpy()
        mean_offset_ = mean_offset_[metric].to_numpy()
        error_offset_ = error_offset_[metric].to_numpy()

        yerr = np.stack(
            (
                error_offset_,
                error_offset_,
            ),
            axis=0,
        )
        kwargs_significant = {
            "ls": "None",
            "c": color_palette[i_hue],
            "marker": "s" if hue_ == hue_reference else "o",
            "zorder": 2 if hue_ == hue_reference else 1.99,
            "mew": 0,
            "alpha": 1 if (hue_ == hue_reference) or (hue_reference is None) else 0.75,
            "label": hue_labels[i_hue] if hue_labels is not None else hue_,
        }

        if hide_insignificant:
            nulls_offset_ = nulls_offset.loc[nulls_offset[hue] == hue_]
            nulls_offset_ = nulls_offset_[f"{metric} (permuted)"].to_numpy()

            significant = mean_offset_ > nulls_offset_
            kwargs_insignificant = kwargs_significant | {
                "mew": 1,
                "alpha": 0.25,
                "mfc": "None",
                "label": "",
            }
            line = ax.errorbar(
                rank[significant],
                mean_offset_[significant],
                yerr=yerr[:, significant],
                **kwargs_significant,
            )
            lines.append(line)

            ax.errorbar(
                rank[~significant],
                mean_offset_[~significant],
                yerr=yerr[:, ~significant],
                **kwargs_insignificant,
            )
        else:
            line = ax.errorbar(
                rank,
                mean_offset_,
                yerr=yerr,
                **kwargs_significant,
            )
