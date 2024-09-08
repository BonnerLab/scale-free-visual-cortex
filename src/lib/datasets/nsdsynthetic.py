"""The nsd-synthetic dataset (Gifford et al., 2025)."""

__all__ = (
    "N_SUBJECTS",
    "StimulusSet",
)
from typing import Literal

import xarray as xr
from bonner.caching import cache
from bonner.datasets.gifford2025_nsd_synthetic import (
    IDENTIFIER,
    N_SUBJECTS,
    StimulusSet,
    create_roi_selector,
    load_betas,
)

from .nsd import ROIS, load_rois


@cache(
    "data"
    f"/dataset={IDENTIFIER}"
    "/betas"
    "/resolution={resolution}"
    "/preprocessing={preprocessing}"
    "/z_score={z_score}"
    "/roi={roi}"
    "/subject={subject}.nc",
)
def _open_betas_by_roi(
    *,
    subject: int,
    resolution: Literal["1mm", "1pt8mm"],
    preprocessing: Literal["fithrf", "fithrf_GLMdenoise_RR"],
    z_score: bool,
    roi: str,
) -> xr.DataArray:
    if roi == "whole-brain":
        betas = load_betas(
            subject=subject,
            resolution=resolution,
            preprocessing=preprocessing,
            z_score=z_score,
        )
    else:
        betas = (
            _open_betas_by_roi(
                subject=subject,
                resolution=resolution,
                preprocessing=preprocessing,
                z_score=z_score,
                roi="whole-brain",
            )
            .load()
            .set_xindex(["x", "y", "z"])
        )
        rois = load_rois(subject=subject, resolution=resolution).load()
        selector = create_roi_selector(rois=rois, selectors=ROIS[roi])
        selector = (
            rois.isel({"neuroid": selector})
            .set_xindex(["x", "y", "z"])
            .indexes["neuroid"]
        )
        # remove invalid voxels present in `selector` but removed in `betas`
        betas = betas.sel(
            neuroid=list(set(selector) & set(betas["neuroid"].data)),
        )

    return betas


def load_dataset(
    *,
    subject: int,
    resolution: Literal["1mm", "1pt8mm"] = "1pt8mm",
    preprocessing: Literal["fithrf", "fithrf_GLMdenoise_RR"] = "fithrf",
    z_score: bool = True,
    roi: str = "general",
) -> xr.DataArray:
    betas = _open_betas_by_roi(
        resolution=resolution,
        preprocessing=preprocessing,
        z_score=z_score,
        roi=roi,
        subject=subject,
    ).assign_attrs({"roi": roi})

    identifier = ".".join([f"{key}={value}" for key, value in betas.attrs.items()])
    return betas.rename(f"{IDENTIFIER}.{identifier}").set_xindex([
        "stimulus",
        "repetition",
    ])
