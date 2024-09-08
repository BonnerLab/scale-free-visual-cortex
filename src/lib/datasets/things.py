__all__ = ("N_SUBJECTS", "StimulusSet", "load_embeddings")

from collections.abc import Mapping, Sequence

import xarray as xr
from bonner.caching import cache
from bonner.datasets.allen2021_natural_scenes import create_roi_selector
from bonner.datasets.hebart2019_things import StimulusSet
from bonner.datasets.hebart2023_things_data.behavior import load_embeddings
from bonner.datasets.hebart2023_things_data.fmri import (
    IDENTIFIER,
    N_SUBJECTS,
    ROIS,
    load_betas,
    load_brain_mask,
    load_rois,
)

ROI_MAPPINGS: Mapping[str, Sequence[dict[str, str]]] = {}
for roi, labels in ROIS.items():
    for label in labels:
        ROI_MAPPINGS |= {label: ({"localizer": roi, "label": label},)}

ROI_MAPPINGS |= {
    "V1-4": (
        {"localizer": "pRF", "label": "V1"},
        {"localizer": "pRF", "label": "V2"},
        {"localizer": "pRF", "label": "V3"},
        {"localizer": "pRF", "label": "hV4"},
    ),
    "visual": (
        {"localizer": "pRF", "label": "V1"},
        {"localizer": "pRF", "label": "V2"},
        {"localizer": "pRF", "label": "V3"},
        {"localizer": "pRF", "label": "hV4"},
        {"localizer": "object", "label": "LOC"},
    ),
}


@cache(
    f"data/dataset={IDENTIFIER}/betas/z_score={{z_score}}/roi={{roi}}/subject={{subject}}.nc",
)
def _open_betas_by_roi(*, subject: int, roi: str, z_score: bool) -> xr.DataArray:
    if roi == "whole-brain":
        neuroid_filter = load_brain_mask(subject=subject).data
        betas = load_betas(
            subject=subject,
            z_score=z_score,
            neuroid_filter=neuroid_filter,
        )
    else:
        betas = (
            _open_betas_by_roi(
                subject=subject,
                roi="whole-brain",
                z_score=z_score,
            )
            .load()
            .set_xindex(["x", "y", "z"])
        )
        rois = load_rois(subject=subject).load()
        selector = create_roi_selector(rois=rois, selectors=ROI_MAPPINGS[roi])
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
    roi: str,
    z_score: bool = True,
) -> xr.DataArray:
    betas = _open_betas_by_roi(subject=subject, roi=roi, z_score=z_score).assign_attrs({
        "roi": roi,
    })

    identifier = ".".join([f"{key}={value}" for key, value in betas.attrs.items()])
    return betas.rename(f"{IDENTIFIER}.{identifier}").set_xindex(["stimulus"])
