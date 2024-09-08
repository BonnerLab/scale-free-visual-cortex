"""The Natural Scenes Dataset (Allen 2021)."""

__all__ = (
    "N_SUBJECTS",
    "StimulusSet",
)

import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from bonner.caching import cache
from bonner.datasets.allen2021_natural_scenes import (
    IDENTIFIER,
    N_SUBJECTS,
    StimulusSet,
    create_roi_selector,
    load_betas,
    load_brain_mask,
    load_rois,
    load_transformation,
    plot_brain_map,
    postprocess_mni_transform,
    reshape_dataarray_to_brain,
)
from bonner.datasets.allen2021_natural_scenes._stimuli import (
    download_annotations,
    get_coco_to_nsd_mapping,
)
from matplotlib.axes import Axes
from scipy.ndimage import map_coordinates
from torch.nn.functional import interpolate
from tqdm.auto import tqdm

RESOLUTION_IN_MM = 1.8
MNI_SHAPE = (182, 218, 182)
ROIS: Mapping[str, Sequence[dict[str, str]]] = {
    "general": ({"source": "nsdgeneral", "label": "nsdgeneral"},),
    "V1-4": ({"source": "prf-visualrois"},),
    "V1": (
        {"source": "prf-visualrois", "label": "V1v"},
        {"source": "prf-visualrois", "label": "V1d"},
    ),
    "V2": (
        {"source": "prf-visualrois", "label": "V2v"},
        {"source": "prf-visualrois", "label": "V2d"},
    ),
    "V3": (
        {"source": "prf-visualrois", "label": "V3v"},
        {"source": "prf-visualrois", "label": "V3d"},
    ),
    "V4": ({"source": "prf-visualrois", "label": "hV4"},),
    "frontal": tuple(
        {"source": "HCP_MMP1", "label": label}
        for label in (
            "44",
            "45",
            "46",
            "47l",
            "55b",
            "6a",
            "6d",
            "6r",
            "6v",
            "8Ad",
            "8Av",
            "8C",
            "9-46d",
            "FEF",
            "9p",
            "FOP5",
            "IFJa",
            "IFJp",
            "IFSa",
            "IFSp",
            "PEF",
            "a47r",
            "a9-46v",
            "p47r",
            "p9-46v",
            "i6-8",
        )
    ),
    "places": ({"source": "floc-places"},),
    "faces": ({"source": "floc-faces"},),
    "bodies": ({"source": "floc-bodies"},),
    "words": ({"source": "floc-words"},),
    "OPA": ({"source": "floc-places", "label": "OPA"},),
    "PPA": ({"source": "floc-places", "label": "PPA"},),
} | {
    f"{stream} visual stream": ({"source": "streams", "label": stream},)
    for stream in (
        "early",
        "lateral",
        "parietal",
        "ventral",
        "midlateral",
        "midparietal",
        "midventral",
    )
}


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
    return (
        betas.rename(f"{IDENTIFIER}.{identifier}")
        .set_xindex(["stimulus", "repetition"])
        .set_xindex(["x", "y", "z"])
    )


def resample_1mm_mni(
    data: xr.DataArray,
    /,
    *,
    resolution: float = RESOLUTION_IN_MM,
    **kwargs,
) -> xr.DataArray:
    mni = reshape_dataarray_to_brain(
        data=data.copy(),
        brain_shape=MNI_SHAPE,
    )
    mni = torch.from_numpy(mni)

    match mni.ndim:
        case 3:
            mni = mni.unsqueeze(0).unsqueeze(0)
        case 4:
            mni = mni.unsqueeze(1)
    mni_interpolated = interpolate(mni, scale_factor=1 / resolution, **kwargs)
    valid_voxels = torch.stack(
        [
            torch.nonzero(~torch.isnan(mni_interpolated_))
            for mni_interpolated_ in mni_interpolated.squeeze(dim=1)
        ],
        dim=0,
    )
    if torch.equal(
        (valid_voxels / valid_voxels[0]) - 1,
        torch.zeros_like(valid_voxels),
    ):
        valid_voxels = valid_voxels[0].cpu().numpy()
    else:
        raise ValueError
    mni_interpolated = mni_interpolated[
        ...,
        valid_voxels[:, 0],
        valid_voxels[:, 1],
        valid_voxels[:, 2],
    ].squeeze()

    coords = {
        "x": ("neuroid", valid_voxels[:, 0]),
        "y": ("neuroid", valid_voxels[:, 1]),
        "z": ("neuroid", valid_voxels[:, 2]),
    }
    if "presentation" in data.dims:
        coords |= {
            str(coord): ("presentation", data[coord].data)
            for coord in data.coords
            if data[coord].dims[0] == "presentation"
        }

    return xr.DataArray(
        data=mni_interpolated.cpu().numpy(),
        dims=data.dims,
        coords=coords,
    )


@cache(
    identifier="mni/order={order}/{label}.nc",
    helper=lambda kwargs: {
        "order": kwargs["order"],
        "label": kwargs["data"].name,
    },
)
def convert_array_to_mni(
    data: xr.DataArray,
    *,
    subject: int,
    order: int = 0,
    batch_size: int = 100,
) -> xr.DataArray:
    brain_shape = load_brain_mask(subject=subject, resolution="1pt8mm").shape

    match data.ndim:
        case 1:
            test_volume = data.copy()
        case 2:
            test_volume = data.isel(presentation=0).copy()
        case _:
            raise ValueError

    test_volume = reshape_dataarray_to_brain(test_volume, brain_shape=brain_shape)

    coordinates = load_transformation(
        subject=subject,
        source_space="func1pt8",
        target_space="MNI",
        suffix=".nii.gz",
    )
    coordinates, good_voxels = postprocess_mni_transform(
        coordinates,
        volume=test_volume,
    )

    test_volume = xr.DataArray(
        name=f"{data.name}.mni.order={order}",
        data=map_coordinates(
            np.nan_to_num(test_volume.astype(np.float64), nan=0),
            coordinates[..., good_voxels],
            order=order,
            mode="nearest",
            output=np.float32,
        ),
        dims=("neuroid",),
        coords=good_voxels[good_voxels].coords,
    )

    if data.ndim == 1:
        return test_volume

    match order:
        case 0 | 1:
            threshold = 0
        case 3:
            threshold = 1e-3
        case _:
            raise ValueError

    filter_ = np.absolute(test_volume) > threshold
    coordinates = coordinates[..., good_voxels][..., filter_]
    test_volume = test_volume[filter_]

    data_mni = xr.DataArray(
        name=f"{data.name}.mni.order={order}",
        data=np.empty((data.sizes["presentation"], len(test_volume)), dtype=np.float32),
        dims=("presentation", "neuroid"),
        coords=dict(test_volume["neuroid"].coords) | dict(data["presentation"].coords),
    )

    for batch in tqdm(
        itertools.batched(
            range(data.sizes["presentation"]),
            n=batch_size,
            strict=False,
        ),
        desc="batch",
        leave=False,
    ):
        data_batch = np.nan_to_num(
            reshape_dataarray_to_brain(
                data.isel({"presentation": list(batch)}).copy(),
                brain_shape=brain_shape,
            ).astype(np.float64),
            nan=0,
        )

        for idx, presentation in enumerate(batch):
            data_mni.data[presentation, :] = map_coordinates(
                data_batch[idx, ...],
                coordinates,
                order=order,
                mode="nearest",
                output=np.float32,
            )
    return data_mni


def plot_multiple_brain_maps(
    *,
    ax: Axes,
    maps: Sequence[xr.DataArray],
    subject: int,
    bottom: float = 0,
    separation: float = 0.15,
    height: float = 0.55,
    width: float = 0.55,
    view: str | tuple[float, float] = (0, 200),
    threshold: float = 1e-8,
    reverse: bool = False,
) -> list[Axes]:
    inset_axes = []
    for i_map, brain_map in enumerate(maps):
        index = (len(maps) - i_map - 1) if reverse else i_map
        ax_ = ax.inset_axes(
            (1 - width - separation * index, bottom, width, height),
            projection="3d",
            facecolor=None,
        )
        ax_.set_rasterized(True)

        plot_brain_map(
            brain_map,
            ax=ax_,
            subject=subject,
            cmap="cold_hot",
            threshold=threshold,
            view=view,
        )
        ax_.set_facecolor((1, 1, 1, 0))
        inset_axes.append(ax_)
    return inset_axes


def get_license_information() -> pd.DataFrame:
    directory = download_annotations()
    mapping = get_coco_to_nsd_mapping()

    metadata = []
    for subset in ("train", "val"):
        with (directory / f"instances_{subset}2017.json").open() as f:
            annotations = json.load(f)

        licenses = (
            pd.DataFrame(annotations["licenses"])
            .rename(
                columns={"id": "license", "url": "license_url", "name": "license_name"},
            )
            .set_index("license")
        )

        images = pd.DataFrame(annotations["images"])
        images = (
            images.assign(
                nsd_id=[
                    mapping.get(id_, pd.NA) for id_ in images["id"]
                ],
            )
            .dropna()
            .assign(nsd_id=lambda x: x["nsd_id"].astype(int))
        )
        metadata.append(images.join(licenses, on="license"))

    metadata = (
        pd.concat(metadata).reset_index(drop=True).set_index("nsd_id").sort_index()
    )
    metadata = metadata.drop(
        columns=[
            "file_name",
            "height",
            "width",
            "date_captured",
            "license",
        ],
    )
    metadata["flickr_url_dynamic"] = (
        "https://flickr.com/photo.gne?id="
        + metadata["flickr_url"]
        .str.split("/", expand=True)[4]
        .str.split("_", expand=True)[0]
    )
    return metadata
