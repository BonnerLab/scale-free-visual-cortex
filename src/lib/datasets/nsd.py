"""The Natural Scenes Dataset (Allen 2021)."""

__all__ = (
    "N_SUBJECTS",
    "StimulusSet",
)

from collections.abc import Mapping, Sequence

import more_itertools
import numpy as np
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
)
from bonner.datasets.allen2021_natural_scenes._visualization import (
    _postprocess_mni_transform,
    load_transformation,
    reshape_dataarray_to_brain,
)
from scipy.ndimage import map_coordinates
from torch.nn.functional import interpolate
from tqdm.auto import tqdm

RESOLUTION = 1.8
MNI_SHAPE = (182, 218, 182)
ROIS: Mapping[str, Sequence[dict[str, str]]] = (
    {
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
        "frontal": (
            {"source": "corticalsulc", "label": "IFG"},
            {"source": "corticalsulc", "label": "IFRS"},
            {"source": "corticalsulc", "label": "MFG"},
            {"source": "corticalsulc", "label": "OS"},
            {"source": "corticalsulc", "label": "PrCS"},
            {"source": "corticalsulc", "label": "SFRS"},
            {"source": "corticalsulc", "label": "SRS"},
            {"source": "HCP_MMP1", "label": "1"},
            {"source": "HCP_MMP1", "label": "2"},
            {"source": "HCP_MMP1", "label": "3a"},
            {"source": "HCP_MMP1", "label": "3b"},
        ),
        "places": ({"source": "floc-places"},),
        "faces": ({"source": "floc-faces"},),
        "bodies": ({"source": "floc-bodies"},),
        "words": ({"source": "floc-words"},),
    }
    | {
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
    | {
        "TG": (
            {"source": "HCP_MMP1", "label": "TGv"},
            {"source": "HCP_MMP1", "label": "TGd"},
        ),
        "TPOJ": (
            {"source": "HCP_MMP1", "label": "TPOJ1"},
            {"source": "HCP_MMP1", "label": "TPOJ2"},
            {"source": "HCP_MMP1", "label": "TPOJ3"},
        ),
        "STSa": (
            {"source": "HCP_MMP1", "label": "STSva"},
            {"source": "HCP_MMP1", "label": "STSda"},
        ),
        "STSp": (
            {"source": "HCP_MMP1", "label": "STSvp"},
            {"source": "HCP_MMP1", "label": "STSdp"},
        ),
        "PH_PHT": (
            {"source": "HCP_MMP1", "label": "PH"},
            {"source": "HCP_MMP1", "label": "PHT"},
        ),
    }
    | {
        "eccentricity-0.5": ({"source": "prf-eccrois", "label": "ecc0pt5"},),
        "eccentricity-1": ({"source": "prf-eccrois", "label": "ecc1"},),
        "eccentricity-2": ({"source": "prf-eccrois", "label": "ecc2"},),
        "eccentricity-4": ({"source": "prf-eccrois", "label": "ecc4"},),
        "eccentricity-4+": ({"source": "prf-eccrois", "label": "ecc4+"},),
    }
)


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
    resolution: str,
    preprocessing: str,
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
            .set_index({"neuroid": ("x", "y", "z")})
        )
        rois = load_rois(subject=subject, resolution=resolution).load()
        selector = create_roi_selector(rois=rois, selectors=ROIS[roi])
        selector = (
            rois.isel({"neuroid": selector})
            .set_index({"neuroid": ("x", "y", "z")})
            .indexes["neuroid"]
        )
        # remove invalid voxels present in `selector` but removed in `betas`
        betas = betas.sel(
            neuroid=list(set(selector) & set(betas["neuroid"].data)),
        ).reset_index("neuroid")

    return betas


def load_dataset(
    *,
    subject: int,
    resolution: str = "1pt8mm",
    preprocessing: str = "fithrf",
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
    resolution: float = RESOLUTION,
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
    coordinates, good_voxels = _postprocess_mni_transform(
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
        more_itertools.chunked(range(data.sizes["presentation"]), n=batch_size),
        desc="batch",
        leave=False,
    ):
        data_batch = np.nan_to_num(
            reshape_dataarray_to_brain(
                data.isel({"presentation": batch}).copy(),
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
    return data_mni.reset_index(["neuroid", "presentation"])
