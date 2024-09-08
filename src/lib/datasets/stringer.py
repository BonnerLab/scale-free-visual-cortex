"""The Stringer dataset (Stringer 2019)."""

__all__ = (
    "IDENTIFIER",
    "SESSIONS",
    "load_dataset",
    "plot_neurons",
)

import xarray as xr
from bonner.caching import cache
from bonner.datasets.stringer2019_mouse import (
    IDENTIFIER,
    SESSIONS,
    create_data_assembly,
    preprocess_assembly,
)
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


@cache(
    "data"
    f"/dataset={IDENTIFIER}"
    "/denoise={denoise}"
    "/z_score={z_score}"
    "/session={session}.nc",
)
def _open_dataset(session: int, *, z_score: bool, denoise: bool) -> xr.DataArray:
    assembly = create_data_assembly(
        mouse=SESSIONS[session]["mouse"],
        date=SESSIONS[session]["date"],
    )
    assembly = preprocess_assembly(assembly, denoise=denoise)

    if z_score:
        assembly_ = (
            assembly - assembly.mean("presentation", keep_attrs=True)
        ) / assembly.std("presentation", keep_attrs=True)
        assembly = assembly_.assign_attrs(assembly.attrs)

    assembly = assembly.dropna(dim="neuroid", how="any")
    return assembly.assign_attrs({"z_score": str(z_score), "denoise": str(denoise)})


def load_dataset(
    session: int,
    *,
    z_score: bool = True,
    denoise: bool = False,
) -> xr.DataArray:
    assembly = _open_dataset(session=session, z_score=z_score, denoise=denoise)
    identifier = ".".join([f"{key}={value}" for key, value in assembly.attrs.items()])
    return (
        assembly.rename(f"{IDENTIFIER}.{identifier}")
        .set_xindex(["stimulus", "repetition"])
        .set_xindex(["x", "y", "z"])
    )


def plot_neurons(
    data: xr.DataArray,
    *,
    ax: Axes3D | Axes,
    cmap: str = "cold_hot",
    vscale: float,
    alpha: float = 0.5,
    s: float = 1,
    all_slices: bool = True,
    elev: float | None = 1,
    azim: float | None = None,
    roll: float | None = None,
    rasterized: bool = True,
    **kwargs,
) -> None:
    kwargs_ = {
        "cmap": cmap,
        "vmin": -vscale,
        "vmax": vscale,
        "alpha": alpha,
        "s": s,
        "edgecolors": None,
    } | kwargs

    if all_slices:
        ax.scatter(
            *[data[coord].to_numpy() for coord in ("x", "y", "z")],
            c=data.to_numpy(),
            **kwargs_,
        )
        ax.margins(0)
        ax.view_init(elev=elev, azim=azim, roll=roll)
    else:
        ax.scatter(
            *[data[coord].to_numpy() for coord in ("x", "y")],
            c=data.to_numpy(),
            **kwargs_,
        )
        ax.set_aspect("equal")

    if rasterized:
        ax.set_rasterization_zorder(2.5)
