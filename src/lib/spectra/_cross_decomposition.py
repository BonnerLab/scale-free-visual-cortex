import functools
import itertools
from collections.abc import Sequence
from hashlib import blake2b
from typing import Self

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from bonner.caching import cache
from bonner.computation.cuda import try_devices
from bonner.computation.decomposition import CrossCovarianceSVD
from bonner.computation.metrics._corrcoef import _helper
from tqdm.auto import tqdm

from lib.spectra._spatial_scale import compute_variance_decay_with_spatial_smoothing

uncentered_covariance = functools.partial(
    _helper,
    center=False,
    scale=False,
)

uncentered_correlation = functools.partial(
    _helper,
    center=False,
    scale=True,
)


def _fit_cross_covariance_svd(
    x: xr.DataArray,
    y: xr.DataArray,
    /,
    *,
    randomized: bool,
) -> CrossCovarianceSVD:
    svd = CrossCovarianceSVD(randomized=randomized)
    try_devices(svd.fit)(
        torch.from_numpy(x.to_numpy()),
        torch.from_numpy(y.to_numpy()),
    )
    return svd


class CrossDecomposition:
    def __init__(
        self: Self,
        *,
        randomized: bool,
    ) -> None:
        self.randomized = randomized

        self._x_name: str
        self._y_name: str
        self._svd: CrossCovarianceSVD
        self._cache_path: str

        self._x_neuroid: dict[str, tuple[str, np.ndarray]]
        self._y_neuroid: dict[str, tuple[str, np.ndarray]]
        self._presentation: dict[str, tuple[str, np.ndarray]]

    def fit(self: Self, x: xr.DataArray, y: xr.DataArray, /) -> None:
        self._x_name = str(x.name)
        self._y_name = str(y.name)

        if "neuroid" in x.indexes:
            self._x_neuroid = {
                str(coord): ("neuroid", x[coord].to_numpy())
                for coord in x.coords
                if x[coord].dims[0] == "neuroid"
            }
        else:
            self._x_neuroid = {}

        if "neuroid" in x.indexes:
            self._y_neuroid = {
                str(coord): ("neuroid", y[coord].to_numpy())
                for coord in y.coords
                if y[coord].dims[0] == "neuroid"
            }
        else:
            self._y_neuroid = {}

        if "presentation" in x.indexes:
            self._presentation = {
                str(coord): ("presentation", x[coord].to_numpy())
                for coord in x.coords
                if x[coord].dims[0] == "presentation"
            }
        else:
            self._presentation = {}

        self._cache_path = (
            "cross_decomposition"
            f"/randomized={self.randomized}"
            f"/x_train={x.name}"
            f"/y_train={y.name}"
        )

        cacher = cache(f"{self._cache_path}/cross_covariance_svd.pkl")

        self._svd = cacher(_fit_cross_covariance_svd)(
            x,
            y,
            randomized=self.randomized,
        )

    @property
    def left_singular_vectors(self: Self) -> xr.DataArray:
        return self.singular_vectors(direction="left")

    @property
    def right_singular_vectors(self: Self) -> xr.DataArray:
        return self.singular_vectors(direction="right")

    @property
    def singular_values(self: Self) -> xr.DataArray:
        return xr.DataArray(
            name="singular value",
            data=self._svd.singular_values.cpu().numpy(),
            dims=("component",),
        ).assign_coords(
            {
                "component": (
                    "component",
                    1 + np.arange(self._svd.n_components),
                ),
            },
        )

    def singular_vectors(self: Self, *, direction: str) -> xr.DataArray:
        match direction:
            case "left":
                singular_vectors = self._svd.left_singular_vectors
                name = self._x_name
                coords = self._x_neuroid
            case "right":
                singular_vectors = self._svd.right_singular_vectors
                name = self._y_name
                coords = self._y_neuroid
            case _:
                raise ValueError

        return xr.DataArray(
            name=f"{direction} singular vectors",
            data=singular_vectors.cpu().numpy(),
            dims=("neuroid", "component"),
            attrs={"name": name},
        ).assign_coords(
            coords
            | {
                "component": (
                    "component",
                    1 + np.arange(self._svd.n_components),
                ),
            },
        )

    def transform(
        self,
        z: xr.DataArray,
        /,
        *,
        direction: str,
        components: int | Sequence[int] | None = None,
    ) -> xr.DataArray:
        transformed = xr.DataArray(
            name=f"{z.name}.transformed",
            data=self._svd.transform(
                torch.from_numpy(z.to_numpy()).to(
                    device=self._svd.device,
                ),
                direction=direction,
                components=components,
            )
            .cpu()
            .numpy(),
            dims=("presentation", "component"),
        )
        if components is None:
            components = 1 + np.arange(self._svd.n_components)
        elif isinstance(components, int):
            components = 1 + np.arange(components)
        else:
            components = 1 + np.array(components)

        return transformed.assign_coords(
            coords={
                "component": ("component", components),
            },
        )

    def inverse_transform(
        self: Self,
        z: xr.DataArray,
        /,
        *,
        direction: str,
        components: int | Sequence[int] | None = None,
    ) -> xr.DataArray:
        match direction:
            case "left":
                neuroid_coords = self._x_neuroid
            case "right":
                neuroid_coords = self._y_neuroid
            case _:
                raise ValueError

        return xr.DataArray(
            name=f"{z.name}.inverse_transformed",
            data=self._svd.inverse_transform(
                torch.from_numpy(z.to_numpy()).to(self._svd.device),
                direction=direction,
                components=components,
            )
            .cpu()
            .numpy(),
            dims=("presentation", "neuroid"),
            coords=neuroid_coords,
        )

    def compute_spectrum(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        /,
        *,
        metric: str,
    ) -> xr.DataArray:
        cacher = cache(
            f"{self._cache_path}/x_test={x.name}/y_test={y.name}/{metric}.nc",
        )
        return cacher(self._compute_spectrum)(x, y, metric=metric)

    def _compute_spectrum(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        /,
        *,
        metric: str,
    ) -> xr.DataArray:
        match metric:
            case "covariance":
                func = uncentered_covariance
            case "correlation":
                func = uncentered_correlation
            case _:
                raise ValueError

        x_transformed = self._svd.transform(
            torch.from_numpy(x.to_numpy()).to(self._svd.device),
            direction="left",
        )
        y_transformed = self._svd.transform(
            torch.from_numpy(y.to_numpy()).to(self._svd.device),
            direction="right",
        )
        return xr.DataArray(
            name=metric,
            data=func(x_transformed, y_transformed).cpu().numpy(),
            dims=("component",),
        ).assign_coords(
            {
                "component": (
                    "component",
                    1 + np.arange(self._svd.n_components),
                ),
            },
        )

    def _compute_bootstrapped_or_permuted_spectra(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        /,
        *,
        metric: str,
        method: str,
        n: int,
        batch_size: int = 10,
        seed: int = 0,
    ) -> xr.DataArray:
        match metric:
            case "covariance":
                func = uncentered_covariance
            case "correlation":
                func = uncentered_correlation
            case _:
                raise ValueError

        rng = np.random.default_rng(seed=seed)

        match method:
            case "bootstrap":
                batches = np.stack(
                    [
                        rng.choice(
                            x.sizes["presentation"],
                            size=(x.sizes["presentation"],),
                            replace=True,
                        )
                        for _ in range(n)
                    ],
                )
                suffix = "bootstrapped"
            case "permutation":
                batches = np.stack(
                    [rng.permutation(x.sizes["presentation"]) for _ in range(n)],
                )
                suffix = "permuted"
            case _:
                raise ValueError

        output = xr.DataArray(
            name=f"{metric} ({suffix})",
            data=np.empty((n, self._svd.n_components)),
            dims=(method, "component"),
            coords={
                "component": (
                    "component",
                    1 + np.arange(self._svd.n_components),
                ),
            },
        )

        start = 0
        for batch in tqdm(
            itertools.batched(batches, n=batch_size, strict=False),
            desc=method,
            leave=False,
        ):
            match method:
                case "bootstrap":
                    x_transformed_batch = self._svd.transform(
                        torch.from_numpy(x.data[np.stack(batch), :]),
                        direction="left",
                    )
                case "permutation":
                    x_transformed_batch = self._svd.transform(
                        torch.from_numpy(np.tile(x.to_numpy(), [len(batch), 1, 1])),
                        direction="left",
                    )

            y_transformed_batch = self._svd.transform(
                torch.from_numpy(y.data[np.stack(batch), :]),
                direction="right",
            )

            indices = slice(start, start + len(batch))

            output.data[indices, :] = (
                func(
                    x_transformed_batch,
                    y_transformed_batch,
                )
                .cpu()
                .numpy()
            )
            start += len(batch)

        return output

    def compute_bootstrapped_spectra(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        /,
        *,
        metric: str,
        n_bootstraps: int,
        batch_size: int = 10,
        seed: int = 0,
    ) -> xr.DataArray:
        cacher = cache(
            f"{self._cache_path}"
            f"/x_test={x.name}"
            f"/y_test={y.name}"
            f"/{metric}.n_bootstraps={n_bootstraps}.seed={seed}.nc",
        )
        return cacher(self._compute_bootstrapped_or_permuted_spectra)(
            x,
            y,
            metric=metric,
            method="bootstrap",
            n=n_bootstraps,
            batch_size=batch_size,
            seed=seed,
        )

    def compute_permuted_spectra(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        /,
        *,
        metric: str,
        n_permutations: int,
        batch_size: int = 10,
        seed: int = 0,
    ) -> xr.DataArray:
        cacher = cache(
            f"{self._cache_path}"
            f"/x_test={x.name}"
            f"/y_test={y.name}"
            f"/{metric}.n_permutations={n_permutations}.seed={seed}.nc",
        )
        return cacher(self._compute_bootstrapped_or_permuted_spectra)(
            x,
            y,
            metric=metric,
            method="permutation",
            n=n_permutations,
            batch_size=batch_size,
            seed=seed,
        )

    def smooth_singular_vectors(
        self,
        *,
        sigmas: npt.NDArray[np.number],
    ) -> xr.DataArray:
        hash_sigmas = blake2b(sigmas, digest_size=8).hexdigest()
        cacher = cache(
            f"{self._cache_path}/smooth_singular_vectors.sigmas={hash_sigmas}.nc",
        )
        return cacher(self._smooth_singular_vectors)(sigmas=sigmas)

    def _smooth_singular_vectors(
        self,
        *,
        sigmas: npt.NDArray[np.number],
    ) -> xr.DataArray:
        return xr.concat(
            [
                compute_variance_decay_with_spatial_smoothing(
                    self.singular_vectors(direction=direction),
                    sigmas=sigmas,
                ).expand_dims({"direction": [direction]})
                for direction in ("left", "right")
            ],
            dim="direction",
        )
