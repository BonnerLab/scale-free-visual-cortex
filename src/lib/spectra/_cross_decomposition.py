import functools
import itertools
from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
import xarray as xr
from bonner.caching import cache
from bonner.computation.cuda import try_devices
from bonner.computation.decomposition import PLSSVD
from bonner.computation.metrics._corrcoef import _helper
from tqdm.auto import tqdm

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


def _fit_plssvd(x: xr.DataArray, y: xr.DataArray, /, *, randomized: bool) -> PLSSVD:
    plssvd = PLSSVD(randomized=randomized)
    try_devices(plssvd.fit)(torch.from_numpy(x.data), torch.from_numpy(y.data))
    return plssvd


class CrossDecomposition:
    def __init__(
        self: Self,
        *,
        randomized: bool,
    ) -> None:
        self.randomized = randomized

        self._x_name: str
        self._y_name: str
        self._plssvd: PLSSVD
        self._cache_path: str

        self._x_neuroid: dict[str, tuple[str, np.ndarray]]
        self._y_neuroid: dict[str, tuple[str, np.ndarray]]
        self._presentation: dict[str, tuple[str, np.ndarray]]

    def fit(self: Self, x: xr.DataArray, y: xr.DataArray, /) -> None:
        self._x_name = str(x.name)
        self._y_name = str(y.name)

        if "neuroid" in x.indexes:
            self._x_neuroid = {
                str(coord): ("neuroid", x[coord].data)
                for coord in x.reset_index("neuroid").coords
                if x[coord].dims[0] == "neuroid"
            }
        else:
            self._x_neuroid = {}

        if "neuroid" in x.indexes:
            self._y_neuroid = {
                str(coord): ("neuroid", y[coord].data)
                for coord in y.reset_index("neuroid").coords
                if y[coord].dims[0] == "neuroid"
            }
        else:
            self._y_neuroid = {}

        if "presentation" in x.indexes:
            self._presentation = {
                str(coord): ("presentation", x[coord].data)
                for coord in x.reset_index("presentation").coords
                if x[coord].dims[0] == "presentation"
            }
        else:
            self._presentation = {}

        self._cache_path = f"cross_decomposition/randomized={self.randomized}/x_train={x.name}/y_train={y.name}"
        cacher = cache(f"{self._cache_path}/plssvd.pkl")

        self._plssvd = cacher(_fit_plssvd)(
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
            data=self._plssvd.singular_values.cpu().numpy(),
            dims=("component",),
        ).assign_coords(
            {"component": ("component", 1 + np.arange(self._plssvd.n_components))},
        )

    def singular_vectors(self: Self, *, direction: str) -> xr.DataArray:
        match direction:
            case "left":
                singular_vectors = self._plssvd.left_singular_vectors
                name = self._x_name
                coords = self._x_neuroid
            case "right":
                singular_vectors = self._plssvd.right_singular_vectors
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
            | {"component": ("component", 1 + np.arange(self._plssvd.n_components))},
        )

    def transform(
        self,
        z: xr.DataArray,
        /,
        *,
        direction: str,
        components: None | int | Sequence[int] = None,
    ) -> xr.DataArray:
        return xr.DataArray(
            name=f"{z.name}.transformed",
            data=self._plssvd.transform(
                torch.from_numpy(z.data).to(device=self._plssvd.device),
                direction=direction,
                components=components,
            )
            .cpu()
            .numpy(),
            dims=("presentation", "component"),
            coords=self._presentation
            | {"component": ("component", 1 + np.arange(self._plssvd.n_components))},
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
            data=self._plssvd.inverse_transform(
                torch.from_numpy(z.data).to(self._plssvd.device),
                direction=direction,
                components=components,
            )
            .cpu()
            .numpy(),
            dims=("presentation", "neuroid"),
            coords=self._presentation | neuroid_coords,
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
            f"{self._cache_path}"
            f"/x_test={x.name}"
            f"/y_test={y.name}"
            f"/{metric}.nc",
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

        x_transformed = self._plssvd.transform(
            torch.from_numpy(x.data).to(self._plssvd.device),
            direction="left",
        )
        y_transformed = self._plssvd.transform(
            torch.from_numpy(y.data).to(self._plssvd.device),
            direction="right",
        )
        return xr.DataArray(
            name=metric,
            data=func(x_transformed, y_transformed).cpu().numpy(),
            dims=("component",),
        ).assign_coords(
            {"component": ("component", 1 + np.arange(self._plssvd.n_components))},
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

        output = xr.DataArray(
            name=f"{metric} ({suffix})",
            data=np.empty((n, self._plssvd.n_components)),
            dims=(method, "component"),
            coords={
                "component": ("component", 1 + np.arange(self._plssvd.n_components)),
            },
        )

        start = 0
        for batch in tqdm(
            itertools.batched(batches, n=batch_size),
            desc=method,
            leave=False,
        ):
            match method:
                case "bootstrap":
                    x_transformed_batch = self._plssvd.transform(
                        torch.from_numpy(x.data[np.stack(batch), :]),
                        direction="left",
                    )
                case "permutation":
                    x_transformed_batch = self._plssvd.transform(
                        torch.from_numpy(np.tile(x.data, [len(batch), 1, 1])),
                        direction="left",
                    )

            y_transformed_batch = self._plssvd.transform(
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
