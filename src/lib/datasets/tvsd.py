from typing import Literal

import xarray as xr
from bonner.caching import cache
from bonner.datasets.papale2025_tvsd import (
    IDENTIFIER,
    load_normalized_data,
)


@cache(
    f"data/dataset={IDENTIFIER}/normalized={{normalized}}/monkey={{monkey}}.nc",
)
def load_dataset(
    *,
    monkey: Literal["F", "N"],
    normalized: bool = False,
) -> xr.DataArray:
    if normalized:
        return load_normalized_data(monkey=monkey)
    raise NotImplementedError
