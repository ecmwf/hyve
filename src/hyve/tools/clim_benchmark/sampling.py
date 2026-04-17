"""Translate DOY pools into xarray selections grouped by (doy, issue_hour)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from hyve.tools.clim_benchmark.dates import (
    build_doy_pools,
    split_pool_by_issue,
)


Slot = tuple[int, int]  # (doy, issue_hour)


def build_slots(
    da: xr.DataArray,
    window_days: int,
    stride: str,
    issue_frequency_hours: int,
) -> dict[Slot, np.ndarray]:
    """Return a dict keyed by ``(doy, issue_hour)`` with the positional
    time indices contributing to each (doy, issue) climatological pool.

    Slots with zero samples are dropped; the caller is responsible for
    filling missing slots with NaN when assembling the final array.
    """
    time_index = pd.DatetimeIndex(da["time"].values)
    pools = build_doy_pools(time_index, window_days, stride)

    slots: dict[Slot, np.ndarray] = {}
    for doy, idx in pools.items():
        per_issue = split_pool_by_issue(idx, time_index, issue_frequency_hours)
        for issue_hour, sub_idx in per_issue.items():
            if sub_idx.size > 0:
                slots[(doy, issue_hour)] = sub_idx
    return slots


def gather(da: xr.DataArray, indices: np.ndarray) -> xr.DataArray:
    """Select the time samples of ``da`` at ``indices``.

    ``time`` is kept as a single rechunked dask block so downstream
    quantile operations can run lazily over it.
    """
    sub = da.isel(time=indices)
    if sub.chunks is not None:
        sub = sub.chunk({"time": -1})
    return sub
