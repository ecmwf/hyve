# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""I/O helpers: auxiliary coordinates and NetCDF output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hyve.tools.clim_benchmark.config import ClimConfig
from hyve.tools.clim_benchmark.dates import doy_to_month_day


def add_calendar_coords(da: xr.DataArray) -> xr.DataArray:
    """Attach non-index ``month`` and ``day`` coords aligned with ``doy``."""
    doys = da["doy"].values
    months = np.empty_like(doys, dtype=np.int64)
    days = np.empty_like(doys, dtype=np.int64)
    for i, d in enumerate(doys):
        m, dd = doy_to_month_day(int(d))
        months[i] = m
        days[i] = dd
    return da.assign_coords(
        month=("doy", months),
        day=("doy", days),
    )


def build_output_dataset(
    da: xr.DataArray,
    config: ClimConfig,
    timestep: pd.Timedelta,
) -> xr.Dataset:
    """Wrap the climatology DataArray in a Dataset with metadata."""
    da = add_calendar_coords(da)
    ds = da.to_dataset()
    ds.attrs.update(
        {
            "time_unit": _format_timestep(timestep),
            "window_days": int(config.window_days),
            "stride": str(config.stride),
            "issue_frequency_hours": int(config.issue_frequency_hours),
            "percentiles": list(map(int, config.percentiles)),
        }
    )
    return ds


def _format_timestep(timestep: pd.Timedelta) -> str:
    """Render a timedelta as a compact ISO-ish string (e.g. ``6h``, ``1D``)."""
    total_seconds = int(timestep.total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}D"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    ds.to_netcdf(path)
