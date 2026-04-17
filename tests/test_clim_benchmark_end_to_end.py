"""End-to-end tests for `hyve.tools.clim_benchmark`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools.clim_benchmark.cli import run
from hyve.tools.clim_benchmark.config import ClimConfig


def _make_daily_dataarray(start: str, end: str, var: str = "dis") -> xr.DataArray:
    """Build a 1-D (time-only) daily DataArray whose value equals the DOY."""
    time = pd.date_range(start, end, freq="1D")
    # Use a custom DOY value (leap-year aware) so it matches doy_of.
    from hyve.tools.clim_benchmark.dates import doy_of

    values = doy_of(time).astype(np.float64)
    return xr.DataArray(values, dims=("time",), coords={"time": time}, name=var)


def _make_6h_dataarray(start: str, end: str, var: str = "dis") -> xr.DataArray:
    """6-hourly DataArray whose value equals hour-of-day (0, 6, 12, 18)."""
    time = pd.date_range(start, end, freq="6h")
    values = time.hour.to_numpy().astype(np.float64)
    return xr.DataArray(values, dims=("time",), coords={"time": time}, name=var)


def _make_dataset_file(tmp_path, da: xr.DataArray) -> str:
    path = tmp_path / "input.nc"
    da.to_dataset().to_netcdf(path)
    return str(path)


# ---------------------------------------------------------------------------
# Use case 1: daily data, daily issue, weekly stride, 31-day window
# ---------------------------------------------------------------------------


def test_daily_data_daily_issue_weekly_stride(tmp_path):
    da = _make_daily_dataarray("2018-01-01", "2022-12-31")
    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "clim.nc")

    config = ClimConfig(
        window_days=31,
        stride="weekly",
        issue_frequency_hours=24,
        percentiles=[0, 50, 100],
    )
    ds = run(config, input_path, output_path, variable="dis")

    assert ds.sizes == {"doy": 366, "issue_hour": 1, "ensemble": 3}
    # For a DOY far from year boundary (e.g. 150), the 50th percentile
    # should be close to the DOY value because input(t) == doy(t).
    median = ds["dis"].sel(doy=150, issue_hour=0, ensemble=50).item()
    assert abs(median - 150) <= 3


# ---------------------------------------------------------------------------
# Use case 2: 6-hourly data, 12h issue frequency
# ---------------------------------------------------------------------------


def test_6h_data_12h_issue_separates_morning_afternoon(tmp_path):
    da = _make_6h_dataarray("2018-01-01", "2020-12-31T18")
    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "clim.nc")

    config = ClimConfig(
        window_days=15,
        stride="daily",
        issue_frequency_hours=12,
        percentiles=[0, 50, 100],
    )
    ds = run(config, input_path, output_path, variable="dis")

    assert ds.sizes["issue_hour"] == 2
    assert sorted(ds["issue_hour"].values.tolist()) == [0, 12]

    # issue_hour=0 pools only samples at hours 0 and 6 -> values in {0, 6}.
    slot_am = ds["dis"].sel(issue_hour=0, doy=150)
    assert slot_am.min().item() == 0
    assert slot_am.max().item() == 6
    # issue_hour=12 pools only hours 12 and 18 -> values in {12, 18}.
    slot_pm = ds["dis"].sel(issue_hour=12, doy=150)
    assert slot_pm.min().item() == 12
    assert slot_pm.max().item() == 18


# ---------------------------------------------------------------------------
# Output shape, metadata, variable name preservation
# ---------------------------------------------------------------------------


def test_output_shape_and_metadata(tmp_path):
    da = _make_daily_dataarray("2018-01-01", "2021-12-31", var="temperature")
    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "clim.nc")

    config = ClimConfig(
        window_days=7, stride="daily", issue_frequency_hours=24, percentiles=[10, 90]
    )
    ds = run(config, input_path, output_path, variable="temperature")

    assert "temperature" in ds.data_vars  # input variable name preserved
    assert ds.sizes == {"doy": 366, "issue_hour": 1, "ensemble": 2}
    assert ds.attrs["time_unit"] == "1D"
    assert ds.attrs["window_days"] == 7
    assert ds.attrs["stride"] == "daily"
    assert ds.attrs["issue_frequency_hours"] == 24

    # Auxiliary calendar coords.
    assert "month" in ds.coords
    assert "day" in ds.coords
    assert int(ds["month"].sel(doy=1)) == 1
    assert int(ds["day"].sel(doy=1)) == 1
    assert int(ds["month"].sel(doy=60)) == 2
    assert int(ds["day"].sel(doy=60)) == 29  # DOY 60 is Feb 29 in our calendar

    # File is readable by xarray.
    reloaded = xr.open_dataset(output_path)
    assert reloaded.sizes["doy"] == 366


# ---------------------------------------------------------------------------
# Leap-year semantics
# ---------------------------------------------------------------------------


def test_doy60_differs_from_doy61_when_leap_year_present(tmp_path):
    """With a leap year in the record, DOY 60 (Feb 29 + nearby days) and
    DOY 61 (Mar 1) should produce different median values because Feb 29
    is not collapsed into Mar 1."""
    # Synthetic value is sensitive to month & day but has noise via year.
    time = pd.date_range("2019-01-01", "2022-12-31", freq="1D")
    # Make Feb 29 a sharp outlier so pool contents matter.
    values = np.ones(len(time), dtype=np.float64)
    is_feb29 = (time.month == 2) & (time.day == 29)
    values[is_feb29] = 1000.0
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="dis")

    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "clim.nc")

    config = ClimConfig(
        window_days=1, stride="daily", issue_frequency_hours=24, percentiles=[100]
    )
    ds = run(config, input_path, output_path, variable="dis")

    # DOY 60 pool at window=1 is exactly Feb 29 samples -> max == 1000.
    # DOY 61 pool is Mar 1 samples -> max == 1.
    assert ds["dis"].sel(doy=60, issue_hour=0, ensemble=100).item() == 1000.0
    assert ds["dis"].sel(doy=61, issue_hour=0, ensemble=100).item() == 1.0


def test_doy366_fallback_when_no_leap_year(tmp_path):
    """With only non-leap years, DOY 366 pool duplicates DOY 365 samples
    (per the Dec 31 fallback rule); their percentiles must be equal."""
    time = pd.date_range("2021-01-01", "2023-12-31", freq="1D")
    # All non-leap years.
    assert not any(time.is_leap_year)
    values = np.arange(len(time), dtype=np.float64)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="dis")

    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "clim.nc")

    config = ClimConfig(
        window_days=1, stride="daily", issue_frequency_hours=24, percentiles=[0, 50, 100]
    )
    ds = run(config, input_path, output_path, variable="dis")

    p365 = ds["dis"].sel(doy=365, issue_hour=0).values
    p366 = ds["dis"].sel(doy=366, issue_hour=0).values
    np.testing.assert_array_equal(p365, p366)


# ---------------------------------------------------------------------------
# CLI entry point smoke test
# ---------------------------------------------------------------------------


def test_cli_main_smoke(tmp_path):
    from hyve.tools.clim_benchmark import main

    da = _make_daily_dataarray("2020-01-01", "2022-12-31")
    input_path = _make_dataset_file(tmp_path, da)
    output_path = str(tmp_path / "cli-clim.nc")

    main(
        [
            "--reanalysis",
            input_path,
            "--output",
            output_path,
            "--variable",
            "dis",
            "--window-days",
            "7",
            "--stride",
            "daily",
            "--issue-frequency",
            "24",
            "--percentiles",
            "0",
            "50",
            "100",
        ]
    )

    ds = xr.open_dataset(output_path)
    assert ds.sizes["doy"] == 366
    assert ds.sizes["issue_hour"] == 1
    assert ds.sizes["ensemble"] == 3
    assert "dis" in ds.data_vars


def test_cli_rejects_bad_window():
    from hyve.tools.clim_benchmark import main

    with pytest.raises(ValueError):
        main(
            [
                "--reanalysis",
                "unused.nc",
                "--output",
                "unused.nc",
                "--window-days",
                "30",  # even, invalid
                "--stride",
                "daily",
            ]
        )
