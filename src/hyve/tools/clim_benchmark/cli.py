"""Command-line interface for the clim-benchmark tool."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

import dask
import pandas as pd
import xarray as xr

from hyve.tools.clim_benchmark.config import ClimConfig
from hyve.tools.clim_benchmark.dates import infer_timestep
from hyve.tools.clim_benchmark.io import build_output_dataset, write_netcdf
from hyve.tools.clim_benchmark.percentiles import compute_climatology
from hyve.tools.clim_benchmark.sampling import build_slots


logger = logging.getLogger("hyve-clim-benchmark")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a day-of-year percentile climatology shaped as a "
            "synthetic forecast dataset (doy, issue_hour, ensemble)."
        ),
    )
    parser.add_argument(
        "--reanalysis", required=True, help="input reanalysis NetCDF file"
    )
    parser.add_argument("--output", required=True, help="output climatology NetCDF")
    parser.add_argument(
        "--variable",
        default=None,
        help="name of the variable to process (default: only data_var in file)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=31,
        help="centered window width in days (must be odd)",
    )
    parser.add_argument(
        "--stride",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="sampling stride inside the window",
    )
    parser.add_argument(
        "--issue-frequency",
        type=int,
        default=24,
        help="hours between forecast issues (divisor of 24, e.g. 24, 12, 6)",
    )
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=list(range(0, 101, 10)),
        help="percentile values in [0, 100]",
    )
    parser.add_argument("--start-date", default=None, help="climatology start date")
    parser.add_argument("--end-date", default=None, help="climatology end date")
    parser.add_argument(
        "--scheduler",
        default="threads",
        choices=["synchronous", "threads"],
        help="dask scheduler",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _select_variable(ds: xr.Dataset, name: str | None) -> xr.DataArray:
    if name is not None:
        return ds[name]
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"dataset has {len(data_vars)} variables; use --variable to "
            f"pick one of {data_vars}"
        )
    return ds[data_vars[0]]


def run(config: ClimConfig, input_path: str, output_path: str, variable: str | None) -> xr.Dataset:
    """Execute the full pipeline and return the output dataset."""
    ds_in = xr.open_dataset(input_path, chunks={})
    da = _select_variable(ds_in, variable)
    if config.start_date is not None or config.end_date is not None:
        da = da.sel(time=slice(config.start_date, config.end_date))

    time_index = pd.DatetimeIndex(da["time"].values)
    timestep = infer_timestep(time_index)
    timestep_hours = timestep.total_seconds() / 3600.0
    config.validate_against_data(timestep_hours)

    logger.info(
        "timestep=%s, window_days=%d, stride=%s, issue_frequency_hours=%d",
        timestep,
        config.window_days,
        config.stride,
        config.issue_frequency_hours,
    )

    slots = build_slots(
        da,
        window_days=config.window_days,
        stride=config.stride,
        issue_frequency_hours=config.issue_frequency_hours,
    )
    logger.info("Built %d (doy, issue_hour) slots", len(slots))

    clim_da = compute_climatology(da, slots, config.percentiles)
    ds_out = build_output_dataset(clim_da, config, timestep)

    if output_path:
        logger.info("Writing %s", output_path)
        write_netcdf(ds_out, output_path)
    return ds_out


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log,
        format="%(name)s - %(levelname)s: %(message)s",
    )

    config = ClimConfig(
        window_days=args.window_days,
        stride=args.stride,
        issue_frequency_hours=args.issue_frequency,
        percentiles=args.percentiles,
        start_date=args.start_date,
        end_date=args.end_date,
        scheduler=args.scheduler,
    )

    with dask.config.set(scheduler=config.scheduler):
        run(config, args.reanalysis, args.output, args.variable)


if __name__ == "__main__":
    main()
