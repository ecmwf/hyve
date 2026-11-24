# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Command line interface of hyve-forecast-crps."""

import argparse
import logging as log
import math
import os
import sys

import dask
import xarray as xr

from hyve.tools.forecast_crps.chunks import chunk_spatially
from hyve.tools.forecast_crps.crps import check_dims
from hyve.tools.forecast_crps.errors import CoordinateMismatchError
from hyve.tools.forecast_crps.matching import match_coords, narrow, relabel
from hyve.tools.forecast_crps.scores import compute_score, latest_first


def _coord_tolerance(text):
    """argparse type: a finite, non-negative float."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be finite and >= 0")
    return value


def _positive_int(text):
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be >= 1")
    return value


def main():

    parser = argparse.ArgumentParser(prog="hyve-forecast-crps")
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--reforecast", required=True, help="reforecast dataset folder")
    parser.add_argument(
        "--climatology",
        help=(
            "climatology dataset file with dimensions ensemble, doy, issue_hour"
            " and the spatial dimensions (optional)"
        ),
    )
    parser.add_argument(
        "--output", help="output folder for individual crps values", default="."
    )
    parser.add_argument(
        "--spatial_dims",
        nargs="+",
        default=None,
        help=(
            "spatial dimensions of the input datasets, for example 'station',"
            " 'lat lon' or 'y x' (default: station)"
        ),
    )
    parser.add_argument(
        "--core_dim",
        default=None,
        help="single spatial dimension; same as --spatial_dims with one name",
    )
    parser.add_argument(
        "--coord_tolerance",
        type=_coord_tolerance,
        default=None,
        help=(
            "absolute tolerance for matching float coordinates (lat, lon, y, x),"
            " in the units of the coordinates; default: relative to the float"
            " precision of the coordinates"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=_positive_int,
        default=None,
        help=(
            "read every input lazily, in chunks of about this many spatial"
            " points per chunk; peak memory is close to the data held in memory"
            " plus threads x 8 x chunk bytes, counted in float64 (default: read"
            " whole files)"
        ),
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=None,
        help="number of dask threads (default: one per CPU core)",
    )
    parser.add_argument(
        "--with_init",
        action="store_true",
        help=(
            "the first time step of each reforecast file is the initial"
            " condition, which is then also the persistence forecast; without"
            " it the persistence date is one time step before the first step"
        ),
    )
    parser.add_argument(
        "--scheduler",
        default="threads",
        choices=["synchronous", "threads"],
        help="dask scheduler (default: threads)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )

    args = parser.parse_args()
    if args.spatial_dims is not None and args.core_dim is not None:
        parser.error("use either --spatial_dims or --core_dim, not both")
    log.basicConfig(
        level=args.log, format="crps - (%(processName)-10s) %(levelname)s: %(message)s"
    )

    try:
        run(args)
    except CoordinateMismatchError as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)


def spatial_dims_from_args(args):
    """Spatial dimensions requested on the command line, station by default."""
    if getattr(args, "spatial_dims", None):
        return list(args.spatial_dims)
    if getattr(args, "core_dim", None):
        return [args.core_dim]
    return ["station"]


def open_climatology(path, spatial_dims, chunk_size=None):
    """Climatology data array, chunked by day of year and spatial points.

    The climatology has no time dimension; one day of year per chunk means a
    forecast reads only the days it is scored against.
    """
    da = xr.open_dataarray(path)
    return chunk_spatially(da, spatial_dims, chunk_size, extra={"doy": 1})


def dask_settings(args):
    """dask configuration for the scheduler and thread count of `args`."""
    settings = {"scheduler": getattr(args, "scheduler", None) or "threads"}
    threads = getattr(args, "threads", None)
    if threads is not None:
        settings["num_workers"] = threads
    return settings


def run(args):
    log.info("Computing the scoring using crps approach")

    with dask.config.set(**dask_settings(args)):

        spatial_dims = spatial_dims_from_args(args)
        log.info(f"Spatial dimensions: {spatial_dims}")
        chunk_size = getattr(args, "chunk_size", None)
        if chunk_size is not None:
            log.info(f"Reading inputs in chunks of about {chunk_size} points")

        # read reanalysis dataset
        ds_reanalysis = chunk_spatially(
            xr.open_dataset(args.reanalysis), spatial_dims, chunk_size
        )

        # the reanalysis holds only time and the spatial dimensions
        check_dims(
            ds_reanalysis["dis"],
            "reanalysis 'dis'",
            [*spatial_dims, "time"],
            hint=". Set --spatial_dims to its spatial dimensions",
        )

        # the latest reforecast file is the reference grid, as in compute_score
        reforecast_files = latest_first(args.reforecast)
        log.info("Found {} files".format(len(reforecast_files)))
        log.debug(reforecast_files)
        ref_name = f"reforecast file {os.path.basename(reforecast_files[0])}"
        ds_reforecast = xr.open_dataset(reforecast_files[0])
        tolerance = getattr(args, "coord_tolerance", None)

        log.info("Reanalysis dataset from {}:".format(args.reanalysis))
        log.debug(ds_reanalysis["dis"])
        log.info("Reforecast dataset from {}:".format(args.reforecast))
        log.debug(ds_reforecast["dis"])

        # observations and climatology are relabelled to the forecast grid
        matches = match_coords(
            ds_reforecast,
            ds_reanalysis,
            spatial_dims,
            ref_name,
            "reanalysis",
            tolerance,
        )
        ds_reanalysis = relabel(ds_reanalysis, matches, ds_reforecast)
        ds_reforecast = narrow(ds_reforecast, matches)
        log.debug(ds_reanalysis["dis"])
        log.debug(ds_reforecast["dis"])

        ds_clim = None
        if args.climatology:
            ds_clim = open_climatology(args.climatology, spatial_dims, chunk_size)
            matches = match_coords(
                ds_reanalysis,
                ds_clim,
                spatial_dims,
                "reanalysis",
                "climatology",
                tolerance,
            )
            ds_clim = relabel(ds_clim, matches, ds_reanalysis)
            ds_clim = ds_clim.reindex(
                {dim: ds_reanalysis[dim].values for dim in spatial_dims}
            ).sel(issue_hour=0)
            log.debug(ds_clim)

        if args.output:
            os.makedirs(args.output, exist_ok=True)

        compute_score(
            args.output,
            args.reforecast,
            ds_reanalysis["dis"],
            ds_clim,
            spatial_dims,
            args.with_init,
            tolerance,
            chunk_size=chunk_size,
        )


if __name__ == "__main__":
    main()
