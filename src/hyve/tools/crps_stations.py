#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import glob
import logging as log
import os
import sys
from os import path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from earthkit.meteo.score.array import crps as ensemble_crps


class StationMismatchError(ValueError):
    """Station IDs of two datasets cannot be matched."""


def station_id_kind(values):
    """Kind of a coordinate (station ID) array.

    One of "integer", "float", "text" (str), "bytes", None (empty) or a dtype
    name. Bytes are never decoded, so they are a kind of their own. Object
    arrays are classified by the type of their elements.
    """
    values = np.asarray(values)
    if values.size == 0:
        return None
    if values.dtype.kind in "iu":
        return "integer"
    if values.dtype.kind == "f":
        return "float"
    if values.dtype.kind == "U":
        return "text"
    if values.dtype.kind == "S":
        return "bytes"
    if values.dtype.kind == "O":
        if all(isinstance(v, str) for v in values.ravel()):
            return "text"
        if all(isinstance(v, bytes) for v in values.ravel()):
            return "bytes"
        if all(isinstance(v, (int, np.integer)) for v in values.ravel()):
            return "integer"
    return str(values.dtype)


def _coordinate_words(dim):
    """Words for the values of `dim` in messages: (plural IDs, count, singular)."""
    if dim == "station":
        return "station IDs", "stations", "station ID"
    return f"{dim} values", f"{dim} values", f"{dim} value"


def _describe_stations(name, values, dim="station"):
    values = np.asarray(values)
    sample = values[:5].tolist()
    _, count, _ = _coordinate_words(dim)
    return f"{name} (dtype {values.dtype}, {values.size} {count}, e.g. {sample})"


def check_same_station_kind(a, b, a_name, b_name, dim="station"):
    """Raise StationMismatchError if coordinates of `a` and `b` differ in kind.

    Values are never converted: integer and text IDs are different things.
    """
    kind_a, kind_b = station_id_kind(a), station_id_kind(b)
    if kind_a is not None and kind_b is not None and kind_a != kind_b:
        ids, _, singular = _coordinate_words(dim)
        raise StationMismatchError(
            "{} of different kinds ({} vs {}) cannot be matched: {} and {}."
            " Make the {} types consistent in the input datasets".format(
                ids,
                kind_a,
                kind_b,
                _describe_stations(a_name, a, dim),
                _describe_stations(b_name, b, dim),
                singular,
            )
        )


def common_stations(a, b, a_name, b_name, dim="station"):
    """Coordinate values of `dim` present in both `a` and `b`.

    Station IDs are returned sorted; values of any other dimension keep the
    order they have in `a`, so a descending latitude axis stays descending.

    Raises StationMismatchError if the values are of different kinds (integer
    vs float vs text vs bytes) or if the datasets share no value. Values are
    never converted: the comparison uses their Python values, so no common
    dtype is imposed on the two arrays (int64 and uint64 IDs are compared
    exactly, and bytes never equal str).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    check_same_station_kind(a, b, a_name, b_name, dim)
    in_b = set(b.tolist())
    if dim == "station":
        shared = sorted({value for value in a.tolist() if value in in_b})
    else:
        shared = [value for value in a.tolist() if value in in_b]
    if not shared:
        _, count, _ = _coordinate_words(dim)
        raise StationMismatchError(
            "no {} in common between {} and {}".format(
                count,
                _describe_stations(a_name, a, dim),
                _describe_stations(b_name, b, dim),
            )
        )
    return shared


def common_coords(a, b, spatial_dims, a_name, b_name):
    """Values of each spatial dimension present in both `a` and `b`.

    Returns a mapping of dimension name to shared values, ready for ``.sel``.
    Raises StationMismatchError if a spatial dimension is missing from either
    dataset or if the datasets share no value along a dimension.
    """
    for obj, name in ((a, a_name), (b, b_name)):
        missing = [dim for dim in spatial_dims if dim not in obj.dims]
        if missing:
            raise StationMismatchError(
                "{} has no {} dimension (its dimensions are {}). Set"
                " --spatial_dims to the spatial dimensions of the input"
                " datasets".format(
                    name, ", ".join(repr(d) for d in missing), list(obj.dims)
                )
            )
    return {
        dim: common_stations(a[dim].values, b[dim].values, a_name, b_name, dim)
        for dim in spatial_dims
    }


def _crps_over_last_axis(x, y):
    """earthkit-meteo CRPS for an ensemble held on the last axis of `x`.

    `y` is broadcast against the non-ensemble axes of `x`, so any number of
    leading (time, spatial or other) axes is supported.
    """
    x = np.moveaxis(np.asarray(x), -1, 0)
    y = np.broadcast_to(np.asarray(y), x.shape[1:])
    return ensemble_crps(x, y, nan_policy="propagate")


def forecast_crps(x, y, core_dims=None, ensemble_dim="ensemble"):
    """CRPS of the ensemble `x` against `y`, broadcast over all other dimensions.

    Only `ensemble_dim` is reduced. Every other dimension of `x` and `y` is
    broadcast by name and aligned on its coordinates (inner join), so the
    computation does not depend on which spatial dimensions are used
    (``station``, ``lat``/``lon``, ``y``/``x``, ...) or on their order.

    `core_dims` names the spatial dimensions; by default every dimension of
    `x` other than `ensemble_dim` and ``time``. Points where the first member
    is missing at the first index of every non-spatial dimension (the first
    lead time) are masked to NaN everywhere. The result has the non-spatial
    dimensions first, then `core_dims`.
    """
    if core_dims is None:
        core_dims = [d for d in x.dims if d not in (ensemble_dim, "time")]
    core_dims = list(core_dims)
    other_dims = [d for d in x.dims if d != ensemble_dim and d not in core_dims]

    first = {dim: 0 for dim in other_dims}
    first[ensemble_dim] = 0
    has_data = x.isel(first, drop=True).notnull()

    crps = xr.apply_ufunc(
        _crps_over_last_axis,
        x,
        y,
        input_core_dims=[[ensemble_dim], []],
        output_core_dims=[[]],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64],
        join="inner",
    )
    crps = crps.where(has_data)
    return crps.transpose(..., *core_dims)


@dask.delayed
def persistence_crps(reanalysis, persistence):

    # compute statistics for one forecast date
    crps_pers = np.fabs(reanalysis - persistence)

    return crps_pers


def shift_dates(dates, istart, n_dates=104, days=[0, 4]):

    dt = dates[1] - dates[0]

    if istart.dtype == np.int64:
        log.info("Start date coordinates not provided, reconstructing...")
        date_year = int(istart % n_dates)
        year_shift = int(istart / n_dates)
        week_shift = int(date_year / len(days))
        day_shift = days[int(date_year % len(days))]

        # compute forecast date
        fc_date_ref = (
            dates[0].values
            + np.timedelta64(day_shift, "D")
            + np.timedelta64(week_shift, "W")
        )
        date_pd = pd.to_datetime(fc_date_ref)
        year = date_pd.year + year_shift
        fc_date = np.datetime64(date_pd.replace(year=year))
    else:
        fc_date = istart + dt

    new_dates = xr.DataArray(np.empty(len(dates), dtype=np.datetime64), dims=["time"])
    for i in range(len(dates)):
        new_dates[i] = fc_date + i * dt

    return new_dates


def coord_doy(
    dates: list[datetime.datetime] | pd.DatetimeIndex | xr.DataArray,
) -> np.ndarray[int]:
    if isinstance(dates, (xr.DataArray, pd.DatetimeIndex)):
        doys = dates.dt.dayofyear.values
    else:
        doys = pd.to_datetime(dates).dayofyear.values
    return doys


def compute_score(
    out_dir, reforecast_dir, ds_reanalysis, ds_clim, spatial_dims, with_init=False
):

    log.info("\nComputing crps and crpss\n")
    spatial_dims = list(spatial_dims)

    reforecast_files = sorted(glob.glob(os.path.join(reforecast_dir, "*.nc")))[::-1]
    da_ref = xr.open_dataset(reforecast_files[0])
    common = common_coords(
        da_ref,
        ds_reanalysis,
        spatial_dims,
        f"reforecast file {os.path.basename(reforecast_files[0])}",
        "reanalysis",
    )
    da_ref = da_ref.sel(common)
    da_ref = da_ref["dis"].sel(ensemble=0).transpose("time", *spatial_dims)

    crps_refc_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)
    crps_pers_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)
    if ds_clim is not None:
        crps_clim_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)

    n_dates = len(reforecast_files)
    log.info("Number of reforecast datasets in folder: " + str(n_dates))

    count = 0
    for ifile, reforecast_path in enumerate(reforecast_files):

        log.info("- {}: {}".format(ifile, os.path.basename(reforecast_path)))
        ds_reforecast = xr.open_dataset(reforecast_path)

        common = common_coords(
            ds_reforecast,
            ds_reanalysis,
            spatial_dims,
            f"reforecast file {os.path.basename(reforecast_path)}",
            "reanalysis",
        )
        ds_reforecast = ds_reforecast.sel(common)
        ds_reanalysis_local = ds_reanalysis.sel(common).transpose("time", *spatial_dims)

        date_range = ds_reforecast.time
        step = (date_range.values[1] - date_range.values[0]).astype(
            "timedelta64[h]"
        ) / np.timedelta64(1, "h")
        base_date = pd.to_datetime(date_range[0].astype(int))
        date_persistence = base_date
        if not with_init:
            date_persistence = date_persistence - pd.DateOffset(hours=int(step))
        log.info(
            "First step: {:%Y-%m-%d}, time step: {:.0f} hours".format(base_date, step)
        )
        log.info(f"Persistence date is {date_persistence:%Y-%m-%d %H}h")

        reforecast = ds_reforecast["dis"].sel(time=date_range)
        reforecast = reforecast.transpose("time", "ensemble", *spatial_dims)
        # points the reforecast has data for; others are never scored
        scored = reforecast.isel(time=0, ensemble=0, drop=True).notnull()

        # extract arrays of interest
        reanalysis = ds_reanalysis_local.reindex(time=date_range.values)
        gaps = reanalysis.isnull() & scored
        if gaps.any():
            msg = f"Gaps in the reanalysis data for base date {base_date:%Y-%m-%d %H}h. Skipping"
            all_missing = (reanalysis.isnull() | ~scored).all(dim=spatial_dims)
            if all_missing.any():
                missing_steps = date_range.values[all_missing.values]
                msg += f"\n Gaps at all points for step(s) {missing_steps.tolist()}"
            log.info(msg)
            continue
        persistence = ds_reanalysis_local.reindex(time=[date_persistence])
        if persistence.isnull().all():
            log.info(
                f"Cannot build persistence with empty step for date {date_persistence:%Y-%m-%d %H}h. Skipping"
            )
            continue

        crps_pers = persistence_crps(reanalysis.values, persistence.values)
        crps_refo = forecast_crps(reforecast, reanalysis, core_dims=spatial_dims)
        if ds_clim is not None:
            log.debug(coord_doy(date_range))
            climatology = (
                ds_clim.sel(doy=coord_doy(date_range))
                .assign_coords(doy=date_range.values)
                .rename(doy="time")
            )
            crps_clim = forecast_crps(climatology, reanalysis, core_dims=spatial_dims)
            crps_refo, crps_pers, crps_clim = dask.compute(
                crps_refo, crps_pers, crps_clim
            )
        else:
            crps_refo, crps_pers = dask.compute(crps_refo, crps_pers)

        # write forecast files
        crps_refo = crps_refo.transpose("time", *spatial_dims).rename("crps")
        refo_file = path.join(out_dir, "crps_refo_{:%Y%m%d}.nc".format(base_date))
        crps_refo.to_netcdf(refo_file)
        pers_out = xr.zeros_like(crps_refo) + crps_pers
        pers_out.name = "crps"
        pers_file = path.join(out_dir, "crps_pers_{:%Y%m%d}.nc".format(base_date))
        pers_out.to_netcdf(pers_file)
        if ds_clim is not None:
            crps_clim = crps_clim.transpose("time", *spatial_dims).rename("crps")
            clim_file = path.join(out_dir, "crps_clim_{:%Y%m%d}.nc".format(base_date))
            crps_clim.to_netcdf(clim_file)

        crps_refo = crps_refo.drop_vars("time")

        crps_refc_mean = crps_refc_mean + crps_refo
        crps_pers_mean = crps_pers_mean + crps_pers
        if ds_clim is not None:
            crps_clim = crps_clim.drop_vars("time")
            crps_clim_mean = crps_clim_mean + crps_clim

        count += 1

    # write statistics files
    crps_refc_mean = crps_refc_mean / count
    crps_pers_mean = crps_pers_mean / count

    log.debug(crps_refc_mean)
    crps_refc_mean = crps_refc_mean.rename("crps")
    crps_refc_mean.to_netcdf(os.path.join(out_dir, "crps_refo.nc"))

    log.debug(crps_pers_mean)
    crps_pers_mean = xr.zeros_like(crps_refc_mean) + crps_pers_mean
    crps_pers_mean.name = "crps"
    crps_pers_mean.to_netcdf(os.path.join(out_dir, "crps_pers.nc"))

    crpss_pers = 1 - (crps_refc_mean / crps_pers_mean)
    crpss_pers = crpss_pers.rename("crpss")

    log.debug(crpss_pers)
    crpss_pers.to_netcdf(os.path.join(out_dir, "crpss_pers.nc"))

    if ds_clim is not None:
        crps_clim_mean = crps_clim_mean / count
        log.debug(crps_clim_mean)
        crps_clim_mean = crps_clim_mean.rename("crps")
        crps_clim_mean.to_netcdf(os.path.join(out_dir, "crps_clim.nc"))

        crpss_clim = 1 - (crps_refc_mean / crps_clim_mean)
        crpss_clim = crpss_clim.rename("crpss")
        log.debug(crpss_clim)
        crpss_clim.to_netcdf(os.path.join(out_dir, "crpss_clim.nc"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--reforecast", required=True, help="reforecast dataset folder")
    parser.add_argument("--climatology", help="reanalysis dataset file")
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
        "--with_init",
        action="store_true",
        help="Activate if reforecast dataset does not include initial condition",
    )
    parser.add_argument(
        "--scheduler",
        default="threads",
        choices=["synchronous", "threads"],
        help="reanalysis dataset file",
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
    except StationMismatchError as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)


def spatial_dims_from_args(args):
    """Spatial dimensions requested on the command line, station by default."""
    if getattr(args, "spatial_dims", None):
        return list(args.spatial_dims)
    if getattr(args, "core_dim", None):
        return [args.core_dim]
    return ["station"]


def run(args):
    log.info("Computing the scoring using crps approach")

    with dask.config.set(scheduler="threads"):

        spatial_dims = spatial_dims_from_args(args)
        log.info(f"Spatial dimensions: {spatial_dims}")

        # read reanalysis dataset
        ds_reanalysis = xr.open_dataset(args.reanalysis)

        # read first reforecast dataset for check
        reforecast_files = glob.glob(os.path.join(args.reforecast, "*.nc"))
        log.info("Found {} files".format(len(reforecast_files)))
        log.debug(reforecast_files)
        ds_reforecast = xr.open_dataset(reforecast_files[0])

        log.info("Reanalysis dataset from {}:".format(args.reanalysis))
        log.debug(ds_reanalysis["dis"])
        log.info("Reforecast dataset from {}:".format(args.reforecast))
        log.debug(ds_reforecast["dis"])

        common = common_coords(
            ds_reforecast,
            ds_reanalysis,
            spatial_dims,
            f"reforecast file {os.path.basename(reforecast_files[0])}",
            "reanalysis",
        )
        ds_reforecast = ds_reforecast.sel(common)
        ds_reanalysis = ds_reanalysis.sel(common)
        log.debug(ds_reanalysis["dis"])
        log.debug(ds_reforecast["dis"])

        ds_clim = None
        if args.climatology:
            ds_clim = xr.open_dataarray(args.climatology, chunks={"time": 1})
            common_coords(
                ds_clim, ds_reanalysis, spatial_dims, "climatology", "reanalysis"
            )
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
        )


if __name__ == "__main__":
    main()
