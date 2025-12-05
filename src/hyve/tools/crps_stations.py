#!/usr/bin/env python3

import argparse
import glob
import logging as log
import os
from os import path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from danu import stats, utils


@dask.delayed
def persistence_crps(reanalysis, persistence):

    # compute statistics for one forecast date
    crps_pers = np.fabs(reanalysis - persistence)

    return crps_pers


def shift_dates(dates, istart, n_dates=104, days=[0, 4]):

    dt = dates[1] - dates[0]

    if istart.dtype == np.int64:
        print("startdate coordinates not provided, reconstructing...")
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


def coord_dmh(dates):
    dates_str = np.datetime_as_string(dates, unit="h")
    days_months = [date[5:] for date in dates_str]
    return days_months


@utils.timer
def compute_score(
    out_dir, reforecast_dir, ds_reanalysis, ds_clim, core_dims, with_init=False
):

    print("\nComputing crps and crpss\n")

    reforecast_files = glob.glob(reforecast_dir + "/*.nc")
    da_ref = xr.open_dataset(os.path.join(reforecast_dir, reforecast_files[0]))
    set1 = set(da_ref.station.values)
    set2 = set(ds_reanalysis.station.values)
    stations = list(set1.intersection(set2))
    da_ref = da_ref.sel(station=stations)
    da_ref = da_ref["dis"].sel(ensemble=0)

    crps_refc_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)
    crps_pers_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)
    if ds_clim is not None:
        crps_clim_mean = xr.DataArray(np.zeros(da_ref.shape), dims=da_ref.dims)

    n_dates = len(reforecast_files)
    log.info("Number of reforecast datasets in folder: " + str(n_dates))

    count = 0
    for reforecast_path in reforecast_files:

        log.info("- {}: {}".format(count, reforecast_path))
        ds_reforecast = xr.open_dataset(os.path.join(reforecast_dir, reforecast_path))

        set1 = set(ds_reforecast.station.values)
        set2 = set(ds_reanalysis.station.values)
        stations = list(set1.intersection(set2))
        ds_reforecast = ds_reforecast.sel(station=stations)
        ds_reanalysis = ds_reanalysis.sel(station=stations)

        date_range = ds_reforecast.time
        step = (date_range.values[1] - date_range.values[0]).astype(
            "timedelta64[h]"
        ) / np.timedelta64(1, "h")
        date_persistence = date_range[0]
        if not with_init:
            date_persistence = date_persistence - np.timedelta64(int(step), "h")
        log.info(
            "First step: {}, time step: {} hours".format(date_range.values[0], step)
        )
        log.info("Persistence date is {}".format(date_persistence.values))

        reforecast = ds_reforecast["dis"].sel(time=date_range)
        reforecast = reforecast.transpose("time", "ensemble", *core_dims)

        # extract arrays of interest
        reanalysis = ds_reanalysis.sel(time=date_range)
        persistence = ds_reanalysis.sel(time=date_persistence)

        crps_pers = persistence_crps(reanalysis, persistence)
        crps_refo = stats.forecast_crps(reforecast, reanalysis, core_dims=core_dims)
        if ds_clim is not None:
            log.debug(coord_dmh(date_range))
            climatology = ds_clim.sel(time=coord_dmh(date_range))
            climatology.coords["time"] = date_range
            crps_clim = stats.forecast_crps(
                climatology, reanalysis, core_dims=core_dims
            )
            crps_refo, crps_pers, crps_clim = dask.compute(
                crps_refo, crps_pers, crps_clim
            )
        else:
            crps_refo, crps_pers = dask.compute(crps_refo, crps_pers)

        # write forecast files
        if out_dir:
            crps_refo = crps_refo.rename("crps")
            refo_file = path.join(out_dir, "crps_refo_{:04}.nc".format(int(count)))
            crps_refo.to_netcdf(refo_file)
            crps_pers = crps_pers.rename("crps")
            pers_file = path.join(out_dir, "crps_pers_{:04}.nc".format(int(count)))
            crps_pers.to_netcdf(pers_file)
            if ds_clim is not None:
                crps_clim = crps_clim.rename("crps")
                clim_file = path.join(out_dir, "crps_clim_{:04}.nc".format(int(count)))
                crps_clim.to_netcdf(clim_file)

        crps_refo = crps_refo.drop("time")
        crps_pers = crps_pers.drop("time")
        crps_refc_mean = crps_refc_mean + crps_refo
        crps_pers_mean = crps_pers_mean + crps_pers
        if ds_clim is not None:
            crps_clim = crps_clim.drop("time")
            crps_clim_mean = crps_clim_mean + crps_clim

        count += 1

    # write statistics files
    crps_refc_mean = crps_refc_mean / n_dates
    crps_pers_mean = crps_pers_mean / n_dates

    print(crps_refc_mean.isel(station=range(10)))
    crps_refc_mean = crps_refc_mean.rename("crps")
    crps_refc_mean.to_netcdf("crps_refo.nc")
    print(crps_pers_mean.isel(station=range(10)))
    crps_pers_mean = crps_pers_mean.rename("crps")
    crps_pers_mean.to_netcdf("crps_pers.nc")

    crpss_pers = 1 - (crps_refc_mean / crps_pers_mean)
    crpss_pers = crpss_pers.rename("crpss")
    print(crpss_pers.isel(station=range(10)))
    crpss_pers.to_netcdf("crpss_pers.nc")

    if ds_clim is not None:
        crps_clim_mean = crps_clim_mean / n_dates
        print(crps_clim_mean.isel(station=range(10)))
        crps_clim_mean = crps_clim_mean.rename("crps")
        crps_clim_mean.to_netcdf("crps_clim.nc")

        crpss_clim = 1 - (crps_refc_mean / crps_clim_mean)
        crpss_clim = crpss_clim.rename("crpss")
        print(crpss_clim.isel(station=range(10)))
        crpss_clim.to_netcdf("crpss_clim.nc")

    assert count == n_dates


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--reforecast", required=True, help="reforecast dataset folder")
    parser.add_argument("--climatology", help="reanalysis dataset file")
    parser.add_argument("--output", help="output folder for individual crps values")
    parser.add_argument("--core_dim", default="station", help="name of core dimension")
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
    log.basicConfig(
        level=args.log, format="crps - (%(processName)-10s) %(levelname)s: %(message)s"
    )

    log.info("Computing the scoring using crps approach")

    with dask.config.set(scheduler="threads"):

        core_dim = args.core_dim
        core_dims = ["station"]

        # read reanalysis dataset
        ds_reanalysis = xr.open_dataset(args.reanalysis)

        # read first reforecast dataset for check
        reforecast_files = glob.glob(os.path.join(args.reforecast, "*.nc"))
        print("Found {} files".format(len(reforecast_files)))
        print(reforecast_files)
        ds_reforecast = xr.open_dataset(reforecast_files[0])

        ds_reforecast = ds_reforecast.rename({core_dim: "station"})
        ds_reanalysis = ds_reanalysis.rename({core_dim: "station"})

        print("Reanalysis dataset from {}:".format(args.reanalysis))
        print(ds_reanalysis["dis"])
        print("Reforecast dataset from {}:".format(args.reforecast))
        print(ds_reforecast["dis"])

        set1 = set(ds_reforecast.station.values)
        set2 = set(ds_reanalysis.station.values)
        stations = list(set1.intersection(set2))
        ds_reforecast = ds_reforecast.sel(station=stations)
        ds_reanalysis = ds_reanalysis.sel(station=stations)
        print(ds_reanalysis["dis"])
        print(ds_reforecast["dis"])

        ds_clim = None
        if args.climatology:
            ds_clim = xr.open_dataarray(args.climatology, chunks={"time": 1})
            if core_dim != "station":
                ds_clim = ds_clim.rename({core_dim: "station"})
            ds_clim = ds_clim.assign_coords(
                {"station": ds_reanalysis.coords["station"]}
            )
            print(ds_clim)

        if args.output:
            os.makedirs(args.output, exist_ok=True)

        compute_score(
            args.output,
            args.reforecast,
            ds_reanalysis["dis"],
            ds_clim,
            core_dims,
            args.with_init,
        )
