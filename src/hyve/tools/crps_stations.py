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


def crps_minmax(x, y):
    """
    Computes CRPS from x using y as reference,
    first x dimension must be ensembles, next dimensions can be arbitrary
    x: ensemble data (n_ens, n_points)
    y: observation/analysis data (n_points)
    returns: crps (n_points)
    REFERENCE
      Hersbach, 2000: Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems.
      Weather and Forecasting 15: 559-570.
    """

    # first sort ensemble
    x.sort(axis=0)

    # construct alpha and beta, size nens+1
    n_ens = x.shape[0]
    shape = (n_ens + 1,) + x.shape[1:]
    alpha = np.zeros(shape)
    beta = np.zeros(shape)

    # x[i+1]-x[i] and x[i]-y[i] arrays
    diffxy = x - y.reshape(1, *(y.shape))
    diffxx = x[1:] - x[:-1]  # x[i+1]-x[i], size ens-1

    # if i == 0
    alpha[0] = 0
    beta[0] = np.fmax(diffxy[0], 0)  # x(0)-y
    # if i == n_ens
    alpha[-1] = np.fmax(-diffxy[-1], 0)  # y-x(n)
    beta[-1] = 0
    # else
    alpha[1:-1] = np.fmin(
        diffxx, np.fmax(-diffxy[:-1], 0)
    )  # x(i+1)-x(i) or y-x(i) or 0
    beta[1:-1] = np.fmin(diffxx, np.fmax(diffxy[1:], 0))  # 0 or x(i+1)-y or x(i+1)-x(i)

    # compute crps
    p_exp = (np.arange(n_ens + 1) / float(n_ens)).reshape(n_ens + 1, *([1] * y.ndim))
    crps = np.sum(alpha * (p_exp**2) + beta * ((1 - p_exp) ** 2), axis=0)
    #
    # p = np.arange(n_ens+1)/float(n_ens)
    # alpha_mean = alpha.mean(axis=1)
    # beat_mean = beta.mean(axis=1)
    # crps = alpha_mean*(p**2) + beat_mean*((1-p)**2)
    # crps_mean = crps2.sum()
    #
    # p_exp = np.expand_dims(np.arange(n_ens+1)/float(n_ens), axis=1)
    # crps = np.nansum(alpha*(p_exp**2) + beta*((1-p_exp)**2), axis=0)
    # crps_mean = crps.mean()
    return crps


def crps_masked(x, y):
    n_steps = x.shape[0]

    mask = np.logical_not(np.isnan(x[0, 0]))
    crps = np.ones(y.shape) * np.nan
    for i in range(n_steps):
        xi = x[i]
        yi = y[i]
        crps_masked = crps_minmax(xi[:, mask], yi[mask])
        crps[i][mask] = crps_masked

    return crps


def forecast_crps(x, y, core_dims=["lat", "lon"]):
    crps = xr.apply_ufunc(
        crps_masked,
        x,
        y,
        input_core_dims=[["ensemble", *core_dims], core_dims],
        dask="parallelized",
        output_core_dims=[core_dims],
        output_dtypes=[np.float32],
        join="inner",
    )
    return crps


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


def coord_dmh(dates):
    dates_str = np.datetime_as_string(dates, unit="h")
    days_months = [date[5:] for date in dates_str]
    return days_months


def compute_score(
    out_dir, reforecast_dir, ds_reanalysis, ds_clim, core_dims, with_init=False
):

    log.info("\nComputing crps and crpss\n")

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
    for ifile, reforecast_path in enumerate(reforecast_files):

        log.info("- {}: {}".format(ifile, os.path.basename(reforecast_path)))
        ds_reforecast = xr.open_dataset(os.path.join(reforecast_dir, reforecast_path))

        set1 = set(ds_reforecast.station.values)
        set2 = set(ds_reanalysis.station.values)
        stations = list(set1.intersection(set2))
        ds_reforecast = ds_reforecast.sel(station=stations)
        ds_reanalysis_local = ds_reanalysis.sel(station=stations)

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
        reforecast = reforecast.transpose("time", "ensemble", *core_dims)

        # extract arrays of interest
        reanalysis = ds_reanalysis_local.reindex(time=date_range.values)
        if reanalysis.isnull().all():
            log.info(f"Any reanalysis data for base date {base_date:%Y-%m-%d %H}h. Skipping")
            continue
        persistence = ds_reanalysis_local.reindex(time=[date_persistence])
        if persistence.isnull().all():
            log.info(f"Cannot build persistence with empty step for date {date_persistence:%Y-%m-%d %H}h. Skipping")
            continue

        crps_pers = persistence_crps(reanalysis.values, persistence.values)
        crps_refo = forecast_crps(reforecast, reanalysis, core_dims=core_dims)
        if ds_clim is not None:
            log.debug(coord_dmh(date_range))
            climatology = ds_clim.sel(time=coord_dmh(date_range))
            climatology.coords["time"] = date_range
            crps_clim = forecast_crps(climatology, reanalysis, core_dims=core_dims)
            crps_refo, crps_pers, crps_clim = dask.compute(
                crps_refo, crps_pers, crps_clim
            )
        else:
            crps_refo, crps_pers = dask.compute(crps_refo, crps_pers)

        # write forecast files
        crps_refo = crps_refo.rename("crps")
        refo_file = path.join(out_dir, "crps_refo_{:%Y%m%d}.nc".format(base_date))
        crps_refo.to_netcdf(refo_file)
        pers_out = xr.zeros_like(crps_refo) + crps_pers
        pers_out.name = "crps"
        pers_file = path.join(out_dir, "crps_pers_{:%Y%m%d}.nc".format(base_date))
        pers_out.to_netcdf(pers_file)
        if ds_clim is not None:
            crps_clim = crps_clim.rename("crps")
            clim_file = path.join(out_dir, "crps_clim_{:%Y%m%d}.nc".format(base_date))
            crps_clim.to_netcdf(clim_file)

        crps_refo = crps_refo.drop_vars("time")

        crps_refc_mean = crps_refc_mean + crps_refo
        crps_pers_mean = crps_pers_mean + crps_pers
        if ds_clim is not None:
            crps_clim = crps_clim.drop("time")
            crps_clim_mean = crps_clim_mean + crps_clim

        count += 1

    # write statistics files
    crps_refc_mean = crps_refc_mean / count
    crps_pers_mean = crps_pers_mean / count

    log.debug(crps_refc_mean.isel(station=range(10)))
    crps_refc_mean = crps_refc_mean.rename("crps")
    crps_refc_mean.to_netcdf(os.path.join(out_dir, "crps_refo.nc"))

    log.debug(crps_pers_mean.isel(station=range(10)))
    crps_pers_mean = xr.zeros_like(crps_refc_mean) + crps_pers_mean
    crps_pers_mean.name = "crps"
    crps_pers_mean.to_netcdf(os.path.join(out_dir, "crps_pers.nc"))

    crpss_pers = 1 - (crps_refc_mean / crps_pers_mean)
    crpss_pers = crpss_pers.rename("crpss")

    log.debug(crpss_pers.isel(station=range(10)))
    crpss_pers.to_netcdf(os.path.join(out_dir, "crpss_pers.nc"))

    if ds_clim is not None:
        crps_clim_mean = crps_clim_mean / count
        log.debug(crps_clim_mean.isel(station=range(10)))
        crps_clim_mean = crps_clim_mean.rename("crps")
        crps_clim_mean.to_netcdf(os.path.join(out_dir, "crps_clim.nc"))

        crpss_clim = 1 - (crps_refc_mean / crps_clim_mean)
        crpss_clim = crpss_clim.rename("crpss")
        log.debug(crpss_clim.isel(station=range(10)))
        crpss_clim.to_netcdf(os.path.join(out_dir, "crpss_clim.nc"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--reforecast", required=True, help="reforecast dataset folder")
    parser.add_argument("--climatology", help="reanalysis dataset file")
    parser.add_argument("--output", help="output folder for individual crps values", default=".")
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
        log.info("Found {} files".format(len(reforecast_files)))
        log.debug(reforecast_files)
        ds_reforecast = xr.open_dataset(reforecast_files[0])

        ds_reforecast = ds_reforecast.rename({core_dim: "station"})
        ds_reanalysis = ds_reanalysis.rename({core_dim: "station"})

        log.info("Reanalysis dataset from {}:".format(args.reanalysis))
        log.debug(ds_reanalysis["dis"])
        log.info("Reforecast dataset from {}:".format(args.reforecast))
        log.debug(ds_reforecast["dis"])

        set1 = set(ds_reforecast.station.values)
        set2 = set(ds_reanalysis.station.values)
        stations = list(set1.intersection(set2))
        ds_reforecast = ds_reforecast.sel(station=stations)
        ds_reanalysis = ds_reanalysis.sel(station=stations)
        log.debug(ds_reanalysis["dis"])
        log.debug(ds_reforecast["dis"])

        ds_clim = None
        if args.climatology:
            ds_clim = xr.open_dataarray(args.climatology, chunks={"time": 1})
            if core_dim != "station":
                ds_clim = ds_clim.rename({core_dim: "station"})
            ds_clim = ds_clim.reindex_like(ds_reanalysis.station)
            ds_clim = ds_clim.assign_coords(station=ds_reanalysis.station)
            log.debug(ds_clim)

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


if __name__ == "__main__":
    main()
