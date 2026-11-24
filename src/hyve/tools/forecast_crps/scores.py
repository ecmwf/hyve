# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Dates helpers and the scoring loop over reforecast files."""

import datetime
import glob
import logging as log
import os
from os import path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from hyve.tools.forecast_crps.chunks import chunk_spatially
from hyve.tools.forecast_crps.crps import forecast_crps, persistence_crps
from hyve.tools.forecast_crps.matching import (
    check_covers_reference,
    match_coords,
    narrow,
    relabel,
)


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


def latest_first(reforecast_dir):
    """Reforecast files, latest by sorted name first.

    The first file is the reference: its grid is the one every score is
    computed and accumulated on.
    """
    return sorted(glob.glob(os.path.join(reforecast_dir, "*.nc")))[::-1]


def _onto_reference(score, ref_coords, spatial_dims):
    """`score` selected by label onto the reference grid, time first."""
    return score.sel(ref_coords).transpose("time", *spatial_dims)


def compute_score(
    out_dir,
    reforecast_dir,
    ds_reanalysis,
    ds_clim,
    spatial_dims,
    with_init=False,
    coord_tolerance=None,
    chunk_size=None,
):
    """Score every reforecast file against the reanalysis and write the results.

    The latest reforecast file defines the reference grid: the reanalysis and
    the other files are matched to it (see `match_coords`) and carry its
    coordinate values, so all scores align by label. `ds_clim`, if given,
    must already be on that grid.
    """

    log.info("\nComputing crps and crpss\n")
    spatial_dims = list(spatial_dims)

    reforecast_files = latest_first(reforecast_dir)
    ref_name = f"reforecast file {os.path.basename(reforecast_files[0])}"
    ds_ref = xr.open_dataset(reforecast_files[0])
    matches = match_coords(
        ds_ref, ds_reanalysis, spatial_dims, ref_name, "reanalysis", coord_tolerance
    )
    ds_reanalysis = relabel(ds_reanalysis, matches, ds_ref).transpose(
        "time", *spatial_dims
    )
    da_ref = narrow(ds_ref, matches)["dis"].sel(ensemble=0)
    da_ref = da_ref.transpose("time", *spatial_dims)
    ref_coords = {dim: da_ref[dim].values for dim in spatial_dims}

    def zeros():
        return xr.DataArray(
            np.zeros(da_ref.shape),
            dims=da_ref.dims,
            coords={dim: ref_coords[dim] for dim in spatial_dims},
        )

    crps_refc_mean = zeros()
    crps_pers_mean = zeros()
    if ds_clim is not None:
        crps_clim_mean = zeros()

    n_dates = len(reforecast_files)
    log.info("Number of reforecast datasets in folder: " + str(n_dates))

    count = 0
    for ifile, reforecast_path in enumerate(reforecast_files):

        log.info("- {}: {}".format(ifile, os.path.basename(reforecast_path)))
        file_name = f"reforecast file {os.path.basename(reforecast_path)}"
        ds_reforecast = chunk_spatially(
            xr.open_dataset(reforecast_path), spatial_dims, chunk_size
        )

        # the reanalysis already carries the reference values: match the file
        # to them, and require that it covers every reference point
        matches = match_coords(
            ds_reanalysis,
            ds_reforecast,
            spatial_dims,
            "reanalysis",
            file_name,
            coord_tolerance,
        )
        check_covers_reference(matches, ref_coords, file_name, ref_name)
        ds_reforecast = relabel(ds_reforecast, matches, ds_reanalysis)

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
        reanalysis = ds_reanalysis.reindex(time=date_range.values)
        gaps = reanalysis.isnull() & scored
        if gaps.any():
            msg = f"Gaps in the reanalysis data for base date {base_date:%Y-%m-%d %H}h. Skipping"
            all_missing = (reanalysis.isnull() | ~scored).all(dim=spatial_dims)
            if all_missing.any():
                missing_steps = date_range.values[all_missing.values]
                msg += f"\n Gaps at all points for step(s) {missing_steps.tolist()}"
            log.info(msg)
            continue
        persistence = ds_reanalysis.reindex(time=[date_persistence])
        if persistence.isnull().all():
            log.info(
                f"Cannot build persistence with empty step for date {date_persistence:%Y-%m-%d %H}h. Skipping"
            )
            continue

        crps_pers = persistence_crps(reanalysis, persistence).astype(np.float64)
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
            crps_clim = _onto_reference(crps_clim, ref_coords, spatial_dims)
        else:
            crps_refo, crps_pers = dask.compute(crps_refo, crps_pers)
        crps_refo = _onto_reference(crps_refo, ref_coords, spatial_dims)
        crps_pers = _onto_reference(crps_pers, ref_coords, spatial_dims)

        # write forecast files
        crps_refo = crps_refo.rename("crps")
        refo_file = path.join(out_dir, "crps_refo_{:%Y%m%d}.nc".format(base_date))
        crps_refo.to_netcdf(refo_file)
        pers_out = crps_pers.rename("crps")
        pers_file = path.join(out_dir, "crps_pers_{:%Y%m%d}.nc".format(base_date))
        pers_out.to_netcdf(pers_file)
        if ds_clim is not None:
            crps_clim = crps_clim.rename("crps")
            clim_file = path.join(out_dir, "crps_clim_{:%Y%m%d}.nc".format(base_date))
            crps_clim.to_netcdf(clim_file)

        # scores are averaged by lead position: the dates differ between files
        crps_refo = crps_refo.drop_vars("time")
        crps_pers = crps_pers.drop_vars("time")

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
    crps_pers_mean = crps_pers_mean.rename("crps")
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
