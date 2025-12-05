#!/usr/bin/env python3

import argparse
import logging as log

import dask
import numpy as np
import xarray as xr


def percentile_ufunc(data, p_values, axis):
    log.debug("Computing percentiles for array of size " + str(data.shape))

    perc_list = []
    for p in p_values:
        if p == 0:
            perc = np.amin(data, axis=axis)
        elif p == 50:
            perc = np.median(data, axis=axis)
        elif p == 100:
            perc = np.amax(data, axis=axis)
        else:
            perc = np.percentile(data, q=p, axis=axis)
        perc_list.append(perc)

    return perc_list


def percentiles_xarray(data, p_values, dim, core_dim):
    perc = xr.apply_ufunc(
        percentile_ufunc,
        data,
        input_core_dims=[["time", *core_dim]],
        dask="parallelized",
        output_core_dims=[["percentiles", *core_dim]],
        output_sizes={"percentiles": p_values.size},
        output_dtypes=[data.dtype],
        kwargs={"p_values": p_values, "axis": data.get_axis_num(dim)},
    )
    return perc


def date_year(dates):
    return dates.astype("datetime64[Y]").astype(int) + 1970


def date_month(dates):
    return dates.astype("datetime64[M]").astype(int) % 12 + 1


def date_day(dates):
    return (dates.astype("datetime64[D]") - dates.astype("datetime64[M]")).astype(
        int
    ) + 1


def date_hour(dates):
    return (dates.astype("datetime64[h]") - dates.astype("datetime64[D]")).astype(int)


def date_ymd(date):
    return date.astype("datetime64[D]")


def clim_dates(dates, window_size):

    index_1 = int(np.floor(window_size / 2.0))
    index_2 = int(np.ceil(window_size / 2.0))

    # remove leap days from array
    leap_days = (
        np.where(np.logical_and(date_month(dates) == 2, date_day(dates) == 29))[0] + 1
    )
    new_dates = np.delete(dates, leap_days)

    # extend array at both ends (31 days window)
    new_dates = np.concatenate((new_dates[-index_1:], new_dates, new_dates[:index_2]))

    return new_dates


def dates_range(dates, window, n_years, n_dates, day, ihour, freq, stride):

    hour = 24.0 / freq * ihour
    date_index = day * freq + ihour

    index_range = np.arange(0, window, stride, dtype=int)[None, :]
    # index_range[:, 2] = -1
    index_range = index_range[index_range >= 0]
    indexer = (
        date_index
        + index_range
        + np.arange(0, n_years * n_dates, n_dates, dtype=int)[:, None]
    ).flatten()
    log.debug("{} ?= {}".format(indexer.size, (int(window / stride) + 1) * n_years))
    assert (
        indexer.size == (int(window / stride) + 1) * n_years
    )  # MUST be +1 to include current day
    dates_clim = dates[indexer]

    hour_mask = np.where(date_hour(dates_clim) == hour)
    dates_clim = dates_clim[hour_mask]

    return dates_clim


def coord_leap_year_dmh(freq):
    if freq == 4:
        dates = np.arange("2020-01-01T06", "2021-01-01T06", 6, dtype="datetime64[h]")
    else:
        dates = np.arange("2020-01-02T00", "2021-01-02T00", 24, dtype="datetime64[h]")

    dates_str = np.datetime_as_string(dates, unit="h")

    print(dates_str)
    # days_months = np.char.replace(dates_str, '2020-', '')
    days_months = [date[5:] for date in dates_str]

    return days_months


def compute_climatology(p_values, n_days, window, freq, stride, dates, dis, core_dim):

    # get dates stats
    n_days_wo_leap = 365
    n_years = int((dates.size - window) / (n_days_wo_leap * freq))
    assert dates.size == n_days_wo_leap * n_years * freq + window

    percentiles_list = []

    log.info("Registering daily percentiles")
    for day in range(n_days):

        # replace leap days by February 28n_days_wo_leap
        if day > 58:
            day = day - 1

        tmp_percentiles = []
        for ihour in range(freq):
            hour = 24.0 / freq * ihour

            log.debug("day: {}, hour: {}".format(day, hour))

            # Apply 31 days window around date
            dates_clim = dates_range(
                dates, window, n_years, n_days_wo_leap * freq, day, ihour, freq, stride
            )
            log.debug(dates_clim[: int(window / stride + 1)])
            percentiles = dask.delayed(percentiles_xarray)(
                dis.sel(time=dates_clim), p_values, "time", core_dim
            )
            percentiles = percentiles.rename({"percentiles": "ensemble"})
            tmp_percentiles.append(percentiles)

        tmp_percentiles = dask.compute(*tmp_percentiles)
        percentiles_list.extend(tmp_percentiles)

    log.info("Exporting climatology dataset")
    new_coords = {"time": coord_leap_year_dmh(freq)}
    new_coords.update({dim: dis.coords[dim] for dim in core_dim})
    clim = xr.DataArray(
        np.zeros((n_days * freq, p_values.size, dis[0].size)),
        dims=("time", "ensemble", *core_dim),
        coords=new_coords,
    )
    for idate in range(n_days * freq):
        clim[idate] = percentiles_list[idate]
    print(clim)
    clim = clim.rename("dis")

    return clim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--output", required=True, help="output climatology file")
    parser.add_argument(
        "--window", type=int, default=30, help="moving window for percentiles (in days)"
    )
    parser.add_argument(
        "--timestep", type=int, default=6, help="reanalysis timestep (in hours)"
    )
    parser.add_argument(
        "--n_days",
        type=int,
        default=366,
        help="number of days to compute in climatology",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride used to compute the climatology (7 for seasonal)",
    )
    parser.add_argument(
        "--start_date",
        default="1991-01-01T06:00:00",
        help="start date of the climatology",
    )
    parser.add_argument(
        "--end_date", default="2020-01-01T00:00:00", help="end date of the climatology"
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
        level=args.log,
        format="dummy-clim - (%(processName)-10s) %(levelname)s: %(message)s",
    )

    log.info("Computing a dummy climatology from reanalysis dataset")

    log.info(" - dask: {}".format(args.scheduler))
    log.info(" - reanalysis: {}".format(args.reanalysis))
    log.info(" - output: {}".format(args.output))
    log.info(" - window: {}".format(args.window))
    log.info(" - timestep: {}".format(args.timestep))
    log.info(" - stride: {}".format(args.stride))
    log.info(" - n_days: {}".format(args.n_days))

    freq = int(24 / args.timestep)
    log.info("Computing climatology using {} steps per day".format(freq))

    stride = args.stride
    window = args.window
    step_window = window * freq - (window * freq) % stride + 1

    # considering statistics on a leap year
    n_days_w_leap = 366
    n_days = args.n_days

    # percentiles
    p_values = np.arange(0, 101, 10)

    with dask.config.set(scheduler=args.scheduler):

        # read reanalysis dataset
        reanalysis_file = args.reanalysis
        ds_reanalysis = xr.open_dataset(reanalysis_file, chunks={})

        # extract core dims from dataset
        core_dim = list(ds_reanalysis.dims)
        core_dim.remove("time")
        log.info("Core dimensions found in dataset: {}".format(core_dim))
        dis = (
            ds_reanalysis["dis"].sel(time=slice(args.start_date, args.end_date)).load()
        )
        log.info("discharge dataset {}".format(dis))

        dates = clim_dates(dis.time.values, step_window)
        dates_index = np.arange(dates.size, dtype=int)

        clim = compute_climatology(
            p_values, n_days, step_window, freq, stride, dates, dis, core_dim
        )

        clim.to_netcdf(args.output)
