# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Scores produced by crps_stations as currently implemented.

These tests pin the numbers, file layout and edge-case handling of
``hyve.tools.crps_stations`` so that changes to how CRPS is computed can be
checked against the existing behaviour. Expected values come from
``reference_crps`` below, an independent implementation of the ensemble CRPS,
and from explicit selections of the synthetic inputs.

The ensemble CRPS itself is computed by earthkit-meteo; the tests here check
the scores hyve writes, not earthkit-meteo's internals.

Tests whose name ends in ``_current_behaviour`` pin behaviour that is likely to
be revisited (known crashes, silent all-NaN output). If such a test fails
after a change, decide whether the new behaviour is intended and update the
test deliberately.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.delayed import Delayed

from hyve.tools import crps_stations
from hyve.tools.crps_stations import (
    coord_doy,
    crps_masked,
    forecast_crps,
    persistence_crps,
)

STATIONS = [101, 102, 103, 104]
N_ENSEMBLE = 5
N_STEPS = 4
REANALYSIS_START = "2020-02-25"
REANALYSIS_DAYS = 12
FORECAST_DATES = ["2020-02-27", "2020-02-28", "2020-02-29"]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def reference_crps(ensemble, obs):
    """Ensemble CRPS in its kernel form, ensemble members along axis 0.

    CRPS = mean_i |x_i - y| - 1 / (2 m^2) * sum_ij |x_i - x_j|
    """
    ensemble = np.asarray(ensemble, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = ensemble.shape[0]
    skill = np.abs(ensemble - obs[None, ...]).mean(axis=0)
    spread = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...]).sum(axis=(0, 1))
    return skill - spread / (2 * m**2)


def test_reference_crps_hand_worked_example():
    # members 1, 2, 3 and observation 2:
    # mean |x - y| = 2/3, sum |x_i - x_j| = 8, 8 / (2 * 9) = 4/9
    assert reference_crps([[1.0], [2.0], [3.0]], [2.0]) == pytest.approx([2 / 9])


def test_reference_crps_single_member_is_absolute_error():
    assert reference_crps([[1.0, 4.0]], [3.0, 0.5]) == pytest.approx([2.0, 3.5])


# ---------------------------------------------------------------------------
# Ensemble CRPS for one step (crps_masked delegates to earthkit-meteo)
# ---------------------------------------------------------------------------


def single_step_crps(x, y):
    """CRPS of one step through crps_masked: x is (ensemble, point...)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return crps_masked(x[None, ...], y[None, ...])[0]


def test_crps_is_computed_by_earthkit_meteo(monkeypatch):
    import earthkit.meteo.score.array as ekm_array

    calls = []

    def spy(x, y, nan_policy="propagate"):
        calls.append(nan_policy)
        return ekm_array.crps(x, y, nan_policy=nan_policy)

    monkeypatch.setattr(crps_stations, "ensemble_crps", spy)
    single_step_crps([[1.0], [2.0], [3.0]], [2.0])
    assert calls == ["propagate"]
    assert crps_stations.ensemble_crps is spy
    assert not hasattr(crps_stations, "crps_minmax")


def test_crps_hand_worked_example():
    x = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 5.0]])
    y = np.array([2.0, 0.0])
    assert single_step_crps(x, y) == pytest.approx([2 / 9, 16 / 9])


def test_crps_single_member_is_absolute_error():
    x = np.array([[1.0, 4.0, -2.0]])
    y = np.array([3.0, 0.5, -2.0])
    assert single_step_crps(x, y) == pytest.approx([2.0, 3.5, 0.0])


def test_crps_is_zero_when_all_members_equal_observation():
    x = np.full((4, 3), 7.0)
    y = np.full(3, 7.0)
    assert single_step_crps(x, y) == pytest.approx(np.zeros(3))


@pytest.mark.parametrize("obs", [-5.0, 0.0, 2.5, 20.0])
def test_crps_observation_below_inside_and_above_ensemble(obs):
    x = np.array([[0.0], [1.0], [3.0], [10.0]])
    y = np.array([obs])
    assert single_step_crps(x, y) == pytest.approx(reference_crps(x, y))


def test_crps_matches_reference_on_random_points():
    rng = np.random.default_rng(10)
    x = rng.normal(5, 2, size=(11, 50))
    y = rng.normal(5, 2, size=50)
    assert single_step_crps(x, y) == pytest.approx(reference_crps(x, y))


def test_crps_accepts_multiple_trailing_dimensions():
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 10, size=(6, 3, 4))
    y = rng.uniform(0, 10, size=(3, 4))
    result = single_step_crps(x, y)
    assert result.shape == (3, 4)
    assert result == pytest.approx(reference_crps(x, y))


def test_crps_is_invariant_to_member_order():
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 10, size=(7, 20))
    y = rng.uniform(0, 10, size=20)
    shuffled = x[rng.permutation(7)]
    assert single_step_crps(x, y) == pytest.approx(single_step_crps(shuffled, y))


def test_crps_is_non_negative():
    rng = np.random.default_rng(13)
    x = rng.normal(0, 3, size=(9, 200))
    y = rng.normal(0, 3, size=200)
    assert (single_step_crps(x, y) >= 0).all()


def test_crps_leaves_inputs_unchanged():
    x = np.array([[[3.0, 1.0], [1.0, 2.0], [2.0, 5.0]]])
    y = np.array([[2.0, 0.0]])
    x_before, y_before = x.copy(), y.copy()
    crps_masked(x, y)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_crps_nan_observation_gives_nan():
    result = single_step_crps([[1.0, 1.0], [2.0, 2.0]], [np.nan, 1.5])
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(reference_crps([[1.0], [2.0]], [1.5])[0])


def test_crps_nan_member_gives_nan():
    # NaN in a member other than the first, so the point is not masked out
    x = [[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]]
    result = single_step_crps(x, [5.0, 5.0])
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(reference_crps([[1.0], [2.0], [3.0]], [5.0])[0])


# ---------------------------------------------------------------------------
# crps_masked
# ---------------------------------------------------------------------------


def test_crps_masked_scores_each_step_against_its_observation():
    rng = np.random.default_rng(20)
    x = rng.uniform(0, 10, size=(3, 5, 6))  # step, ensemble, point
    y = rng.uniform(0, 10, size=(3, 6))
    expected = np.stack([reference_crps(x[i], y[i]) for i in range(3)])
    assert crps_masked(x.copy(), y) == pytest.approx(expected)


def test_crps_masked_uses_first_step_and_member_as_mask():
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 10, size=(3, 5, 6))
    y = rng.uniform(0, 10, size=(3, 6))
    x[0, 0, 2] = np.nan
    result = crps_masked(x.copy(), y)
    assert np.isnan(result[:, 2]).all()
    others = [0, 1, 3, 4, 5]
    assert not np.isnan(result[:, others]).any()
    for i in range(3):
        assert result[i, others] == pytest.approx(
            reference_crps(x[i][:, others], y[i][others])
        )


def test_crps_masked_nan_outside_first_step_and_member_gives_nan_at_that_step():
    rng = np.random.default_rng(22)
    x = rng.uniform(0, 10, size=(2, 3, 4))
    y = rng.uniform(0, 10, size=(2, 4))
    x[1, 2, 1] = np.nan
    result = crps_masked(x.copy(), y)
    # the point is not masked: its other step keeps a score
    assert np.isnan(result[1, 1])
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1, [0, 2, 3]]).all()


# ---------------------------------------------------------------------------
# forecast_crps
# ---------------------------------------------------------------------------


def _forecast_and_obs(seed=30, stations=STATIONS):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2020-01-01", periods=N_STEPS, freq="D")
    fc = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, N_ENSEMBLE, len(stations))),
        dims=("time", "ensemble", "station"),
        coords={"time": time, "ensemble": np.arange(N_ENSEMBLE), "station": stations},
    )
    obs = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, len(stations))),
        dims=("time", "station"),
        coords={"time": time, "station": stations},
    )
    return fc, obs


def _expected_forecast_crps(fc, obs):
    fc = fc.transpose("time", "ensemble", "station")
    obs = obs.transpose("time", "station")
    return np.stack(
        [reference_crps(fc.values[i], obs.values[i]) for i in range(fc.sizes["time"])]
    )


def test_forecast_crps_values_and_dims():
    fc, obs = _forecast_and_obs()
    result = forecast_crps(fc, obs, core_dims=["station"])
    assert result.dims == ("time", "station")
    np.testing.assert_array_equal(result.station, STATIONS)
    np.testing.assert_array_equal(result.time, fc.time)
    assert result.values == pytest.approx(_expected_forecast_crps(fc, obs))


def test_forecast_crps_does_not_depend_on_input_dimension_order():
    fc, obs = _forecast_and_obs()
    expected = forecast_crps(fc.copy(), obs, core_dims=["station"])
    result = forecast_crps(
        fc.transpose("ensemble", "station", "time").copy(),
        obs.transpose("station", "time"),
        core_dims=["station"],
    )
    assert result.transpose("time", "station").values == pytest.approx(expected.values)


def test_forecast_crps_keeps_only_stations_in_both_inputs():
    fc, _ = _forecast_and_obs(stations=[1, 2, 3, 4])
    _, obs = _forecast_and_obs(seed=31, stations=[3, 4, 5])
    result = forecast_crps(fc.copy(), obs, core_dims=["station"])
    np.testing.assert_array_equal(result.station, [3, 4])
    expected = _expected_forecast_crps(fc.sel(station=[3, 4]), obs.sel(station=[3, 4]))
    assert result.values == pytest.approx(expected)


def test_forecast_crps_with_dask_inputs():
    fc, obs = _forecast_and_obs()
    expected = _expected_forecast_crps(fc, obs)
    result = forecast_crps(
        fc.chunk({"time": -1, "ensemble": -1, "station": -1}),
        obs.chunk({"time": -1, "station": -1}),
        core_dims=["station"],
    )
    assert result.chunks is not None
    assert result.compute().values == pytest.approx(expected)


# ---------------------------------------------------------------------------
# persistence_crps and coord_doy
# ---------------------------------------------------------------------------


def test_persistence_crps_is_delayed_absolute_error():
    result = persistence_crps(np.array([1.0, 5.0, -2.0]), np.array([3.0, 3.0, -2.0]))
    assert isinstance(result, Delayed)
    assert result.compute() == pytest.approx([2.0, 2.0, 0.0])


def test_coord_doy_from_dataarray_handles_leap_day():
    dates = xr.DataArray(pd.date_range("2020-02-28", periods=3, freq="D"))
    np.testing.assert_array_equal(coord_doy(dates), [59, 60, 61])


def test_coord_doy_from_list_of_datetimes():
    dates = [datetime.datetime(2020, 2, 29), datetime.datetime(2021, 12, 31)]
    np.testing.assert_array_equal(coord_doy(dates), [60, 365])


def test_coord_doy_from_datetimeindex_raises_current_behaviour():
    with pytest.raises(AttributeError):
        coord_doy(pd.date_range("2020-02-28", periods=3, freq="D"))


# ---------------------------------------------------------------------------
# End-to-end: hyve-crps-stations
# ---------------------------------------------------------------------------


class Inputs:
    """Synthetic inputs written to disk, kept in memory for expectations."""

    def __init__(
        self,
        root,
        dim="station",
        freq="D",
        forecast_dates=FORECAST_DATES,
        reanalysis_start=REANALYSIS_START,
        with_climatology=True,
    ):
        rng = np.random.default_rng(40)
        self.root = root
        self.freq = freq
        step = pd.Timedelta(1, "D") if freq == "D" else pd.Timedelta(freq)
        time = pd.date_range(reanalysis_start, periods=REANALYSIS_DAYS, freq=freq)
        self.reanalysis = xr.DataArray(
            rng.uniform(1, 10, size=(len(time), len(STATIONS))),
            dims=("time", dim),
            coords={"time": time, dim: STATIONS},
            name="dis",
        )

        self.reforecasts = {}
        for date in forecast_dates:
            ftime = pd.date_range(date, periods=N_STEPS, freq=step)
            self.reforecasts[pd.Timestamp(date)] = xr.DataArray(
                rng.uniform(1, 10, size=(N_ENSEMBLE, N_STEPS, len(STATIONS))),
                dims=("ensemble", "time", dim),
                coords={
                    "ensemble": np.arange(N_ENSEMBLE),
                    "time": ftime,
                    dim: STATIONS,
                },
                name="dis",
            )

        self.climatology = None
        if with_climatology:
            doy = np.arange(1, 367)
            self.climatology = xr.DataArray(
                rng.uniform(1, 10, size=(N_ENSEMBLE, len(doy), 2, len(STATIONS))),
                dims=("ensemble", "doy", "issue_hour", dim),
                coords={
                    "ensemble": np.arange(N_ENSEMBLE),
                    "doy": doy,
                    "issue_hour": [0, 12],
                    dim: STATIONS,
                },
                name="dis",
            )
        self.step = step

    def write(self):
        self.root.mkdir(exist_ok=True)
        self.reanalysis.to_dataset().to_netcdf(self.reanalysis_path)
        self.reforecast_dir.mkdir()
        for date, da in self.reforecasts.items():
            da.to_dataset().to_netcdf(
                self.reforecast_dir / f"reforecast_{date:%Y%m%d%H}.nc"
            )
        if self.climatology is not None:
            self.climatology.to_netcdf(self.climatology_path)
        return self

    @property
    def reanalysis_path(self):
        return self.root / "reanalysis.nc"

    @property
    def reforecast_dir(self):
        return self.root / "reforecast"

    @property
    def climatology_path(self):
        return self.root / "climatology.nc"

    @property
    def output(self):
        return self.root / "out"

    # expectations -----------------------------------------------------------

    def observed(self, date):
        return self.reanalysis.sel(time=self.reforecasts[date].time).values

    def expected_refo(self, date):
        fc = self.reforecasts[date].transpose("time", "ensemble", ...).values
        obs = self.observed(date)
        return np.stack([reference_crps(fc[i], obs[i]) for i in range(N_STEPS)])

    def expected_pers(self, date, with_init=False):
        persistence_date = date if with_init else date - self.step
        persistence = self.reanalysis.sel(time=persistence_date).values
        return np.abs(self.observed(date) - persistence[None, :])

    def expected_clim(self, date):
        time = self.reforecasts[date].time
        clim = (
            self.climatology.sel(issue_hour=0)
            .sel(doy=time.dt.dayofyear.values)
            .transpose("doy", "ensemble", ...)
            .values
        )
        obs = self.observed(date)
        return np.stack([reference_crps(clim[i], obs[i]) for i in range(N_STEPS)])


def run_cli(monkeypatch, inputs, *extra, climatology=True):
    argv = [
        "hyve-crps-stations",
        "--reanalysis",
        str(inputs.reanalysis_path),
        "--reforecast",
        str(inputs.reforecast_dir),
        "--output",
        str(inputs.output),
        *extra,
    ]
    if climatology and inputs.climatology is not None:
        argv += ["--climatology", str(inputs.climatology_path)]
    monkeypatch.setattr("sys.argv", argv)
    crps_stations.main()
    return inputs.output


def open_crps(path):
    with xr.open_dataset(path) as ds:
        return ds.load()


@pytest.fixture
def inputs(tmp_path):
    return Inputs(tmp_path / "case").write()


def test_cli_writes_expected_files(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs)
    dated = [
        f"crps_{kind}_{date:%Y%m%d}.nc"
        for kind in ("refo", "pers", "clim")
        for date in inputs.reforecasts
    ]
    totals = [
        "crps_refo.nc",
        "crps_pers.nc",
        "crps_clim.nc",
        "crpss_pers.nc",
        "crpss_clim.nc",
    ]
    assert sorted(os.listdir(out)) == sorted(dated + totals)


def test_cli_without_climatology_writes_no_climatology_files(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs, climatology=False)
    assert not [name for name in os.listdir(out) if "clim" in name]
    assert (out / "crpss_pers.nc").exists()


def test_cli_dated_files_layout(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs)
    date = pd.Timestamp(FORECAST_DATES[0])
    for kind in ("refo", "pers", "clim"):
        ds = open_crps(out / f"crps_{kind}_{date:%Y%m%d}.nc")
        assert list(ds.data_vars) == ["crps"]
        assert set(ds["crps"].dims) == {"time", "station"}
        assert ds["crps"].dtype == np.float64
        np.testing.assert_array_equal(ds.station, STATIONS)
        np.testing.assert_array_equal(ds.time, inputs.reforecasts[date].time)
    clim = open_crps(out / f"crps_clim_{date:%Y%m%d}.nc")
    assert int(clim.issue_hour) == 0


def test_cli_forecast_scores(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs)
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
        crps = crps.transpose("time", "station")
        assert crps.values == pytest.approx(inputs.expected_refo(date))


def test_cli_persistence_scores_use_reanalysis_one_step_before_start(
    monkeypatch, inputs
):
    out = run_cli(monkeypatch, inputs)
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_pers_{date:%Y%m%d}.nc")["crps"]
        crps = crps.transpose("time", "station")
        assert crps.values == pytest.approx(inputs.expected_pers(date))


def test_cli_with_init_uses_reanalysis_at_start_for_persistence(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs, "--with_init")
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_pers_{date:%Y%m%d}.nc")["crps"]
        crps = crps.transpose("time", "station")
        expected = inputs.expected_pers(date, with_init=True)
        assert crps.values == pytest.approx(expected)
        np.testing.assert_array_equal(crps.values[0], 0.0)


def test_cli_climatology_scores_use_day_of_year_and_issue_hour_zero(
    monkeypatch, inputs
):
    out = run_cli(monkeypatch, inputs)
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_clim_{date:%Y%m%d}.nc")["crps"]
        crps = crps.transpose("time", "station")
        assert crps.values == pytest.approx(inputs.expected_clim(date))


def test_cli_mean_scores_average_dates_by_lead_position(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs)
    dates = list(inputs.reforecasts)
    expected = {
        "refo": np.mean([inputs.expected_refo(d) for d in dates], axis=0),
        "pers": np.mean([inputs.expected_pers(d) for d in dates], axis=0),
        "clim": np.mean([inputs.expected_clim(d) for d in dates], axis=0),
    }
    for kind, values in expected.items():
        ds = open_crps(out / f"crps_{kind}.nc")
        crps = ds["crps"].transpose("time", "station")
        assert "time" not in ds.coords
        np.testing.assert_array_equal(ds.station, STATIONS)
        assert crps.values == pytest.approx(values)


def test_cli_skill_scores(monkeypatch, inputs):
    out = run_cli(monkeypatch, inputs)
    refo = open_crps(out / "crps_refo.nc")["crps"]
    for benchmark in ("pers", "clim"):
        ref = open_crps(out / f"crps_{benchmark}.nc")["crps"]
        crpss = open_crps(out / f"crpss_{benchmark}.nc")
        assert list(crpss.data_vars) == ["crpss"]
        expected = 1 - refo / ref
        assert crpss["crpss"].transpose(*expected.dims).values == pytest.approx(
            expected.values
        )


def test_cli_six_hourly_steps_use_six_hour_persistence(monkeypatch, tmp_path):
    inputs = Inputs(
        tmp_path / "case",
        freq="6h",
        forecast_dates=["2020-02-26 00:00", "2020-02-27 00:00"],
        reanalysis_start="2020-02-25 00:00",
        with_climatology=False,
    )
    inputs.reanalysis = xr.DataArray(
        np.random.default_rng(41).uniform(1, 10, size=(16, len(STATIONS))),
        dims=("time", "station"),
        coords={
            "time": pd.date_range("2020-02-25", periods=16, freq="6h"),
            "station": STATIONS,
        },
        name="dis",
    )
    inputs.write()
    out = run_cli(monkeypatch, inputs)
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_pers_{date:%Y%m%d}.nc")["crps"]
        crps = crps.transpose("time", "station")
        assert crps.values == pytest.approx(inputs.expected_pers(date))


def test_cli_skips_dates_with_reanalysis_gaps(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case")
    # 2020-03-03 is only inside the window of the last forecast date
    gap = inputs.reanalysis.copy()
    gap.loc[{"time": "2020-03-03", "station": STATIONS[0]}] = np.nan
    inputs.reanalysis = gap
    inputs.write()
    out = run_cli(monkeypatch, inputs)
    kept = [pd.Timestamp(d) for d in FORECAST_DATES[:2]]
    assert not (out / "crps_refo_20200229.nc").exists()
    for date in kept:
        assert (out / f"crps_refo_{date:%Y%m%d}.nc").exists()
    mean = open_crps(out / "crps_refo.nc")["crps"].transpose("time", "station")
    expected = np.mean([inputs.expected_refo(d) for d in kept], axis=0)
    assert mean.values == pytest.approx(expected)


def test_cli_skips_dates_without_persistence(monkeypatch, tmp_path):
    # reanalysis starts on the first forecast date, so its persistence
    # date (one step earlier) is missing
    inputs = Inputs(tmp_path / "case", reanalysis_start=FORECAST_DATES[0]).write()
    out = run_cli(monkeypatch, inputs, climatology=False)
    assert not (out / "crps_refo_20200227.nc").exists()
    assert (out / "crps_refo_20200228.nc").exists()
    assert (out / "crps_refo_20200229.nc").exists()


def test_cli_all_dates_skipped_writes_nan_means_current_behaviour(
    monkeypatch, tmp_path
):
    inputs = Inputs(tmp_path / "case")
    # 2020-03-01 is inside the window of every forecast date
    gap = inputs.reanalysis.copy()
    gap.loc[{"time": "2020-03-01", "station": STATIONS[0]}] = np.nan
    inputs.reanalysis = gap
    inputs.write()
    out = run_cli(monkeypatch, inputs, climatology=False)
    assert sorted(os.listdir(out)) == ["crps_pers.nc", "crps_refo.nc", "crpss_pers.nc"]
    assert open_crps(out / "crps_refo.nc")["crps"].isnull().all()


def test_cli_station_missing_in_reforecast_current_behaviour(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case")
    missing = STATIONS[1]
    for date, da in inputs.reforecasts.items():
        da = da.copy()
        da.loc[{"station": missing}] = np.nan
        inputs.reforecasts[date] = da
    inputs.write()
    out = run_cli(monkeypatch, inputs)
    refo = open_crps(out / "crps_refo.nc")["crps"]
    pers = open_crps(out / "crps_pers.nc")["crps"]
    clim = open_crps(out / "crps_clim.nc")["crps"]
    crpss = open_crps(out / "crpss_pers.nc")["crpss"]
    assert refo.sel(station=missing).isnull().all()
    assert not pers.sel(station=missing).isnull().any()
    assert not clim.sel(station=missing).isnull().any()
    assert crpss.sel(station=missing).isnull().all()
    others = [s for s in STATIONS if s != missing]
    assert not refo.sel(station=others).isnull().any()


def test_cli_core_dim_other_than_station_fails_current_behaviour(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case", dim="id", with_climatology=False).write()
    with pytest.raises(AttributeError):
        run_cli(monkeypatch, inputs, "--core_dim", "id")
