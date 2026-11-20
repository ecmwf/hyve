# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Scores produced by forecast_crps as currently implemented.

These tests pin the numbers, file layout and edge-case handling of
``hyve.tools.forecast_crps`` so that changes to how CRPS is computed can be
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
import glob
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools import forecast_crps as forecast_crps_tool
from hyve.tools.forecast_crps import (
    DimensionMismatchError,
    GridMismatchError,
    coord_doy,
)
from hyve.tools.forecast_crps import crps as crps_module
from hyve.tools.forecast_crps import forecast_crps, persistence_crps

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
# Ensemble CRPS (forecast_crps delegates to earthkit-meteo)
# ---------------------------------------------------------------------------


def as_ensemble(x, y):
    """DataArrays for an ensemble `x` (ensemble, point...) and observation `y`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    point_dims = [f"p{i}" for i in range(y.ndim)]
    return (
        xr.DataArray(x, dims=("ensemble", *point_dims)),
        xr.DataArray(y, dims=point_dims),
    )


def single_step_crps(x, y):
    """CRPS through forecast_crps of `x` (ensemble, point...) against `y`."""
    fc, obs = as_ensemble(x, y)
    return forecast_crps(fc, obs).values


def test_crps_is_computed_by_earthkit_meteo(monkeypatch):
    import earthkit.meteo.score.array as ekm_array

    calls = []

    def spy(x, y, nan_policy="propagate"):
        calls.append(nan_policy)
        return ekm_array.crps(x, y, nan_policy=nan_policy)

    monkeypatch.setattr(crps_module, "ensemble_crps", spy)
    single_step_crps([[1.0], [2.0], [3.0]], [2.0])
    assert calls == ["propagate"]
    assert crps_module.ensemble_crps is spy
    assert not hasattr(crps_module, "crps_masked")
    assert not hasattr(crps_module, "crps_minmax")


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
    fc, obs = as_ensemble([[3.0, 1.0], [1.0, 2.0], [2.0, 5.0]], [2.0, 0.0])
    fc_before, obs_before = fc.copy(deep=True), obs.copy(deep=True)
    forecast_crps(fc, obs)
    xr.testing.assert_identical(fc, fc_before)
    xr.testing.assert_identical(obs, obs_before)


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
# forecast_crps: masking of points without forecast data
# ---------------------------------------------------------------------------


def _steps(x, y):
    """DataArrays for x (time, ensemble, point) and y (time, point)."""
    return (
        xr.DataArray(x, dims=("time", "ensemble", "point")),
        xr.DataArray(y, dims=("time", "point")),
    )


def test_forecast_crps_scores_each_step_against_its_observation():
    rng = np.random.default_rng(20)
    x = rng.uniform(0, 10, size=(3, 5, 6))
    y = rng.uniform(0, 10, size=(3, 6))
    expected = np.stack([reference_crps(x[i], y[i]) for i in range(3)])
    result = forecast_crps(*_steps(x, y), core_dims=["point"])
    assert result.values == pytest.approx(expected)


def test_forecast_crps_uses_first_step_and_member_as_mask():
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 10, size=(3, 5, 6))
    y = rng.uniform(0, 10, size=(3, 6))
    x[0, 0, 2] = np.nan
    result = forecast_crps(*_steps(x, y), core_dims=["point"]).values
    assert np.isnan(result[:, 2]).all()
    others = [0, 1, 3, 4, 5]
    assert not np.isnan(result[:, others]).any()
    for i in range(3):
        assert result[i, others] == pytest.approx(
            reference_crps(x[i][:, others], y[i][others])
        )


def test_forecast_crps_nan_outside_first_step_and_member_gives_nan_at_that_step():
    rng = np.random.default_rng(22)
    x = rng.uniform(0, 10, size=(2, 3, 4))
    y = rng.uniform(0, 10, size=(2, 4))
    x[1, 2, 1] = np.nan
    result = forecast_crps(*_steps(x, y), core_dims=["point"]).values
    # the point is not masked: its other step keeps a score
    assert np.isnan(result[1, 1])
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1, [0, 2, 3]]).all()


def test_forecast_crps_grid_mask_uses_first_step_and_member():
    rng = np.random.default_rng(23)
    x = rng.uniform(0, 10, size=(3, 4, 2, 3))  # time, ensemble, lat, lon
    y = rng.uniform(0, 10, size=(3, 2, 3))
    x[0, 0, 1, 2] = np.nan
    fc = xr.DataArray(x, dims=("time", "ensemble", "lat", "lon"))
    obs = xr.DataArray(y, dims=("time", "lat", "lon"))
    result = forecast_crps(fc, obs, core_dims=["lat", "lon"]).values
    assert np.isnan(result[:, 1, 2]).all()
    assert int(np.isnan(result).sum()) == 3


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


def _expected_by_reference(fc, obs, dims):
    """Reference CRPS with the ensemble first and `dims` after it."""
    x = fc.transpose("ensemble", *dims).values
    y = obs.broadcast_like(fc.isel(ensemble=0)).transpose(*dims).values
    return reference_crps(x, y)


@pytest.mark.parametrize(
    "spatial", [["station"], ["lat", "lon"], ["y", "x"]], ids=lambda d: "_".join(d)
)
def test_forecast_crps_any_spatial_dims_without_flattening(spatial):
    rng = np.random.default_rng(32)
    sizes = dict(zip(spatial, [4, 3][: len(spatial)]))
    fc = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, N_ENSEMBLE, *sizes.values())),
        dims=("time", "ensemble", *spatial),
    )
    obs = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, *sizes.values())),
        dims=("time", *spatial),
    )
    result = forecast_crps(fc, obs, core_dims=spatial)
    assert result.dims == ("time", *spatial)
    expected = _expected_by_reference(fc, obs, ["time", *spatial])
    assert result.values == pytest.approx(expected)


def test_forecast_crps_output_puts_spatial_dims_last_in_given_order():
    fc, obs = _forecast_and_obs()
    fc = fc.expand_dims(lat=[50.0, 49.0]).rename(station="lon")
    obs = obs.expand_dims(lat=[50.0, 49.0]).rename(station="lon")
    result = forecast_crps(
        fc.transpose("lon", "ensemble", "lat", "time"),
        obs.transpose("lon", "lat", "time"),
        core_dims=["lat", "lon"],
    )
    assert result.dims == ("time", "lat", "lon")


def test_forecast_crps_broadcasts_extra_forecast_dims():
    # a dimension only the forecast has (several forecast systems, say) is
    # kept, and every system is scored against the same observation
    fc, obs = _forecast_and_obs()
    systems = xr.concat([fc, fc + 1.0, fc * 2.0], dim="system")
    result = forecast_crps(systems, obs, core_dims=["station"])
    assert result.dims == ("system", "time", "station")
    for i in range(3):
        expected = _expected_forecast_crps(systems.isel(system=i), obs)
        assert result.isel(system=i).values == pytest.approx(expected)


def test_forecast_crps_broadcasts_observation_missing_a_dim():
    # an observation without a time dimension applies to every step
    fc, obs = _forecast_and_obs()
    single = obs.isel(time=0, drop=True)
    result = forecast_crps(fc, single, core_dims=["station"])
    expected = _expected_forecast_crps(fc, single.broadcast_like(obs))
    assert result.values == pytest.approx(expected)


@pytest.mark.parametrize("extra", ["ensemble", "system"])
def test_forecast_crps_observation_with_extra_dim_the_forecast_has_raises(extra):
    # the forecast has `system` and `ensemble` itself; the observation must not
    fc, obs = _forecast_and_obs()
    fc = fc.expand_dims(system=[0, 1])
    obs = obs.expand_dims({extra: [0, 1]})
    with pytest.raises(DimensionMismatchError, match=f"unexpected dimension '{extra}'"):
        forecast_crps(fc, obs, core_dims=["station"])


def test_forecast_crps_observation_with_extra_dim_missing_from_forecast_raises():
    fc, obs = _forecast_and_obs()
    obs = obs.expand_dims(level=[500, 850])
    with pytest.raises(DimensionMismatchError) as excinfo:
        forecast_crps(fc, obs, core_dims=["station"])
    message = str(excinfo.value)
    assert "unexpected dimension 'level'" in message
    assert "allowed: 'station' and optionally 'time'" in message


def test_forecast_crps_observation_missing_a_spatial_dim_raises():
    fc, obs = _forecast_and_obs()
    fc = fc.expand_dims(lat=[50.0, 49.0])
    obs = obs.expand_dims(lat=[50.0, 49.0]).isel(station=0, drop=True)
    with pytest.raises(DimensionMismatchError, match="no 'station' dimension"):
        forecast_crps(fc, obs, core_dims=["lat", "station"])


def test_forecast_crps_default_core_dims_apply_the_observation_contract():
    fc, obs = _forecast_and_obs()
    with pytest.raises(DimensionMismatchError, match="no 'station' dimension"):
        forecast_crps(fc, obs.isel(station=0, drop=True))
    with pytest.raises(DimensionMismatchError, match="unexpected dimension 'level'"):
        forecast_crps(fc, obs.expand_dims(level=[1]))


def test_forecast_crps_without_time_dimension():
    fc, obs = _forecast_and_obs()
    result = forecast_crps(fc.isel(time=1), obs.isel(time=1), core_dims=["station"])
    assert result.dims == ("station",)
    expected = reference_crps(
        fc.isel(time=1).transpose("ensemble", "station").values,
        obs.isel(time=1).values,
    )
    assert result.values == pytest.approx(expected)


def test_forecast_crps_custom_ensemble_dim():
    fc, obs = _forecast_and_obs()
    result = forecast_crps(
        fc.rename(ensemble="number"), obs, core_dims=["station"], ensemble_dim="number"
    )
    assert result.values == pytest.approx(_expected_forecast_crps(fc, obs))


def test_forecast_crps_aligns_on_time_coordinates():
    fc, obs = _forecast_and_obs()
    shifted = obs.isel(time=slice(1, None))  # observation misses the first step
    result = forecast_crps(fc, shifted, core_dims=["station"])
    np.testing.assert_array_equal(result.time, shifted.time)
    expected = _expected_forecast_crps(fc.isel(time=slice(1, None)), shifted)
    assert result.values == pytest.approx(expected)


def test_forecast_crps_dask_chunks_over_spatial_and_ensemble_dims():
    rng = np.random.default_rng(33)
    fc = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, N_ENSEMBLE, 4, 6)),
        dims=("time", "ensemble", "lat", "lon"),
    )
    obs = xr.DataArray(
        rng.uniform(0, 10, size=(N_STEPS, 4, 6)), dims=("time", "lat", "lon")
    )
    result = forecast_crps(
        fc.chunk({"time": 1, "ensemble": 2, "lat": 2, "lon": 3}),
        obs.chunk({"time": 1, "lat": 2, "lon": 3}),
        core_dims=["lat", "lon"],
    )
    assert result.chunks is not None
    expected = _expected_by_reference(fc, obs, ["time", "lat", "lon"])
    assert result.compute().values == pytest.approx(expected)


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


def test_persistence_crps_is_labelled_absolute_error():
    # labelled contract: the singleton persistence time is dropped, the field
    # broadcasts over the steps and cells are matched by coordinate
    time = pd.date_range("2020-01-02", periods=2, freq="D")
    reanalysis = xr.DataArray(
        [[1.0, 5.0, -2.0], [4.0, 0.0, 1.0]],
        dims=("time", "x"),
        coords={"time": time, "x": [1, 2, 3]},
    )
    persistence = xr.DataArray(
        [[-2.0, 3.0, 3.0]],
        dims=("time", "x"),
        coords={"time": time[:1] - pd.Timedelta(1, "D"), "x": [3, 2, 1]},
    )
    result = persistence_crps(reanalysis, persistence)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time", "x")
    np.testing.assert_array_equal(result.x, [1, 2, 3])
    np.testing.assert_array_equal(result.time, time)
    np.testing.assert_allclose(result.values, [[2.0, 2.0, 0.0], [1.0, 3.0, 3.0]])


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
# End-to-end: hyve-forecast-crps
# ---------------------------------------------------------------------------


class Inputs:
    """Synthetic inputs written to disk, kept in memory for expectations.

    ``spatial`` maps each spatial dimension to its coordinate values. The same
    random numbers are drawn for any layout with the same number of points, so
    a 2 x 2 grid holds the values of four stations in row-major order.
    """

    def __init__(
        self,
        root,
        dim="station",
        freq="D",
        forecast_dates=FORECAST_DATES,
        reanalysis_start=REANALYSIS_START,
        with_climatology=True,
        spatial=None,
        reversed_dims_on_disk=False,
    ):
        if spatial is None:
            spatial = {dim: STATIONS}
        self.spatial = {name: list(values) for name, values in spatial.items()}
        self.spatial_dims = list(self.spatial)
        shape = tuple(len(values) for values in self.spatial.values())
        n_points = int(np.prod(shape))
        self.reversed_dims_on_disk = reversed_dims_on_disk

        rng = np.random.default_rng(40)
        self.root = root
        self.freq = freq
        step = pd.Timedelta(1, "D") if freq == "D" else pd.Timedelta(freq)
        time = pd.date_range(reanalysis_start, periods=REANALYSIS_DAYS, freq=freq)
        self.reanalysis = xr.DataArray(
            rng.uniform(1, 10, size=(len(time), n_points)).reshape(len(time), *shape),
            dims=("time", *self.spatial_dims),
            coords={"time": time, **self.spatial},
            name="dis",
        )

        self.reforecasts = {}
        for date in forecast_dates:
            ftime = pd.date_range(date, periods=N_STEPS, freq=step)
            values = rng.uniform(1, 10, size=(N_ENSEMBLE, N_STEPS, n_points))
            self.reforecasts[pd.Timestamp(date)] = xr.DataArray(
                values.reshape(N_ENSEMBLE, N_STEPS, *shape),
                dims=("ensemble", "time", *self.spatial_dims),
                coords={
                    "ensemble": np.arange(N_ENSEMBLE),
                    "time": ftime,
                    **self.spatial,
                },
                name="dis",
            )

        self.climatology = None
        if with_climatology:
            doy = np.arange(1, 367)
            values = rng.uniform(1, 10, size=(N_ENSEMBLE, len(doy), 2, n_points))
            self.climatology = xr.DataArray(
                values.reshape(N_ENSEMBLE, len(doy), 2, *shape),
                dims=("ensemble", "doy", "issue_hour", *self.spatial_dims),
                coords={
                    "ensemble": np.arange(N_ENSEMBLE),
                    "doy": doy,
                    "issue_hour": [0, 12],
                    **self.spatial,
                },
                name="dis",
            )
        self.step = step

    def _on_disk(self, da):
        if self.reversed_dims_on_disk:
            return da.transpose(*reversed(da.dims))
        return da

    def write(self):
        self.root.mkdir(exist_ok=True)
        self._on_disk(self.reanalysis).to_dataset().to_netcdf(self.reanalysis_path)
        self.reforecast_dir.mkdir()
        for date, da in self.reforecasts.items():
            self._on_disk(da).to_dataset().to_netcdf(
                self.reforecast_dir / f"reforecast_{date:%Y%m%d%H}.nc"
            )
        if self.climatology is not None:
            self._on_disk(self.climatology).to_netcdf(self.climatology_path)
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

    @property
    def spatial_args(self):
        return ["--spatial_dims", *self.spatial_dims]

    # expectations: arrays shaped (time, *spatial_dims) --------------------------

    # Selected by label onto ``self.spatial``, so the order in which a
    # reforecast stores its spatial axes does not matter.

    def observed(self, date):
        obs = self.reanalysis.sel(time=self.reforecasts[date].time)
        return obs.sel(self.spatial).transpose("time", *self.spatial_dims).values

    def expected_refo(self, date):
        fc = self.reforecasts[date].sel(self.spatial)
        fc = fc.transpose("time", "ensemble", *self.spatial_dims)
        fc = fc.values
        obs = self.observed(date)
        return np.stack([reference_crps(fc[i], obs[i]) for i in range(N_STEPS)])

    def expected_pers(self, date, with_init=False):
        persistence_date = date if with_init else date - self.step
        persistence = self.reanalysis.sel(time=persistence_date).sel(self.spatial)
        persistence = persistence.transpose(*self.spatial_dims).values
        return np.abs(self.observed(date) - persistence[None, ...])

    def expected_clim(self, date):
        time = self.reforecasts[date].time
        clim = (
            self.climatology.sel(issue_hour=0)
            .sel(doy=time.dt.dayofyear.values)
            .sel(self.spatial)
            .transpose("doy", "ensemble", *self.spatial_dims)
            .values
        )
        obs = self.observed(date)
        return np.stack([reference_crps(clim[i], obs[i]) for i in range(N_STEPS)])


def run_cli(monkeypatch, inputs, *extra, climatology=True):
    argv = [
        "hyve-forecast-crps",
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
    forecast_crps_tool.main()
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


def test_cli_core_dim_names_a_single_spatial_dimension(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case", dim="id").write()
    out = run_cli(monkeypatch, inputs, "--core_dim", "id")
    date = pd.Timestamp(FORECAST_DATES[0])
    crps = open_crps(out / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
    assert crps.dims == ("time", "id")
    assert crps.values == pytest.approx(inputs.expected_refo(date))


def test_cli_spatial_dims_and_core_dim_together_is_an_error(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case").write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, "--spatial_dims", "station", "--core_dim", "id")
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Spatial dimensions
# ---------------------------------------------------------------------------

LAYOUTS = {
    "station": {"station": STATIONS},
    "lat_lon": {"lat": [50.0, 49.5, 49.0], "lon": [5.0, 5.5]},
    "y_x": {"y": [0, 1], "x": [100, 200, 300]},
}


@pytest.fixture(params=list(LAYOUTS))
def layout(request):
    return LAYOUTS[request.param]


def test_common_coords_per_dimension_keeps_grid_order():
    a = xr.Dataset(coords={"lat": [50.0, 49.5, 49.0], "lon": [5.0, 5.5, 6.0]})
    b = xr.Dataset(coords={"lat": [49.0, 49.5], "lon": [6.0, 5.5, 5.0, 4.5]})
    common = forecast_crps_tool.common_coords(a, b, ["lat", "lon"], "a", "b")
    assert common == {"lat": [49.5, 49.0], "lon": [5.0, 5.5, 6.0]}


def test_common_coords_stations_are_sorted():
    a = xr.Dataset(coords={"station": [4, 1, 3]})
    b = xr.Dataset(coords={"station": [3, 4, 5]})
    assert forecast_crps_tool.common_coords(a, b, ["station"], "a", "b") == {
        "station": [3, 4]
    }


def test_common_coords_float32_and_float64_are_the_same_kind():
    a = xr.Dataset(coords={"lat": np.array([50.0, 49.5], dtype="float32")})
    b = xr.Dataset(coords={"lat": np.array([49.5, 49.0], dtype="float64")})
    assert forecast_crps_tool.common_coords(a, b, ["lat"], "a", "b") == {"lat": [49.5]}


def test_common_coords_disjoint_grid_names_the_dimension():
    a = xr.Dataset(coords={"lat": [50.0, 49.5], "lon": [5.0]})
    b = xr.Dataset(coords={"lat": [50.0], "lon": [7.0, 8.0]})
    with pytest.raises(GridMismatchError) as excinfo:
        forecast_crps_tool.common_coords(a, b, ["lat", "lon"], "forecast", "reanalysis")
    message = str(excinfo.value)
    assert message.startswith("no lon values in common")
    assert "forecast" in message and "reanalysis" in message
    assert "1 lon values" in message and "2 lon values" in message


def test_common_coords_text_vs_float_grid_raises():
    a = xr.Dataset(coords={"x": [1.0, 2.0]})
    b = xr.Dataset(coords={"x": ["1", "2"]})
    with pytest.raises(GridMismatchError, match="x values of different kinds"):
        forecast_crps_tool.common_coords(a, b, ["x"], "a", "b")


def test_common_coords_missing_dimension_raises():
    a = xr.Dataset(coords={"lat": [50.0], "lon": [5.0]})
    b = xr.Dataset(coords={"station": [1]})
    with pytest.raises(DimensionMismatchError, match="b has no 'lat', 'lon' dimension"):
        forecast_crps_tool.common_coords(a, b, ["lat", "lon"], "a", "b")


def test_cli_scores_for_each_spatial_layout(monkeypatch, tmp_path, layout):
    inputs = Inputs(tmp_path / "case", spatial=layout).write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    dims = ("time", *inputs.spatial_dims)
    dates = list(inputs.reforecasts)
    expected = {
        "refo": {d: inputs.expected_refo(d) for d in dates},
        "pers": {d: inputs.expected_pers(d) for d in dates},
        "clim": {d: inputs.expected_clim(d) for d in dates},
    }
    for kind, by_date in expected.items():
        for date, values in by_date.items():
            crps = open_crps(out / f"crps_{kind}_{date:%Y%m%d}.nc")["crps"]
            assert crps.dims == dims
            for dim, coords in inputs.spatial.items():
                np.testing.assert_array_equal(crps[dim], coords)
            assert crps.values == pytest.approx(values)
        mean = open_crps(out / f"crps_{kind}.nc")["crps"]
        assert mean.dims == dims
        assert mean.values == pytest.approx(np.mean(list(by_date.values()), axis=0))
    refo = open_crps(out / "crps_refo.nc")["crps"]
    for benchmark in ("pers", "clim"):
        ref = open_crps(out / f"crps_{benchmark}.nc")["crps"]
        crpss = open_crps(out / f"crpss_{benchmark}.nc")["crpss"]
        assert crpss.dims == dims
        assert crpss.values == pytest.approx((1 - refo / ref).values)


def test_cli_with_init_for_each_spatial_layout(monkeypatch, tmp_path, layout):
    inputs = Inputs(tmp_path / "case", spatial=layout).write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args, "--with_init")
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_pers_{date:%Y%m%d}.nc")["crps"]
        assert crps.values == pytest.approx(inputs.expected_pers(date, True))


@pytest.mark.parametrize("name", ["lat_lon", "y_x"])
def test_cli_grid_scores_equal_station_scores(monkeypatch, tmp_path, name):
    grid = {"lat_lon": {"lat": [50.0, 49.5], "lon": [5.0, 5.5]}}.get(
        name, {"y": [0, 1], "x": [100, 200]}
    )
    stations = Inputs(tmp_path / "stations").write()
    gridded = Inputs(tmp_path / "grid", spatial=grid).write()
    out_s = run_cli(monkeypatch, stations)
    out_g = run_cli(monkeypatch, gridded, *gridded.spatial_args)
    for file in sorted(os.listdir(out_s)):
        s = open_crps(out_s / file)
        g = open_crps(out_g / file)
        var = list(s.data_vars)[0]
        assert g[var].values.reshape(s[var].shape) == pytest.approx(s[var].values)


def test_cli_dimension_order_on_disk_does_not_change_scores(monkeypatch, tmp_path):
    layout = LAYOUTS["lat_lon"]
    plain = Inputs(tmp_path / "plain", spatial=layout).write()
    flipped = Inputs(
        tmp_path / "flipped", spatial=layout, reversed_dims_on_disk=True
    ).write()
    out_p = run_cli(monkeypatch, plain, *plain.spatial_args)
    out_f = run_cli(monkeypatch, flipped, *flipped.spatial_args)
    assert sorted(os.listdir(out_p)) == sorted(os.listdir(out_f))
    for file in os.listdir(out_p):
        p = open_crps(out_p / file)
        f = open_crps(out_f / file)
        var = list(p.data_vars)[0]
        assert f[var].dims == p[var].dims == ("time", "lat", "lon")
        assert f[var].values == pytest.approx(p[var].values)


def test_cli_grid_scores_only_overlapping_cells(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    # the reanalysis covers a larger grid than the reforecast
    larger = inputs.reanalysis.reindex(
        lat=[51.0, 50.0, 49.5, 49.0], lon=[4.5, 5.0, 5.5], fill_value=3.0
    )
    inputs.reanalysis = larger
    inputs.write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    date = pd.Timestamp(FORECAST_DATES[0])
    crps = open_crps(out / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
    np.testing.assert_array_equal(crps.lat, [50.0, 49.5, 49.0])
    np.testing.assert_array_equal(crps.lon, [5.0, 5.5])
    inputs.reanalysis = larger.sel(lat=[50.0, 49.5, 49.0], lon=[5.0, 5.5])
    assert crps.values == pytest.approx(inputs.expected_refo(date))


# The reanalysis and the climatology keep the layout of ``LAYOUTS``; the
# reforecast of each listed date (index in FORECAST_DATES) stores its spatial
# axes in another order. The reference grid is the file of the latest date.
REORDERED_AXES = {
    "lat_lon_lat_reversed_in_one_file": ("lat_lon", {1: {"lat": [2, 1, 0]}}),
    "lat_lon_reference_file_reordered": (
        "lat_lon",
        {0: {"lon": [1, 0]}, 2: {"lat": [2, 1, 0], "lon": [1, 0]}},
    ),
    "y_x_x_permuted_and_y_reversed": (
        "y_x",
        {0: {"x": [2, 0, 1]}, 1: {"y": [1, 0]}},
    ),
    "y_x_reference_file_x_permuted": ("y_x", {2: {"x": [2, 0, 1]}}),
}


def _check_scores_by_label(out, inputs, dates):
    """Per-date and mean scores equal the expectations, compared by label."""
    dims = ("time", *inputs.spatial_dims)

    def by_label(crps):
        assert crps.dims == dims
        return crps.sel(inputs.spatial).values

    expected = {
        "refo": {d: inputs.expected_refo(d) for d in dates},
        "pers": {d: inputs.expected_pers(d) for d in dates},
        "clim": {d: inputs.expected_clim(d) for d in dates},
    }
    means = {}
    for kind, per_date in expected.items():
        for date, values in per_date.items():
            crps = open_crps(out / f"crps_{kind}_{date:%Y%m%d}.nc")["crps"]
            assert by_label(crps) == pytest.approx(values)
        mean = open_crps(out / f"crps_{kind}.nc")["crps"]
        means[kind] = by_label(mean)
        assert means[kind] == pytest.approx(np.mean(list(per_date.values()), axis=0))
    for benchmark in ("pers", "clim"):
        crpss = open_crps(out / f"crpss_{benchmark}.nc")["crpss"]
        expected_crpss = 1 - means["refo"] / means[benchmark]
        assert by_label(crpss) == pytest.approx(expected_crpss)


@pytest.mark.parametrize(
    "case", list(REORDERED_AXES.values()), ids=list(REORDERED_AXES)
)
def test_cli_scores_do_not_mix_cells_when_reforecast_axes_are_reordered(
    monkeypatch, tmp_path, case
):
    layout, reordered = case
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS[layout])
    dates = list(inputs.reforecasts)
    for index, order in reordered.items():
        date = dates[index]
        inputs.reforecasts[date] = inputs.reforecasts[date].isel(order)
    inputs.write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    _check_scores_by_label(out, inputs, dates)


def test_cli_reordered_station_axis_does_not_mix_stations(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case")
    date = list(inputs.reforecasts)[0]
    inputs.reforecasts[date] = inputs.reforecasts[date].isel(station=[3, 1, 0, 2])
    inputs.write()
    out = run_cli(monkeypatch, inputs)
    _check_scores_by_label(out, inputs, list(inputs.reforecasts))


@pytest.mark.parametrize(
    "name, dim, dropped",
    [("station", "station", 104), ("lat_lon", "lat", 49.0), ("y_x", "x", 300)],
)
def test_cli_later_file_missing_a_reference_point_exits_1(
    monkeypatch, tmp_path, capsys, name, dim, dropped
):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS[name])
    first = list(inputs.reforecasts)[0]  # the latest file is the reference
    da = inputs.reforecasts[first]
    inputs.reforecasts[first] = da.sel(
        {dim: [v for v in da[dim].values if v != dropped]}
    )
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert f"reforecast_{first:%Y%m%d}00.nc" in err
    assert repr(dim) in err
    assert "Traceback" not in err


def test_cli_disjoint_grid_exits_1(monkeypatch, tmp_path, capsys):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    inputs.reanalysis = inputs.reanalysis.assign_coords(lon=[15.0, 15.5])
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: no lon values in common")
    assert "reanalysis" in err
    assert "Traceback" not in err


def test_cli_spatial_dims_not_in_inputs_exits_1(monkeypatch, tmp_path, capsys):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"]).write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs)  # default spatial dimension is station
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "has no 'station' dimension" in err
    assert "--spatial_dims" in err


@pytest.mark.parametrize("extra", ["ensemble", "system", "level"])
def test_cli_reanalysis_with_extra_dim_exits_1(monkeypatch, tmp_path, capsys, extra):
    inputs = Inputs(tmp_path / "case")
    inputs.reanalysis = inputs.reanalysis.expand_dims({extra: [0]})
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: reanalysis 'dis' has unexpected dimension ")
    assert repr(extra) in err
    assert "--spatial_dims" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err
    assert not inputs.output.exists() or not list(inputs.output.iterdir())


@pytest.mark.parametrize("missing", ["station", "time"])
def test_cli_reanalysis_missing_a_dim_exits_1(monkeypatch, tmp_path, capsys, missing):
    inputs = Inputs(tmp_path / "case")
    inputs.reanalysis = inputs.reanalysis.isel({missing: 0}, drop=True)
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith(f"Error: reanalysis 'dis' has no {missing!r} dimension")
    assert "--spatial_dims" in err
    assert "Traceback" not in err


def test_cli_cells_without_forecast_data_are_not_reanalysis_gaps(monkeypatch, tmp_path):
    # a land/sea style mask: one cell is NaN in every reforecast and in the
    # reanalysis, and must not cause dates to be skipped
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    cell = {"lat": 49.5, "lon": 5.5}
    inputs.reanalysis = inputs.reanalysis.copy()
    inputs.reanalysis.loc[cell] = np.nan
    for date, da in inputs.reforecasts.items():
        da = da.copy()
        da.loc[cell] = np.nan
        inputs.reforecasts[date] = da
    inputs.write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    for date in inputs.reforecasts:
        crps = open_crps(out / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
        assert crps.sel(cell).isnull().all()
        assert int(crps.isnull().sum()) == N_STEPS
        expected = inputs.expected_refo(date)  # NaN at the masked cell
        assert crps.values == pytest.approx(expected, nan_ok=True)


def test_cli_grid_reanalysis_gap_at_scored_cell_skips_date(monkeypatch, tmp_path):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["y_x"])
    # 2020-03-03 is only inside the window of the last forecast date
    inputs.reanalysis = inputs.reanalysis.copy()
    inputs.reanalysis.loc[{"time": "2020-03-03", "y": 1, "x": 200}] = np.nan
    inputs.write()
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert not (out / "crps_refo_20200229.nc").exists()
    kept = [pd.Timestamp(d) for d in FORECAST_DATES[:2]]
    mean = open_crps(out / "crps_refo.nc")["crps"]
    expected = np.mean([inputs.expected_refo(d) for d in kept], axis=0)
    assert mean.values == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Float grids: the forecast grid is the reference
# ---------------------------------------------------------------------------

FLOAT_GRID = {"lat": [50.1, 49.6, 49.1], "lon": [5.2, 5.7]}


def _as_float32(da):
    return da.assign_coords(
        {dim: da[dim].astype("float32") for dim in FLOAT_GRID if dim in da.coords}
    )


def _float32_inputs(root, which="all"):
    """Inputs whose forecast (and climatology) grid is stored as float32.

    ``which="all"``: every reforecast file and the climatology; ``"later"``:
    only the earliest reforecast file, so the reference (latest) file stays
    float64. The reanalysis is always float64.
    """
    inputs = Inputs(root, spatial=FLOAT_GRID)
    dates = list(inputs.reforecasts)
    narrowed = dates if which == "all" else dates[:1]
    for date in narrowed:
        inputs.reforecasts[date] = _as_float32(inputs.reforecasts[date])
    if which == "all":
        inputs.climatology = _as_float32(inputs.climatology)
    return inputs.write()


@pytest.mark.parametrize("which", ["all", "later"])
def test_cli_float32_forecast_grid_gives_the_scores_of_a_float64_run(
    monkeypatch, tmp_path, which
):
    expected = Inputs(tmp_path / "float64", spatial=FLOAT_GRID).write()
    mixed = _float32_inputs(tmp_path / "float32", which)
    out_64 = run_cli(monkeypatch, expected, *expected.spatial_args)
    out_32 = run_cli(monkeypatch, mixed, *mixed.spatial_args)
    assert sorted(os.listdir(out_32)) == sorted(os.listdir(out_64))
    reference_dtype = np.dtype("float32" if which == "all" else "float64")
    for file in os.listdir(out_64):
        want = open_crps(out_64 / file)
        got = open_crps(out_32 / file)
        var = list(want.data_vars)[0]
        assert got[var].dims == want[var].dims
        np.testing.assert_allclose(got[var].values, want[var].values)
        for dim, values in FLOAT_GRID.items():
            # output coordinates are the forecast's (reference file) values
            assert got[dim].dtype == reference_dtype
            np.testing.assert_array_equal(got[dim], np.array(values, reference_dtype))
    for date in expected.reforecasts:
        crps = open_crps(out_32 / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
        assert crps.values == pytest.approx(expected.expected_refo(date))


def test_cli_coord_tolerance_override_matches_a_near_miss(
    monkeypatch, tmp_path, capsys
):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    inputs.reanalysis = inputs.reanalysis.assign_coords(lat=[50.001, 49.501, 49.001])
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    assert capsys.readouterr().err.startswith("Error: no lat values in common")
    out = run_cli(
        monkeypatch, inputs, *inputs.spatial_args, "--coord_tolerance", "0.01"
    )
    date = pd.Timestamp(FORECAST_DATES[0])
    crps = open_crps(out / f"crps_refo_{date:%Y%m%d}.nc")["crps"]
    np.testing.assert_array_equal(crps.lat, LAYOUTS["lat_lon"]["lat"])
    inputs.reanalysis = inputs.reanalysis.assign_coords(lat=LAYOUTS["lat_lon"]["lat"])
    assert crps.values == pytest.approx(inputs.expected_refo(date))


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "abc"])
def test_cli_invalid_coord_tolerance_exits_2(monkeypatch, tmp_path, capsys, value):
    inputs = Inputs(tmp_path / "case").write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, "--coord_tolerance", value)
    assert excinfo.value.code == 2
    assert "--coord_tolerance" in capsys.readouterr().err
    assert not inputs.output.exists()


def test_cli_ambiguous_coord_tolerance_exits_1(monkeypatch, tmp_path, capsys):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"]).write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args, "--coord_tolerance", "0.25")
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "ambiguous" in err
    assert "absolute tolerance 0.25" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err
    assert not inputs.output.exists()


def test_cli_duplicate_climatology_coordinate_exits_1(monkeypatch, tmp_path, capsys):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    inputs.climatology = inputs.climatology.assign_coords(lat=[50.0, 49.5, 49.5])
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: climatology has duplicate lat values")
    assert "climatology" in err and "Traceback" not in err


@pytest.mark.parametrize("descending", [False, True])
def test_cli_reference_file_is_the_latest_whatever_the_listing_order(
    monkeypatch, tmp_path, descending
):
    # The latest file (the reference) lacks a point that earlier files have:
    # the scored grid is the reference's, however the file system lists files.
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["lat_lon"])
    dates = list(inputs.reforecasts)
    inputs.reforecasts[dates[-1]] = inputs.reforecasts[dates[-1]].sel(lat=[50.0, 49.5])
    inputs.write()
    glob_glob = glob.glob
    monkeypatch.setattr(
        glob, "glob", lambda p: sorted(glob_glob(p), reverse=descending)
    )
    out = run_cli(monkeypatch, inputs, *inputs.spatial_args)
    crps = open_crps(out / "crps_refo.nc")["crps"]
    np.testing.assert_array_equal(crps.lat, [50.0, 49.5])


@pytest.mark.parametrize("where", ["reanalysis", "later_file", "reference_file"])
def test_cli_duplicate_integer_grid_value_exits_1(monkeypatch, tmp_path, capsys, where):
    inputs = Inputs(tmp_path / "case", spatial=LAYOUTS["y_x"])
    duplicate = {"x": [0, 0, 1, 2]}  # x = 100, 100, 200, 300
    if where == "reanalysis":
        inputs.reanalysis = inputs.reanalysis.isel(duplicate)
    else:
        dates = list(inputs.reforecasts)
        date = dates[0] if where == "later_file" else dates[-1]
        inputs.reforecasts[date] = inputs.reforecasts[date].isel(duplicate)
    inputs.write()
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, inputs, *inputs.spatial_args)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "duplicate x values" in err
    assert "[100]" in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
