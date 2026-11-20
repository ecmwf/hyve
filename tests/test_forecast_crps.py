# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools import forecast_crps
from hyve.tools.forecast_crps import (
    CoordinateMismatchError,
    DimensionMismatchError,
    GridMismatchError,
    StationMismatchError,
    common_coords,
    common_stations,
    match_coords,
    relabel,
)

N_ENSEMBLE = 3
N_STEPS = 3
FORECAST_DATES = ["2020-01-02", "2020-01-03"]


def _write_reanalysis(path, stations):
    time = pd.date_range("2020-01-01", periods=8, freq="D")
    rng = np.random.default_rng(0)
    dis = rng.uniform(1, 10, size=(len(time), len(stations)))
    xr.Dataset(
        {"dis": (("time", "station"), dis)},
        coords={"time": time, "station": stations},
    ).to_netcdf(path)


def _write_reforecasts(folder, stations):
    folder.mkdir()
    rng = np.random.default_rng(1)
    for date in FORECAST_DATES:
        time = pd.date_range(date, periods=N_STEPS, freq="D")
        dis = rng.uniform(1, 10, size=(N_ENSEMBLE, N_STEPS, len(stations)))
        xr.Dataset(
            {"dis": (("ensemble", "time", "station"), dis)},
            coords={
                "ensemble": np.arange(N_ENSEMBLE),
                "time": time,
                "station": stations,
            },
        ).to_netcdf(folder / f"reforecast_{date}.nc")


def _write_climatology(path, stations):
    doy = np.arange(1, 367)
    rng = np.random.default_rng(2)
    values = rng.uniform(1, 10, size=(N_ENSEMBLE, len(doy), 1, len(stations)))
    xr.DataArray(
        values,
        dims=("ensemble", "doy", "issue_hour", "station"),
        coords={
            "ensemble": np.arange(N_ENSEMBLE),
            "doy": doy,
            "issue_hour": [0],
            "station": stations,
        },
        name="dis",
    ).to_netcdf(path)


def _run_cli(monkeypatch, tmp_path, reanalysis, reforecast, climatology=None):
    out = tmp_path / "out"
    argv = [
        "hyve-forecast-crps",
        "--reanalysis",
        str(reanalysis),
        "--reforecast",
        str(reforecast),
        "--output",
        str(out),
    ]
    if climatology is not None:
        argv += ["--climatology", str(climatology)]
    monkeypatch.setattr("sys.argv", argv)
    forecast_crps.main()
    return out


def _build(tmp_path, reanalysis_ids, reforecast_ids, climatology_ids=None):
    reanalysis = tmp_path / "reanalysis.nc"
    reforecast = tmp_path / "reforecast"
    _write_reanalysis(reanalysis, reanalysis_ids)
    _write_reforecasts(reforecast, reforecast_ids)
    climatology = None
    if climatology_ids is not None:
        climatology = tmp_path / "climatology.nc"
        _write_climatology(climatology, climatology_ids)
    return reanalysis, reforecast, climatology


def test_common_stations_returns_overlap():
    shared = common_stations(
        np.array([1, 2, 3, 4]), np.array([3, 4, 5]), "reanalysis", "reforecast"
    )
    assert shared == [3, 4]


def test_common_stations_int_vs_text_raises_with_details():
    ints = np.array([1, 2, 3], dtype="int64")
    texts = np.array(["1", "2", "3", "4"], dtype=object)
    with pytest.raises(StationMismatchError) as excinfo:
        common_stations(ints, texts, "reforecast", "reanalysis")
    message = str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert "\n" not in message
    assert "reforecast" in message and "reanalysis" in message
    assert "int64" in message and "object" in message
    assert "3 stations" in message and "4 stations" in message
    assert "[1, 2, 3]" in message and "['1', '2', '3', '4']" in message


def test_common_stations_limits_samples_to_five():
    ints = np.arange(20)
    texts = np.array([f"s{i}" for i in range(20)])
    with pytest.raises(ValueError) as excinfo:
        common_stations(ints, texts, "a", "b")
    message = str(excinfo.value)
    assert "[0, 1, 2, 3, 4]" in message
    assert "['s0', 's1', 's2', 's3', 's4']" in message
    assert "s5" not in message


def test_common_stations_disjoint_raises():
    with pytest.raises(ValueError, match="no stations in common") as excinfo:
        common_stations(np.array([1, 2]), np.array([3, 4]), "reanalysis", "reforecast")
    message = str(excinfo.value)
    assert "reanalysis" in message and "reforecast" in message
    assert "int64" in message


def test_common_stations_never_converts_ids():
    with pytest.raises(ValueError):
        common_stations(np.array([1, 2]), np.array(["1", "2"]), "a", "b")
    with pytest.raises(ValueError):
        common_stations(np.array([1, 2]), np.array([b"1", b"2"]), "a", "b")


def test_common_stations_large_int64_vs_uint64_do_not_falsely_match():
    big = 2**53
    with pytest.raises(StationMismatchError, match="no stations in common"):
        common_stations(
            np.array([big + 1], dtype="int64"),
            np.array([big], dtype="uint64"),
            "a",
            "b",
        )


def test_common_stations_int64_vs_uint64_equal_values_match_exactly():
    big = 2**53
    shared = common_stations(
        np.array([big + 1, 7, big], dtype="int64"),
        np.array([big + 1, big, 9], dtype="uint64"),
        "a",
        "b",
    )
    assert shared == [big, big + 1]
    assert all(type(station) is int for station in shared)


def test_common_stations_bytes_vs_text_raises_without_decoding():
    with pytest.raises(StationMismatchError, match="different kinds") as excinfo:
        common_stations(np.array([b"a", b"b"]), np.array(["a", "b"]), "a", "b")
    assert "bytes" in str(excinfo.value) and "text" in str(excinfo.value)
    with pytest.raises(StationMismatchError, match="different kinds"):
        common_stations(
            np.array([b"a"], dtype=object), np.array(["a"], dtype=object), "a", "b"
        )


def test_common_stations_bytes_match_bytes():
    shared = common_stations(np.array([b"a", b"b"]), np.array([b"b", b"c"]), "a", "b")
    assert shared == [b"b"]


def test_cli_int_reforecast_vs_text_reanalysis_exits_1(monkeypatch, tmp_path, capsys):
    reanalysis, reforecast, _ = _build(
        tmp_path,
        reanalysis_ids=np.array([str(i) for i in range(1, 6)], dtype=object),
        reforecast_ids=np.arange(1, 6),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, reanalysis, reforecast)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: station IDs of different kinds")
    assert "int64" in err and "<U1" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


def test_cli_disjoint_stations_exits_1(monkeypatch, tmp_path, capsys):
    reanalysis, reforecast, _ = _build(
        tmp_path,
        reanalysis_ids=np.arange(1, 6),
        reforecast_ids=np.arange(10, 15),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, reanalysis, reforecast)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "no stations in common" in err
    assert len(err.strip().splitlines()) == 1


def test_cli_text_climatology_vs_int_reanalysis_exits_1(monkeypatch, tmp_path, capsys):
    reanalysis, reforecast, climatology = _build(
        tmp_path,
        reanalysis_ids=np.arange(1, 6),
        reforecast_ids=np.arange(1, 6),
        climatology_ids=np.array([str(i) for i in range(1, 6)], dtype=object),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, reanalysis, reforecast, climatology)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "climatology" in err and "reanalysis" in err
    assert "int64" in err and "<U1" in err
    assert len(err.strip().splitlines()) == 1


def test_cli_disjoint_climatology_exits_1_without_output(monkeypatch, tmp_path, capsys):
    reanalysis, reforecast, climatology = _build(
        tmp_path,
        reanalysis_ids=np.arange(1, 6),
        reforecast_ids=np.arange(1, 6),
        climatology_ids=np.arange(100, 105),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, reanalysis, reforecast, climatology)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: no stations in common")
    assert "climatology" in err and "reanalysis" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("with_climatology", [False, True])
def test_cli_partial_overlap_completes(monkeypatch, tmp_path, capsys, with_climatology):
    reanalysis, reforecast, climatology = _build(
        tmp_path,
        reanalysis_ids=np.array([1, 2, 3, 4]),
        reforecast_ids=np.array([3, 4, 5, 6]),
        climatology_ids=np.array([3, 4, 5, 6]) if with_climatology else None,
    )
    out = _run_cli(
        monkeypatch,
        tmp_path,
        reanalysis,
        reforecast,
        climatology if with_climatology else None,
    )
    assert capsys.readouterr().err == ""
    expected = ["crps_refo.nc", "crps_pers.nc", "crpss_pers.nc"]
    if with_climatology:
        expected += ["crps_clim.nc", "crpss_clim.nc"]
    for name in expected:
        assert (out / name).exists(), name
    crps = xr.open_dataarray(out / "crps_refo.nc")
    assert sorted(crps.station.values.tolist()) == [3, 4]
    assert not crps.isnull().any()


def test_deprecated_crps_stations_main_still_runs(monkeypatch, tmp_path, capsys):
    from hyve.tools import crps_stations

    reanalysis, reforecast, _ = _build(
        tmp_path,
        reanalysis_ids=np.array([1, 2, 3, 4]),
        reforecast_ids=np.array([3, 4, 5, 6]),
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hyve-crps-stations",
            "--reanalysis",
            str(reanalysis),
            "--reforecast",
            str(reforecast),
            "--output",
            str(out),
        ],
    )
    crps_stations.main()
    assert "deprecated" in capsys.readouterr().err
    crps = xr.open_dataarray(out / "crps_refo.nc")
    assert sorted(crps.station.values.tolist()) == [3, 4]


def test_deprecated_crps_stations_reexports_station_mismatch_error():
    from hyve.tools import crps_stations
    from hyve.tools.forecast_crps import errors

    assert crps_stations.StationMismatchError is errors.StationMismatchError


# ---------------------------------------------------------------------------
# Float grids: precision-aware matching, forecast grid as the reference
# ---------------------------------------------------------------------------


def _axis(dim, values, dtype="float64"):
    return xr.Dataset(coords={dim: np.array(values, dtype=dtype)})


def test_error_hierarchy():
    for error in (StationMismatchError, GridMismatchError, DimensionMismatchError):
        assert issubclass(error, CoordinateMismatchError)
    assert issubclass(CoordinateMismatchError, ValueError)
    assert not issubclass(GridMismatchError, StationMismatchError)


@pytest.mark.parametrize(
    "dim, values",
    [
        ("lat", [49.1, 49.2]),
        ("lat", [-0.1, 0.0, 0.1]),
        ("lon", [359.7, 359.8, 359.9]),
        ("x", [5.0e6 + 0.1, 5.0e6 + 1000.1, 5.0e6 + 2000.1]),
    ],
)
@pytest.mark.parametrize("forecast_dtype", ["float32", "float64"])
def test_float32_and_float64_grids_match_and_return_forecast_values(
    dim, values, forecast_dtype
):
    other_dtype = "float64" if forecast_dtype == "float32" else "float32"
    forecast = _axis(dim, values, forecast_dtype)
    observation = _axis(dim, values, other_dtype)
    shared = common_coords(forecast, observation, [dim], "forecast", "reanalysis")
    assert shared == {dim: forecast[dim].values.tolist()}
    matches = match_coords(forecast, observation, [dim], "forecast", "reanalysis")
    relabelled = relabel(observation, matches, forecast)
    assert relabelled[dim].dtype == np.dtype(forecast_dtype)
    np.testing.assert_array_equal(relabelled[dim], forecast[dim])


def test_matching_the_reproduction_from_the_review():
    a = _axis("lat", [49.1, 49.2], "float32")
    b = _axis("lat", [49.1, 49.2])
    assert common_coords(a, b, ["lat"], "forecast", "reanalysis")["lat"] == [
        float(np.float32(49.1)),
        float(np.float32(49.2)),
    ]


def test_relabel_moves_data_with_its_point_and_uses_forecast_order():
    forecast = _axis("lat", [49.3, 49.2, 49.1], "float32")
    observation = xr.DataArray(
        [10.0, 20.0, 30.0, 40.0],
        dims="lat",
        coords={"lat": [49.0, 49.1, 49.2, 49.3]},
    )
    matches = match_coords(forecast, observation, ["lat"], "forecast", "reanalysis")
    relabelled = relabel(observation, matches, forecast)
    np.testing.assert_array_equal(relabelled.lat, forecast.lat)
    assert relabelled.values.tolist() == [40.0, 30.0, 20.0]


def test_float_match_keeps_partial_overlap():
    a = _axis("lat", [49.1, 49.2, 49.3], "float32")
    b = _axis("lat", [49.2, 49.3, 49.4])
    assert common_coords(a, b, ["lat"], "a", "b") == {"lat": a.lat.values[1:].tolist()}


def test_near_miss_needs_the_coord_tolerance_override():
    a = _axis("lat", [49.0, 49.5])
    b = _axis("lat", [49.0005, 49.5005])
    with pytest.raises(GridMismatchError, match="no lat values in common"):
        common_coords(a, b, ["lat"], "forecast", "reanalysis")
    shared = common_coords(a, b, ["lat"], "forecast", "reanalysis", 1e-3)
    assert shared == {"lat": [49.0, 49.5]}


def test_coord_tolerance_zero_is_exact_matching():
    a = _axis("lat", [49.1, 49.2], "float32")
    b = _axis("lat", [49.1, 49.2])
    with pytest.raises(GridMismatchError, match="no lat values in common"):
        common_coords(a, b, ["lat"], "a", "b", 0.0)
    assert common_coords(a, a, ["lat"], "a", "a", 0.0)["lat"] == a.lat.values.tolist()


def test_tolerance_at_least_half_the_spacing_is_ambiguous():
    a = _axis("lat", [49.0, 49.5, 50.0])
    with pytest.raises(GridMismatchError, match="ambiguous") as excinfo:
        common_coords(a, a, ["lat"], "forecast", "reanalysis", 0.25)
    message = str(excinfo.value)
    assert "forecast" in message and "reanalysis" in message
    assert "lat" in message and "absolute tolerance 0.25" in message
    assert common_coords(a, a, ["lat"], "a", "b", 0.2)["lat"] == [49.0, 49.5, 50.0]


def test_ambiguous_default_tolerance_for_a_grid_finer_than_precision():
    # spacing 1e-6 of a float32 axis near 100 is below its precision
    a = _axis("lat", [100.0, 100.000001], "float32")
    b = _axis("lat", [100.0, 100.000001])
    with pytest.raises(GridMismatchError):
        common_coords(a, b, ["lat"], "a", "b")


def test_two_values_snapping_to_one_point_raise():
    a = _axis("lat", [0.0, 1.0])
    b = _axis("lat", [0.5])
    with pytest.raises(GridMismatchError, match="snap to the same") as excinfo:
        common_coords(a, b, ["lat"], "forecast", "reanalysis", 0.5)
    message = str(excinfo.value)
    assert "forecast" in message and "reanalysis" in message
    assert "absolute tolerance 0.5" in message


@pytest.mark.parametrize("side", ["first", "second"])
def test_duplicate_coordinates_raise(side):
    good = _axis("lat", [49.0, 49.5])
    duplicated = _axis("lat", [49.0, 49.5, 49.5])
    a, b = (duplicated, good) if side == "first" else (good, duplicated)
    with pytest.raises(GridMismatchError, match="duplicate lat values"):
        common_coords(a, b, ["lat"], "forecast", "reanalysis")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["first", "second"])
def test_non_finite_coordinates_raise(bad, side):
    good = _axis("lat", [49.0, 49.5])
    broken = _axis("lat", [49.0, bad])
    a, b = (broken, good) if side == "first" else (good, broken)
    with pytest.raises(GridMismatchError, match="NaN or infinite lat values"):
        common_coords(a, b, ["lat"], "forecast", "reanalysis")


def test_singleton_axis_matches_without_spacing_guard():
    a = _axis("lat", [49.1], "float32")
    b = _axis("lat", [49.1])
    assert common_coords(a, b, ["lat"], "a", "b") == {"lat": [float(np.float32(49.1))]}
    assert common_coords(a, b, ["lat"], "a", "b", 5.0)["lat"] == a.lat.values.tolist()


def test_descending_forecast_latitude_keeps_forecast_order():
    a = _axis("lat", [50.1, 49.6, 49.1], "float32")
    b = _axis("lat", [49.1, 49.6, 50.1, 50.6])
    assert common_coords(a, b, ["lat"], "a", "b")["lat"] == a.lat.values.tolist()
    ascending = common_coords(b, a, ["lat"], "b", "a")["lat"]
    assert ascending == [49.1, 49.6, 50.1]


def test_unordered_axes_are_matched_by_value():
    a = _axis("x", [300.1, 100.1, 200.1], "float32")
    b = _axis("x", [200.1, 300.1, 100.1])
    matches = match_coords(a, b, ["x"], "a", "b")
    assert matches["x"][0].tolist() == [0, 1, 2]
    assert matches["x"][1].tolist() == [1, 2, 0]


def test_integer_and_text_station_ids_stay_exact():
    assert common_stations(np.array([3, 1, 2]), np.array([2, 3]), "a", "b") == [2, 3]
    texts = common_stations(np.array(["b", "a"]), np.array(["a", "c"]), "a", "b")
    assert texts == ["a"]
    with pytest.raises(StationMismatchError, match="no stations in common"):
        common_stations(np.array([1, 2]), np.array([3]), "a", "b", "station", 100.0)
    with pytest.raises(StationMismatchError, match="no stations in common"):
        common_stations(np.array(["a"]), np.array(["b"]), "a", "b")


def test_integer_grid_axes_stay_exact_and_ignore_the_tolerance():
    a = xr.Dataset(coords={"y": [0, 1, 2]})
    b = xr.Dataset(coords={"y": [1, 2, 3]})
    assert common_coords(a, b, ["y"], "a", "b", 10.0) == {"y": [1, 2]}


def test_integer_vs_float_grid_is_a_different_kind_error():
    a = xr.Dataset(coords={"y": [0, 1, 2]})
    b = _axis("y", [0.0, 1.0, 2.0])
    with pytest.raises(GridMismatchError, match="y values of different kinds"):
        common_coords(a, b, ["y"], "a", "b")


def test_error_class_follows_the_dimension():
    ids = xr.Dataset(coords={"station": ["a"]})
    other = xr.Dataset(coords={"station": [1]})
    with pytest.raises(StationMismatchError, match="different kinds"):
        common_coords(ids, other, ["station"], "a", "b")
    floats = xr.Dataset(coords={"station": [1.0, 2.0]})
    with pytest.raises(StationMismatchError, match="duplicate station values"):
        common_coords(
            xr.Dataset(coords={"station": [1.0, 1.0]}), floats, ["station"], "a", "b"
        )
    assert common_coords(
        xr.Dataset(coords={"station": [2.0, 1.0]}), floats, ["station"], "a", "b"
    ) == {"station": [1.0, 2.0]}


def test_missing_dimension_raises_dimension_mismatch_error():
    a = xr.Dataset(coords={"lat": [50.0]})
    b = xr.Dataset(coords={"station": [1]})
    with pytest.raises(DimensionMismatchError, match="has no 'lat' dimension"):
        common_coords(a, b, ["lat"], "a", "b")
    with pytest.raises(DimensionMismatchError, match="b has no 'lat' dimension"):
        match_coords(a, b, ["lat"], "a", "b")


# ---------------------------------------------------------------------------
# Duplicate coordinate values, whatever their kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["forecast", "observation"])
def test_duplicate_integer_grid_values_raise(side):
    good = xr.Dataset(coords={"x": [100, 200, 300]})
    duplicated = xr.Dataset(coords={"x": [100, 100, 200, 300]})
    a, b = (duplicated, good) if side == "forecast" else (good, duplicated)
    with pytest.raises(GridMismatchError) as excinfo:
        common_coords(a, b, ["x"], "forecast", "reanalysis")
    message = str(excinfo.value)
    assert "has duplicate x values" in message and "[100]" in message
    name = "forecast" if side == "forecast" else "reanalysis"
    assert name in message and "\n" not in message
    with pytest.raises(GridMismatchError, match="duplicate x values"):
        match_coords(a, b, ["x"], "forecast", "reanalysis")


def test_duplicate_text_station_ids_raise():
    ids = np.array(["a", "b", "b", "c"])
    with pytest.raises(StationMismatchError, match="duplicate station values") as exc:
        common_stations(ids, np.array(["a", "b"]), "forecast", "reanalysis")
    assert "forecast" in str(exc.value) and "['b']" in str(exc.value)
    with pytest.raises(StationMismatchError, match="duplicate station values"):
        common_stations(np.array(["a", "b"]), ids, "forecast", "reanalysis")


@pytest.mark.parametrize("dtype", ["int64", "uint64", "bytes", "object"])
def test_duplicate_station_ids_raise_for_every_kind(dtype):
    if dtype == "bytes":
        ids = np.array([b"1", b"2", b"2"])
    else:
        ids = np.array([1, 2, 2], dtype=dtype)
    other = ids[:2]
    with pytest.raises(StationMismatchError, match="duplicate station values"):
        common_stations(ids, other, "a", "b")
    with pytest.raises(StationMismatchError, match="duplicate station values"):
        common_stations(other, ids, "a", "b")


def test_unique_integer_ids_are_still_matched():
    assert common_stations(np.array([2, 1]), np.array([1, 2, 3]), "a", "b") == [1, 2]
