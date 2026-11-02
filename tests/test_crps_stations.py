import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools import crps_stations
from hyve.tools.crps_stations import StationMismatchError, common_stations

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
        "hyve-crps-stations",
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
    crps_stations.main()
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
