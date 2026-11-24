# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked, lazy reading in hyve-forecast-crps.

`--chunk_size` reads every input in chunks over the spatial dimensions, so
memory follows the chunk size rather than the size of the domain. The scores
must not depend on it.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools import forecast_crps
from hyve.tools.forecast_crps import cli, scores
from hyve.tools.forecast_crps.chunks import chunk_spatially, spatial_chunks

N_ENSEMBLE = 4
N_STEPS = 6
N_POINTS = 23
FORECAST_STARTS = ["2020-02-26", "2020-02-27", "2020-02-28"]


def _write_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    points = np.arange(N_POINTS)
    time = pd.date_range("2020-02-24", periods=10 * 4, freq="6h")
    xr.Dataset(
        {"dis": (("time", "point"), rng.uniform(1, 10, (len(time), N_POINTS)))},
        coords={"time": time, "point": points},
    ).to_netcdf(tmp_path / "reanalysis.nc")

    folder = tmp_path / "forecasts"
    folder.mkdir()
    for start in FORECAST_STARTS:
        valid = pd.date_range(
            pd.Timestamp(start) + pd.Timedelta("6h"), periods=N_STEPS, freq="6h"
        )
        dis = rng.uniform(1, 10, (N_ENSEMBLE, N_STEPS, N_POINTS)).astype("float32")
        xr.Dataset(
            {"dis": (("ensemble", "time", "point"), dis)},
            coords={"ensemble": np.arange(N_ENSEMBLE), "time": valid, "point": points},
        ).to_netcdf(folder / f"{pd.Timestamp(start):%Y%m%d%H}.nc")

    doy = np.arange(1, 367)
    xr.DataArray(
        rng.uniform(1, 10, (N_ENSEMBLE, len(doy), 1, N_POINTS)).astype("float32"),
        dims=("ensemble", "doy", "issue_hour", "point"),
        coords={
            "ensemble": np.arange(N_ENSEMBLE),
            "doy": doy,
            "issue_hour": [0],
            "point": points,
        },
        name="dis",
    ).to_netcdf(tmp_path / "climatology.nc")
    return tmp_path / "reanalysis.nc", folder, tmp_path / "climatology.nc"


def _run(monkeypatch, tmp_path, out_name, *extra):
    reanalysis, forecasts, climatology = _write_inputs(tmp_path)
    out = tmp_path / out_name
    argv = [
        "hyve-forecast-crps",
        "--reanalysis",
        str(reanalysis),
        "--reforecast",
        str(forecasts),
        "--climatology",
        str(climatology),
        "--spatial_dims",
        "point",
        "--output",
        str(out),
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    forecast_crps.main()
    return out


# ---------------------------------------------------------------------------
# chunk layout
# ---------------------------------------------------------------------------


def test_single_spatial_dim_is_chunked_by_points():
    ds = xr.Dataset({"dis": (("time", "point"), np.zeros((3, 10)))})
    assert spatial_chunks(ds, ["point"], 4) == {"point": 4}


def test_chunk_larger_than_domain_is_one_chunk():
    ds = xr.Dataset({"dis": (("time", "point"), np.zeros((3, 10)))})
    assert spatial_chunks(ds, ["point"], 100) == {"point": 10}


def test_two_spatial_dims_split_the_first_into_rows():
    ds = xr.Dataset({"dis": (("lat", "lon"), np.zeros((10, 50)))})
    # 120 points per chunk = 2 rows of 50
    assert spatial_chunks(ds, ["lat", "lon"], 120) == {"lat": 2, "lon": 50}


def test_two_spatial_dims_never_go_below_one_row():
    ds = xr.Dataset({"dis": (("lat", "lon"), np.zeros((10, 50)))})
    assert spatial_chunks(ds, ["lat", "lon"], 7) == {"lat": 1, "lon": 50}


def test_extra_chunks_apply_only_to_present_dims():
    ds = xr.Dataset({"dis": (("doy", "point"), np.zeros((366, 10)))})
    assert spatial_chunks(ds, ["point"], 4, extra={"doy": 1, "time": 1}) == {
        "point": 4,
        "doy": 1,
    }


def test_no_chunk_size_leaves_the_data_unchunked():
    ds = xr.Dataset({"dis": (("time", "point"), np.zeros((3, 10)))})
    assert chunk_spatially(ds, ["point"], None) is ds


def test_chunk_spatially_gives_dask_arrays_in_spatial_chunks(tmp_path):
    path = tmp_path / "x.nc"
    xr.Dataset({"dis": (("time", "point"), np.zeros((3, 10)))}).to_netcdf(path)
    with xr.open_dataset(path) as ds:
        chunked = chunk_spatially(ds, ["point"], 4)
        assert isinstance(chunked["dis"].data, dask.array.Array)
        assert chunked["dis"].chunks == ((3,), (4, 4, 2))


def test_climatology_without_time_is_chunked_by_doy_and_point(tmp_path):
    _, _, climatology = _write_inputs(tmp_path)
    da = cli.open_climatology(str(climatology), ["point"], 5)
    assert "time" not in da.dims
    assert isinstance(da.data, dask.array.Array)
    chunks = dict(zip(da.dims, da.chunks))
    assert set(chunks["doy"]) == {1}
    assert max(chunks["point"]) == 5


# ---------------------------------------------------------------------------
# scores do not depend on the chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", ["1", "5", "1000"])
def test_chunked_scores_equal_unchunked_scores(monkeypatch, tmp_path, chunk_size):
    plain = _run(monkeypatch, tmp_path / "a", "out")
    chunked = _run(monkeypatch, tmp_path / "b", "out", "--chunk_size", chunk_size)
    names = sorted(p.name for p in plain.iterdir())
    assert names == sorted(p.name for p in chunked.iterdir())
    assert "crps_clim_20200226.nc" in names
    for name in names:
        expected = xr.open_dataarray(plain / name)
        actual = xr.open_dataarray(chunked / name)
        xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_compute_score_reads_each_forecast_file_in_chunks(monkeypatch, tmp_path):
    reanalysis, forecasts, _ = _write_inputs(tmp_path)
    seen = []
    real = scores.forecast_crps

    def spy(x, y, **kwargs):
        seen.append(x.chunks)
        return real(x, y, **kwargs)

    monkeypatch.setattr(scores, "forecast_crps", spy)
    da = xr.open_dataset(reanalysis)["dis"]
    out = tmp_path / "out"
    out.mkdir()
    scores.compute_score(str(out), str(forecasts), da, None, ["point"], chunk_size=7)
    assert len(seen) == len(FORECAST_STARTS)
    for chunks in seen:
        assert chunks is not None
        assert chunks[-1] == (7, 7, 7, 2)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-3", "1.5", "x"])
def test_chunk_size_must_be_a_positive_integer(monkeypatch, tmp_path, value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, "out", "--chunk_size", value)
    assert excinfo.value.code == 2
    assert "--chunk_size" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_threads_must_be_a_positive_integer(monkeypatch, tmp_path, value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, "out", "--threads", value)
    assert excinfo.value.code == 2
    assert "--threads" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, scheduler, workers",
    [
        ([], "threads", None),
        (["--threads", "3"], "threads", 3),
        (["--scheduler", "synchronous"], "synchronous", None),
    ],
)
def test_scheduler_and_threads_reach_dask(
    monkeypatch, tmp_path, extra, scheduler, workers
):
    seen = {}

    def fake_compute_score(*args, **kwargs):
        seen["scheduler"] = dask.config.get("scheduler")
        seen["num_workers"] = dask.config.get("num_workers", None)
        seen["chunk_size"] = kwargs.get("chunk_size")

    monkeypatch.setattr(cli, "compute_score", fake_compute_score)
    _run(monkeypatch, tmp_path, "out", "--chunk_size", "4", *extra)
    assert seen == {"scheduler": scheduler, "num_workers": workers, "chunk_size": 4}


def test_help_texts_describe_the_options(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hyve-forecast-crps", "--help"])
    with pytest.raises(SystemExit):
        forecast_crps.main()
    out = " ".join(capsys.readouterr().out.split())
    assert "points per chunk" in out
    assert "dask scheduler" in out
    assert "climatology dataset file" in out
