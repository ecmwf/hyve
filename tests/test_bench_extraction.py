# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks/bench_extraction.py (synthetic mode only, no MARS)."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_extraction.py"

NT, NY, NX = 4, 7, 9


@pytest.fixture(scope="module")
def bench():
    # benchmarks/ is not a package, so import the script by file path.
    spec = importlib.util.spec_from_file_location("bench_extraction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stations(bench, da, n=12, seed=3):
    """Random stations; 0 and 1 share a cell, 2 is the corner cell (0, 0)."""
    df = bench.generate_stations(da, "latitude", "longitude", n, seed)
    df.loc[1, ["x_index", "y_index"]] = df.loc[0, ["x_index", "y_index"]].values
    df.loc[1, "x_coord"] = df.loc[0, "x_coord"] + 1e-6
    df.loc[1, "y_coord"] = df.loc[0, "y_coord"] - 1e-6
    df.loc[2, ["x_index", "y_index"]] = [0, 0]
    df.loc[2, "x_coord"] = da["latitude"].values[0]
    df.loc[2, "y_coord"] = da["longitude"].values[0]
    return df


@pytest.mark.parametrize("descending_lat", [False, True])
@pytest.mark.parametrize("descending_lon", [False, True])
@pytest.mark.parametrize("chunks", [None, 2])
def test_all_methods_return_known_cell_values(
    bench, descending_lat, descending_lon, chunks
):
    da = bench.make_synthetic(
        NT,
        NY,
        NX,
        descending_lat=descending_lat,
        descending_lon=descending_lon,
        chunks=chunks,
        dtype="float64",
        encode_position=True,
        nan_fraction=0.1,
        seed=1,
    )
    assert (da.chunks is not None) == (chunks is not None)
    raw = da.values  # plain numpy array, in the order as loaded
    df = _stations(bench, da)
    xi, yi = df["x_index"].values, df["y_index"].values
    expected = raw[:, xi, yi]  # (time, station), plain numpy fancy indexing
    has_value = ~np.isnan(expected)
    assert (~has_value).any() and has_value.any(), "test needs NaN and valid cells"
    # Valid values carry their own address in the loaded array.
    address = np.arange(NT)[:, None] * NY * NX + xi[None, :] * NX + yi[None, :]
    np.testing.assert_array_equal(expected[has_value], address[has_value])

    df_sorted = bench.remap_indices_to_sorted(df, da, "latitude", "longitude")
    frames = {
        "coord-mask": df,
        "coord-sel": df,
        "index-mask": df_sorted,
        "index-isel": df,
    }
    results = {}
    for method, fn in bench.EXTRACTORS.items():
        ds = fn(da, frames[method], "latitude", "longitude")
        np.testing.assert_array_equal(ds["station"].values, df["station_name"].values)
        got = ds[bench.VAR].transpose("time", "station").values
        np.testing.assert_array_equal(got, expected, err_msg=method)
        results[method] = ds
    assert bench.compare_results(results) == []


def test_stations_sharing_a_cell_get_same_values(bench):
    da = bench.make_synthetic(NT, NY, NX, dtype="float64", nan_fraction=0)
    df = _stations(bench, da)
    out = bench.extract_coord_sel(da, df, "latitude", "longitude")
    np.testing.assert_array_equal(
        out[bench.VAR].isel(station=0).values, out[bench.VAR].isel(station=1).values
    )


def test_legacy_index_needs_remap_on_descending_grid(bench):
    """Without the remap, the legacy index path would read other cells."""
    da = bench.make_synthetic(
        NT, NY, NX, descending_lat=True, dtype="float64", encode_position=True
    )
    df = _stations(bench, da)
    wrong = bench.extract_index_mask(da, df, "latitude", "longitude")
    right = bench.extract_index_isel(da, df, "latitude", "longitude")
    assert bench.compare_results({"index-isel": right, "index-mask": wrong})


def test_sort_inverse_matches_sortby(bench):
    coord = np.array([3.0, 1.0, 2.0, 0.5])
    inv = bench.sort_inverse(coord)
    sorted_da = xr.DataArray(coord, dims="x", coords={"x": coord}).sortby("x")
    # The cell at original index i sits at position inv[i] after sortby.
    np.testing.assert_array_equal(sorted_da["x"].values[inv], coord)


def test_compare_results_reports_differing_station(bench):
    ds = xr.Dataset(
        {bench.VAR: (("time", "station"), np.arange(6.0).reshape(2, 3))},
        coords={"station": ["a", "b", "c"]},
    )
    other = ds.copy(deep=True)
    other[bench.VAR][0, 1] = 99.0
    problems = bench.compare_results({"one": ds, "two": other})
    assert len(problems) == 1
    assert "two vs one" in problems[0] and "b" in problems[0]

    # NaN equals NaN.
    with_nan = ds.copy(deep=True)
    with_nan[bench.VAR][0, 0] = np.nan
    assert (
        bench.compare_results({"one": with_nan, "two": with_nan.copy(deep=True)}) == []
    )

    # Station order matters.
    assert bench.compare_results({"one": ds, "two": ds.isel(station=[1, 0, 2])})


def test_build_request(bench):
    efas = bench.build_request("efas", "20200101", "20200103")
    assert efas["hdate"] == ["20200101", "20200102", "20200103"]
    assert efas["stream"] == "efcl" and efas["param"] == "240023"
    glofas = bench.build_request("glofas", "20200101", "20200103")
    assert glofas["hdate"] == ["20200101", "20200102", "20200103"]
    assert glofas["date"] == "2022-10-01"
    assert glofas["stream"] == "wfcl" and glofas["param"] == "240024"
    with pytest.raises(bench.BenchmarkError):
        bench.build_request("nope", "20200101", "20200103")


def test_no_progress_bar_in_script(bench):
    assert not hasattr(bench, "ProgressBar")
    assert "ProgressBar" not in SCRIPT.read_text()


def _run_main(bench, out, *extra):
    return bench.main(
        [
            "--synthetic",
            "3,40,50",
            "--n-stations",
            "20",
            "--repeats",
            "2",
            "--output-dir",
            str(out),
            *extra,
        ]
    )


@pytest.mark.parametrize("extra", [[], ["--descending-lat", "--chunks", "1"]])
def test_main_synthetic_writes_outputs(bench, tmp_path, extra, capsys):
    out = tmp_path / "out"
    assert _run_main(bench, out, "--materialize-early", *extra) == 0
    printed = capsys.readouterr().out
    assert "identical results" in printed
    for name in bench.ALL_METHODS:
        assert name in printed

    manifest = json.loads((out / "manifest.json").read_text())
    for key in (
        "request",
        "synthetic",
        "dims",
        "dtype",
        "backend",
        "chunks",
        "materialize_early",
        "seed",
        "method_order",
        "versions",
        "git",
        "dask",
    ):
        assert key in manifest
    assert manifest["materialize_early"] is True
    assert manifest["backend"] == "numpy"  # materialized early
    assert manifest["lazy_before_materialize"] is bool(extra)
    assert manifest["seed"] == 42
    assert manifest["outputs_agree"] is True
    assert len(manifest["method_order"]["rounds"]) == 2
    assert set(manifest["versions"]) >= {"numpy", "xarray", "dask"}

    with open(out / "timings.csv") as f:
        rows = list(csv.DictReader(f))
    extract = [r for r in rows if r["phase"] == "extract"]
    assert len(extract) == 2 * len(bench.ALL_METHODS)  # warm-up is not recorded
    assert any(r["phase"] == "materialize_early" for r in rows)
    with open(out / "summary.csv") as f:
        assert len(list(csv.DictReader(f))) == len(bench.ALL_METHODS)


def test_main_dask_backend_in_manifest(bench, tmp_path):
    out = tmp_path / "out"
    assert _run_main(bench, out, "--chunks", "1") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["backend"] == "dask"
    assert manifest["chunks"]["time"]["max_size"] == 1
    assert manifest["dask"]["scheduler"]


def test_method_order_is_reproducible_for_a_seed(bench, tmp_path):
    orders = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert _run_main(bench, out, "--seed", "5") == 0
        orders.append(json.loads((out / "manifest.json").read_text())["method_order"])
    assert orders[0] == orders[1]


def test_mismatched_result_gives_exit_code_1(bench, tmp_path, monkeypatch, capsys):
    real = bench.EXTRACTORS["coord-sel"]

    def broken(da, df, x_dim, y_dim):
        ds = real(da, df, x_dim, y_dim)
        values = ds[bench.VAR].values.copy()
        values[..., 0] += 1.0  # corrupt the first station
        ds[bench.VAR] = ds[bench.VAR].copy(data=values)
        return ds

    monkeypatch.setitem(bench.EXTRACTORS, "coord-sel", broken)
    out = tmp_path / "out"
    assert _run_main(bench, out, "--methods", "coord-mask,coord-sel") == 1
    err = capsys.readouterr().err
    assert "MISMATCH" in err and "S0000" in err
    assert json.loads((out / "manifest.json").read_text())["outputs_agree"] is False


def test_missing_dim_gives_clear_error(bench, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_main(bench, tmp_path / "out", "--x-dim", "lat")
    assert exc.value.code == 2
    assert "'lat' not found" in capsys.readouterr().err


def test_synthetic_only_options_need_synthetic(bench, tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["--descending-lat", "--output-dir", str(tmp_path)])


def test_swapped_spatial_dims_still_agree(bench, tmp_path):
    out = tmp_path / "out"
    code = _run_main(bench, out, "--x-dim", "longitude", "--y-dim", "latitude")
    assert code == 0
