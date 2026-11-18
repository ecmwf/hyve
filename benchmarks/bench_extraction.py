# SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: mask-based (legacy) vs vectorized (new) station extraction.

Compares four extraction strategies on EFAS or GloFAS reanalysis data (or on
a generated synthetic array):

  coord-mask   - legacy hyve: sortby -> create_mask_from_coords -> apply_mask
  coord-sel    - new hyve: xr.DataArray.sel(method="nearest")
  index-mask   - legacy hyve: sortby -> create_mask_from_index -> apply_mask
  index-isel   - new hyve: xr.DataArray.isel with vectorized DataArray indices

Only the extraction step is timed. The legacy methods do what hyve on main
did: they first sort the grid with ``da.sortby([x_dim, y_dim])`` and that sort
is part of their timed run. The new methods work on the grid as loaded, like
``_process_regular`` in ``src/hyve/extraction.py``.

Stations are drawn on the grid as loaded (unsorted). The legacy index path
gets indices remapped to the sorted grid (in untimed station preparation), so
that every method targets the same physical cells. No progress bar is used
inside any timed section.

Each method gets one untimed warm-up run, then ``--repeats`` timed runs. The
order of the methods is shuffled in every round (``--seed``). After timing,
the outputs of all methods are compared; a mismatch gives exit code 1.

Usage:
  python benchmarks/bench_extraction.py --dataset efas \\
      --start-date 20200101 --end-date 20200110
  python benchmarks/bench_extraction.py --synthetic 24,500,1000 \\
      --descending-lat --chunks 8

See benchmarks/README.md for details.
"""

import argparse
import csv
import gc
import json
import logging
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

ALL_METHODS = ["coord-mask", "coord-sel", "index-mask", "index-isel"]
# Methods that use the station *indices*; legacy ones need them remapped to the
# sorted grid because they run on ``da.sortby(...)``.
LEGACY_INDEX_METHODS = {"index-mask"}
# (new method, legacy method) pairs for the speed-up column.
SPEEDUP_PAIRS = [("coord-sel", "coord-mask"), ("index-isel", "index-mask")]

DATASETS = ("efas", "glofas")
VAR = "var"  # variable name in every extractor output


class BenchmarkError(Exception):
    """A user-facing error (bad option, missing dimension, ...)."""


# ---------------------------------------------------------------------------
# Legacy mask-based extraction (copied from hyve before vectorized refactor)
# ---------------------------------------------------------------------------


def construct_mask(x_indices, y_indices, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[x_indices, y_indices] = True

    flat_indices = np.ravel_multi_index((x_indices, y_indices), shape)
    _, duplication_indexes = np.unique(flat_indices, return_inverse=True)
    return mask, duplication_indexes


def create_mask_from_index(df, shape):
    logger.info(f"Creating mask {shape} from index")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    x_indices = df["x_index"].values
    y_indices = df["y_index"].values
    if (
        np.any(x_indices < 0)
        or np.any(x_indices >= shape[0])
        or np.any(y_indices < 0)
        or np.any(y_indices >= shape[1])
    ):
        raise ValueError(
            f"Station indices out of grid bounds. Grid shape={shape}, "
            f"x_index range=({int(x_indices.min())},{int(x_indices.max())}), "
            f"y_index range=({int(y_indices.min())},{int(y_indices.max())})"
        )
    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def create_mask_from_coords(df, gridx, gridy, shape):
    logger.info(f"Creating mask {shape} from coordinates")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    station_x = df["x_coord"].values
    station_y = df["y_coord"].values

    x_distances = np.abs(station_x[:, np.newaxis] - gridx)
    x_indices = np.argmin(x_distances, axis=1)
    y_distances = np.abs(station_y[:, np.newaxis] - gridy)
    y_indices = np.argmin(y_distances, axis=1)

    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def mask_array_np(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[..., mask]


def apply_mask(
    da: xr.DataArray, mask: np.ndarray, coordx: str, coordy: str
) -> xr.DataArray:
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[(coordx, coordy), (coordx, coordy)],
        output_core_dims=[["index"]],
        output_dtypes=[da.dtype],
        exclude_dims={coordx, coordy},
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"index": int(mask.sum())},
            "allow_rechunk": True,
        },
    )
    # No progress bar here: it would add a fixed delay to every dask compute.
    return task.compute()


# ---------------------------------------------------------------------------
# Requests and data loading
# ---------------------------------------------------------------------------


def build_request(dataset: str, start_date: str, end_date: str) -> dict:
    """Build the MARS request for ``dataset`` over a daily date range.

    EFAS reanalysis uses an ``hdate`` list. For GloFAS the list of days goes
    in ``date``; this key is unconfirmed (see benchmarks/README.md).
    """
    if dataset not in DATASETS:
        raise BenchmarkError(f"Unknown dataset {dataset!r}. Choose from {DATASETS}")
    days = [
        d.strftime("%Y%m%d")
        for d in pd.date_range(start=start_date, end=end_date, freq="D")
    ]
    if not days:
        raise BenchmarkError(f"No days between {start_date} and {end_date}")

    if dataset == "efas":
        return {
            "class": "ce",
            "expver": "0001",
            "stream": "efcl",
            "date": "20230101",
            "model": "lisflood",
            "domain": "g",
            "origin": "ecmf",
            "step": 6,
            "type": "sfo",
            "levtype": "sfc",
            "param": "240023",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "hdate": days,
        }
    return {
        "class": "ce",
        "expver": "1",
        "stream": "wfcl",
        "model": "global_lisflood",
        "type": "sfo",
        "levtype": "sfc",
        "param": "240024",
        "time": "00:00",
        "date": days,
    }


def load_dataarray(
    request: dict, variable: str | None = None
) -> tuple[xr.DataArray, str]:
    """Retrieve from MARS via earthkit-data and return (DataArray, name)."""
    import earthkit.data as ekd  # imported here so synthetic mode needs no MARS

    ds = ekd.from_source("mars", **request).to_xarray()
    if variable is None:
        candidates = [k for k in ds.data_vars if len(ds[k].dims) >= 3]
        if len(candidates) != 1:
            raise BenchmarkError(
                f"Expected 1 main variable (>=3 dims), found {candidates}. "
                "Pick one with --variable."
            )
        variable = candidates[0]
    elif variable not in ds.data_vars:
        raise BenchmarkError(
            f"Variable {variable!r} not in dataset; found {list(ds.data_vars)}"
        )
    return ds[variable], variable


def check_spatial_dims(da: xr.DataArray, x_dim: str, y_dim: str) -> None:
    """Fail with a clear message if the spatial dimensions are not usable."""
    if x_dim == y_dim:
        raise BenchmarkError("--x-dim and --y-dim must be different")
    for name in (x_dim, y_dim):
        if name not in da.dims:
            raise BenchmarkError(
                f"Dimension {name!r} not found in the data (dims: {list(da.dims)}). "
                "Set --x-dim/--y-dim to the names of the spatial dimensions."
            )
        if name not in da.coords:
            raise BenchmarkError(
                f"Dimension {name!r} has no coordinate variable "
                f"(coords: {list(da.coords)})"
            )


def make_synthetic(
    nt: int,
    ny: int,
    nx: int,
    *,
    descending_lat: bool = False,
    descending_lon: bool = False,
    chunks: int | None = None,
    nan_fraction: float = 0.01,
    dtype: str = "float32",
    encode_position: bool = False,
    seed: int = 0,
) -> xr.DataArray:
    """Generate a (time, latitude, longitude) array on a 1/60 degree grid.

    ``ny`` is the number of latitudes, ``nx`` the number of longitudes.
    ``chunks`` is the chunk size along time (``None`` gives a numpy array;
    spatial dimensions are never chunked). With ``encode_position`` each value
    is its flat index in the array as generated (exact for float64), so a
    value identifies its cell; otherwise values are random. A fraction
    ``nan_fraction`` of the values is set to NaN.
    """
    rng = np.random.default_rng(seed)
    lat = 70.0 - np.arange(ny) / 60.0  # north to south, like EFAS
    lon = -20.0 + np.arange(nx) / 60.0
    if not descending_lat:
        lat = lat[::-1].copy()
    if descending_lon:
        lon = lon[::-1].copy()

    if encode_position:
        values = np.arange(nt * ny * nx, dtype=dtype).reshape(nt, ny, nx)
    else:
        values = rng.standard_normal((nt, ny, nx)).astype(dtype)
    if nan_fraction > 0:
        values[rng.random(values.shape) < nan_fraction] = np.nan

    da = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.date_range("2020-01-01", periods=nt, freq="6h"),
            "latitude": lat,
            "longitude": lon,
        },
        name="dis06",
    )
    if chunks is not None:
        da = da.chunk({"time": chunks})
    return da


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------


def _min_spacing(coord: np.ndarray) -> float:
    if coord.size < 2:
        return 0.0
    return float(np.min(np.abs(np.diff(coord.astype("float64")))))


def generate_stations(
    da: xr.DataArray, x_dim: str, y_dim: str, n: int, seed: int
) -> pd.DataFrame:
    """Draw random stations on the grid *as loaded* (unsorted).

    Returns a DataFrame with columns station_name, x_index, y_index, x_coord,
    y_coord. The indices are positions in ``da`` as given. The coordinates are
    those of the same cells, moved by up to a quarter of the smallest grid
    spacing, so that nearest-neighbour lookup does real work but still
    selects the same cell. Stations may share a cell.
    """
    rng = np.random.default_rng(seed)
    gx = da[x_dim].values
    gy = da[y_dim].values

    x_indices = rng.integers(0, gx.shape[0], size=n)
    y_indices = rng.integers(0, gy.shape[0], size=n)

    x_coords = gx[x_indices].astype("float64")
    y_coords = gy[y_indices].astype("float64")
    x_coords = x_coords + rng.uniform(-0.25, 0.25, size=n) * _min_spacing(gx)
    y_coords = y_coords + rng.uniform(-0.25, 0.25, size=n) * _min_spacing(gy)

    return pd.DataFrame(
        {
            "station_name": [f"S{i:04d}" for i in range(n)],
            "x_index": x_indices,
            "y_index": y_indices,
            "x_coord": x_coords,
            "y_coord": y_coords,
        }
    )


def sort_inverse(coord: np.ndarray) -> np.ndarray:
    """Return ``inv`` with ``inv[i]`` = position of original index ``i`` after
    ``sortby`` on ``coord``.

    ``sortby`` reorders an axis by ``order = np.lexsort((coord,))``, so the
    sorted axis is ``coord[order]``. The inverse permutation maps an original
    index to its position in the sorted axis.
    """
    order = np.lexsort((coord,))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return inv


def remap_indices_to_sorted(
    df: pd.DataFrame, da: xr.DataArray, x_dim: str, y_dim: str
) -> pd.DataFrame:
    """Copy of ``df`` whose indices point at the same cells in the sorted grid.

    Done in untimed station preparation. The ``sortby`` itself runs inside
    every timed legacy run.
    """
    out = df.copy()
    out["x_index"] = sort_inverse(da[x_dim].values)[df["x_index"].values]
    out["y_index"] = sort_inverse(da[y_dim].values)[df["y_index"].values]
    return out


# ---------------------------------------------------------------------------
# Extraction methods. All take (da, df, x_dim, y_dim) and return a Dataset
# with variable VAR and a "station" dimension, in the order of df.
# ---------------------------------------------------------------------------


def _legacy_finish(
    masked_da: xr.DataArray, dup_idx: np.ndarray, df: pd.DataFrame
) -> xr.Dataset:
    ds = xr.Dataset({VAR: masked_da})
    ds = ds.isel(index=dup_idx)
    ds = ds.rename({"index": "station"})
    ds["station"] = df["station_name"].values
    return ds


def extract_coord_mask(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Legacy hyve: sortby -> coords -> mask -> apply_ufunc."""
    da = da.sortby([x_dim, y_dim])
    shape = (da[x_dim].shape[0], da[y_dim].shape[0])
    mask, dup_idx = create_mask_from_coords(
        df, da[x_dim].values, da[y_dim].values, shape
    )
    return _legacy_finish(apply_mask(da, mask, x_dim, y_dim), dup_idx, df)


def extract_index_mask(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Legacy hyve: sortby -> index -> mask -> apply_ufunc.

    ``df`` must hold indices of the *sorted* grid (see
    ``remap_indices_to_sorted``).
    """
    da = da.sortby([x_dim, y_dim])
    shape = (da[x_dim].shape[0], da[y_dim].shape[0])
    mask, dup_idx = create_mask_from_index(df, shape)
    return _legacy_finish(apply_mask(da, mask, x_dim, y_dim), dup_idx, df)


def extract_coord_sel(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """New hyve: ``sel(method="nearest")`` with vectorized targets."""
    target_x = xr.DataArray(df["x_coord"].values, dims="station")
    target_y = xr.DataArray(df["y_coord"].values, dims="station")
    result = da.sel({x_dim: target_x, y_dim: target_y}, method="nearest")
    result = result.compute()
    ds = xr.Dataset({VAR: result})
    ds["station"] = df["station_name"].values
    return ds


def extract_index_isel(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """New hyve: ``isel`` with vectorized indices (indices of the loaded grid)."""
    shape = (da[x_dim].shape[0], da[y_dim].shape[0])
    x_indices = df["x_index"].values
    y_indices = df["y_index"].values
    if (
        np.any(x_indices < 0)
        or np.any(x_indices >= shape[0])
        or np.any(y_indices < 0)
        or np.any(y_indices >= shape[1])
    ):
        raise ValueError(f"Station indices out of grid bounds. Grid shape={shape}")
    xi = xr.DataArray(x_indices, dims="station")
    yi = xr.DataArray(y_indices, dims="station")
    result = da.isel({x_dim: xi, y_dim: yi})
    result = result.compute()
    ds = xr.Dataset({VAR: result})
    ds["station"] = df["station_name"].values
    return ds


EXTRACTORS: dict[str, Callable[..., xr.Dataset]] = {
    "coord-mask": extract_coord_mask,
    "coord-sel": extract_coord_sel,
    "index-mask": extract_index_mask,
    "index-isel": extract_index_isel,
}


# ---------------------------------------------------------------------------
# Timing and comparison
# ---------------------------------------------------------------------------


def run_benchmark(
    da: xr.DataArray,
    df: pd.DataFrame,
    x_dim: str,
    y_dim: str,
    methods: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Time ``methods``: one untimed warm-up each, then ``repeats`` rounds.

    The order of the methods is shuffled in the warm-up and in every round.
    Stations for the legacy index method are remapped to the sorted grid here,
    outside the timed sections.

    Returns a dict with ``timings`` (list of dicts: round, method, seconds),
    ``results`` (output of the last run of each method) and ``orders``
    (``warmup`` order and ``rounds`` orders).
    """
    if repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    frames = {m: df for m in methods}
    if LEGACY_INDEX_METHODS & set(methods):
        df_sorted = remap_indices_to_sorted(df, da, x_dim, y_dim)
        for m in LEGACY_INDEX_METHODS & set(methods):
            frames[m] = df_sorted

    rng = random.Random(seed)
    extractors = EXTRACTORS  # looked up at call time (tests patch it)
    results: dict[str, xr.Dataset] = {}

    def run_one(method: str) -> tuple[xr.Dataset, float]:
        gc.collect()
        t0 = time.perf_counter()
        out = extractors[method](da, frames[method], x_dim, y_dim)
        return out, time.perf_counter() - t0

    warmup_order = list(methods)
    rng.shuffle(warmup_order)
    for m in warmup_order:
        logger.info(f"[warm-up] {m}")
        results[m], _ = run_one(m)

    timings: list[dict[str, Any]] = []
    round_orders: list[list[str]] = []
    for rnd in range(repeats):
        order = list(methods)
        rng.shuffle(order)
        round_orders.append(order)
        for m in order:
            results[m], seconds = run_one(m)
            logger.info(f"[round {rnd}] {m}: {seconds:.3f}s")
            timings.append({"round": rnd, "method": m, "seconds": seconds})

    return {
        "timings": timings,
        "results": results,
        "orders": {"warmup": warmup_order, "rounds": round_orders},
    }


def summarize(timings: list[dict[str, Any]], methods: list[str]) -> list[dict]:
    """Per-method min/median/max of the timed runs, plus speed-up vs legacy."""
    rows = []
    for m in methods:
        secs = [t["seconds"] for t in timings if t["method"] == m]
        rows.append(
            {
                "method": m,
                "runs": len(secs),
                "min": min(secs),
                "median": statistics.median(secs),
                "max": max(secs),
            }
        )
    by_method = {r["method"]: r for r in rows}
    for new, old in SPEEDUP_PAIRS:
        if new in by_method and old in by_method:
            by_method[new]["speedup_vs"] = old
            by_method[new]["speedup_median"] = (
                by_method[old]["median"] / by_method[new]["median"]
            )
    return rows


def compare_results(results: dict[str, xr.Dataset], max_listed: int = 20) -> list[str]:
    """Compare every method's output with the first one.

    Checks the station order, the shape and the values (NaN equal to NaN).
    Returns a list of problem descriptions; empty if all methods agree.
    """
    names = list(results)
    if len(names) < 2:
        return []
    ref_name = names[0]
    ref = results[ref_name]
    ref_stations = ref["station"].values
    ref_vals = ref[VAR].transpose("station", ...)
    problems: list[str] = []
    for name in names[1:]:
        ds = results[name]
        label = f"{name} vs {ref_name}"
        stations = ds["station"].values
        if stations.shape != ref_stations.shape or not np.array_equal(
            stations, ref_stations
        ):
            problems.append(f"{label}: station order or count differs")
            continue
        if set(ds[VAR].dims) != set(ref_vals.dims):
            problems.append(f"{label}: dims differ ({ds[VAR].dims} vs {ref_vals.dims})")
            continue
        vals = ds[VAR].transpose(*ref_vals.dims)
        if vals.shape != ref_vals.shape:
            problems.append(
                f"{label}: shape differs ({vals.shape} vs {ref_vals.shape})"
            )
            continue
        a, b = vals.values, ref_vals.values
        equal = (a == b) | (pd.isna(a) & pd.isna(b))
        bad = ~equal.reshape(equal.shape[0], -1).all(axis=1)
        if bad.any():
            bad_names = [str(s) for s in ref_stations[bad][:max_listed]]
            more = "" if bad.sum() <= max_listed else f" (+{bad.sum() - max_listed})"
            problems.append(
                f"{label}: values differ for {int(bad.sum())} station(s): "
                f"{', '.join(bad_names)}{more}"
            )
    return problems


# ---------------------------------------------------------------------------
# Manifest and output files
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, Any]:
    cwd = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"revision": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "xarray", "dask", "earthkit-data", "hyve"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _dask_info(da: xr.DataArray) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import dask
        import dask.system
        from dask.base import get_scheduler

        info["config_scheduler"] = dask.config.get("scheduler", None)
        info["config_num_workers"] = dask.config.get("num_workers", None)
        info["cpu_count"] = dask.system.CPU_COUNT
        sched = get_scheduler(collections=[da.data] if da.chunks else None)
        info["scheduler"] = (
            getattr(sched, "__module__", "?")
            + "."
            + getattr(sched, "__name__", repr(sched))
        )
    except Exception as exc:  # noqa: BLE001 - manifest info is best effort
        info["error"] = repr(exc)
    return info


def _chunk_info(da: xr.DataArray) -> dict[str, Any] | None:
    if not da.chunks:
        return None
    return {
        str(dim): {"n_chunks": len(c), "max_size": max(c), "min_size": min(c)}
        for dim, c in zip(da.dims, da.chunks)
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    request: dict | None,
    synthetic: dict | None,
    da: xr.DataArray,
    lazy_before_materialize: bool,
    methods: list[str],
    orders: dict[str, Any],
    n_stations: int,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset": None if synthetic else args.dataset,
        "request": request,
        "synthetic": synthetic,
        "variable": da.name,
        "x_dim": args.x_dim,
        "y_dim": args.y_dim,
        "dims": {str(k): int(v) for k, v in da.sizes.items()},
        "dtype": str(da.dtype),
        "backend": "dask" if da.chunks else "numpy",
        "lazy_before_materialize": lazy_before_materialize,
        "chunks": _chunk_info(da),
        "materialize_early": args.materialize_early,
        "n_stations": n_stations,
        "repeats": args.repeats,
        "seed": args.seed,
        "methods": methods,
        "method_order": orders,
        "versions": _package_versions(),
        "git": _git_info(),
        "dask": _dask_info(da),
        "platform": platform.platform(),
    }


def write_outputs(
    output_dir: Path,
    setup_timings: list[dict[str, Any]],
    run: dict[str, Any],
    summary: list[dict],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "timings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "round", "method", "seconds"])
        writer.writeheader()
        for row in setup_timings:
            writer.writerow({"phase": row["phase"], "round": "", "method": "-", **row})
        for row in run["timings"]:
            writer.writerow({"phase": "extract", **row})
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "runs",
                "min",
                "median",
                "max",
                "speedup_vs",
                "speedup_median",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")


def print_summary(summary: list[dict]) -> None:
    print("\n=== Extraction timings (seconds, timed runs only) ===")
    print(f"{'Method':<12} {'Runs':>4} {'Min':>9} {'Median':>9} {'Max':>9}  Speed-up")
    print("-" * 62)
    for r in summary:
        extra = ""
        if "speedup_median" in r:
            extra = f"{r['speedup_median']:.2f}x vs {r['speedup_vs']} (median)"
        print(
            f"{r['method']:<12} {r['runs']:>4} {r['min']:>9.4f} "
            f"{r['median']:>9.4f} {r['max']:>9.4f}  {extra}"
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_synthetic(spec: str) -> tuple[int, int, int]:
    try:
        nt, ny, nx = (int(v) for v in spec.split(","))
    except ValueError:
        raise BenchmarkError(f"--synthetic expects NT,NY,NX; got {spec!r}") from None
    if min(nt, ny, nx) < 1:
        raise BenchmarkError("--synthetic sizes must be positive")
    return nt, ny, nx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark mask-based (legacy) vs vectorized extraction"
    )
    p.add_argument("--dataset", choices=DATASETS, default="efas")
    p.add_argument("--start-date", default="20200101", help="First day (YYYYMMDD)")
    p.add_argument("--end-date", default="20200110", help="Last day (YYYYMMDD)")
    p.add_argument("--variable", help="Variable to use (default: the only 3D+ one)")
    p.add_argument(
        "--x-dim",
        default="latitude",
        help="First spatial dimension (default: %(default)s)",
    )
    p.add_argument(
        "--y-dim",
        default="longitude",
        help="Second spatial dimension (default: %(default)s)",
    )
    p.add_argument("--n-stations", type=int, default=500)
    p.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods or 'all': " + ",".join(ALL_METHODS),
    )
    p.add_argument(
        "--repeats", type=int, default=5, help="Timed runs per method (default: 5)"
    )
    p.add_argument(
        "--materialize-early",
        action="store_true",
        help="Load all data into memory (.compute()) before timing",
    )
    p.add_argument(
        "--synthetic",
        metavar="NT,NY,NX",
        help="Use a generated (time, latitude, longitude) array instead of MARS",
    )
    p.add_argument(
        "--descending-lat",
        action="store_true",
        help="Synthetic: store latitude north to south (like EFAS)",
    )
    p.add_argument(
        "--descending-lon",
        action="store_true",
        help="Synthetic: store longitude in descending order",
    )
    p.add_argument(
        "--chunks",
        type=int,
        help="Synthetic: dask chunk size along time (default: numpy array)",
    )
    p.add_argument("--output-dir", default="bench-output", help="Directory for results")
    p.add_argument("--seed", type=int, default=42, help="Seed for stations and order")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return _run(args)
    except BenchmarkError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error exits


def _run(args: argparse.Namespace) -> int:
    methods = ALL_METHODS if args.methods == "all" else args.methods.split(",")
    for m in methods:
        if m not in ALL_METHODS:
            raise BenchmarkError(f"Unknown method: {m}. Choose from {ALL_METHODS}")
    methods = list(dict.fromkeys(methods))
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    if args.synthetic is None and (
        args.descending_lat or args.descending_lon or args.chunks is not None
    ):
        raise BenchmarkError(
            "--descending-lat, --descending-lon and --chunks need --synthetic"
        )

    setup_timings: list[dict[str, Any]] = []
    request = None
    synthetic = None
    if args.synthetic is not None:
        nt, ny, nx = parse_synthetic(args.synthetic)
        synthetic = {
            "nt": nt,
            "ny": ny,
            "nx": nx,
            "descending_lat": args.descending_lat,
            "descending_lon": args.descending_lon,
            "chunks": args.chunks,
        }
        da = make_synthetic(
            nt,
            ny,
            nx,
            descending_lat=args.descending_lat,
            descending_lon=args.descending_lon,
            chunks=args.chunks,
            seed=args.seed,
        )
    else:
        request = build_request(args.dataset, args.start_date, args.end_date)
        logger.info(f"MARS request: {request}")
        t0 = time.perf_counter()
        da, _ = load_dataarray(request, args.variable)
        setup_timings.append(
            {"phase": "load_to_xarray", "seconds": time.perf_counter() - t0}
        )

    check_spatial_dims(da, args.x_dim, args.y_dim)
    # Not just dask: earthkit-data can also return lazy, non-dask arrays.
    lazy = not da.variable._in_memory
    if args.materialize_early:
        t0 = time.perf_counter()
        da = da.compute()
        setup_timings.append(
            {"phase": "materialize_early", "seconds": time.perf_counter() - t0}
        )

    logger.info(
        f"Data: dims={dict(da.sizes)}, dtype={da.dtype}, "
        f"lazy={not da.variable._in_memory}"
    )
    df = generate_stations(da, args.x_dim, args.y_dim, args.n_stations, args.seed)

    run = run_benchmark(
        da, df, args.x_dim, args.y_dim, methods, args.repeats, args.seed
    )
    summary = summarize(run["timings"], methods)
    manifest = build_manifest(
        args=args,
        request=request,
        synthetic=synthetic,
        da=da,
        lazy_before_materialize=lazy,
        methods=methods,
        orders=run["orders"],
        n_stations=len(df),
    )
    problems = compare_results(run["results"])
    manifest["outputs_agree"] = not problems
    manifest["mismatches"] = problems

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, setup_timings, run, summary, manifest)
    print_summary(summary)
    print(f"\nResults written to {output_dir}/")

    if problems:
        print("\nMISMATCH: methods returned different results:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1
    if len(methods) > 1:
        print(f"All {len(methods)} methods returned identical results.")
    else:
        print("Only one method run: nothing to compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
