"""Benchmark: mask-based vs vectorized extraction on EFAS reanalysis data.

Compares four extraction strategies:
  coord-mask   – current hyve (create_mask_from_coords → apply_mask)
  coord-sel    – proposed xr.DataArray.sel(method="nearest")
  index-mask   – current hyve (create_mask_from_index → apply_mask)
  index-isel   – proposed xr.DataArray.isel with vectorized DataArray indices

Usage:
  python benchmarks/bench_extraction.py --start-date 20200101 --end-date 20200110
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr

from hyve.extraction import (
    apply_mask,
    construct_mask,
    create_mask_from_coords,
    create_mask_from_index,
)

logger = logging.getLogger(__name__)

ALL_METHODS = ["coord-mask", "coord-sel", "index-mask", "index-isel"]


def build_mars_request(start_date: str, end_date: str) -> dict:
    """Build MARS request dict with hdate list from date range."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    hdate_list = [d.strftime("%Y%m%d") for d in dates]

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
        "hdate": hdate_list,
    }


def load_data(request: dict) -> ekd.core.Base:
    """Load data from MARS via earthkit-data."""
    return ekd.from_source("mars", **request)


def to_xarray(source: ekd.core.Base) -> tuple[xr.DataArray, str, str, str]:
    """Convert earthkit source to dask-backed xarray, return da and dim names."""
    ds = source.to_xarray()
    var_names = [k for k in ds.data_vars if len(ds[k].dims) >= 3]
    if len(var_names) != 1:
        raise ValueError(f"Expected 1 main variable, found {len(var_names)}: {var_names}")
    var_name = var_names[0]
    da = ds[var_name]

    spatial_dims = [d for d in da.dims if d not in ("time", "step", "number", "forecast_reference_time")]
    if len(spatial_dims) < 2:
        raise ValueError(f"Cannot identify 2 spatial dims from {da.dims}")

    x_dim, y_dim = spatial_dims[0], spatial_dims[1]
    logger.info(f"Loaded variable={var_name}, dims={da.dims}, x={x_dim}, y={y_dim}")
    logger.info(f"Grid shape: {da[x_dim].shape[0]} x {da[y_dim].shape[0]}")
    return da, var_name, x_dim, y_dim


def generate_stations(
    da: xr.DataArray, x_dim: str, y_dim: str, n: int, seed: int
) -> pd.DataFrame:
    """Generate random station locations from the grid.

    Returns a DataFrame with columns: station_name, x_index, y_index,
    x_coord, y_coord (all derived from the same random grid cells).
    """
    rng = np.random.default_rng(seed)
    nx = da[x_dim].shape[0]
    ny = da[y_dim].shape[0]

    x_indices = rng.integers(0, nx, size=n)
    y_indices = rng.integers(0, ny, size=n)

    x_coords = da[x_dim].values[x_indices]
    y_coords = da[y_dim].values[y_indices]

    # Add small jitter to coords so sel(method="nearest") does real work
    x_coords = x_coords + rng.uniform(-1e-4, 1e-4, size=n)
    y_coords = y_coords + rng.uniform(-1e-4, 1e-4, size=n)

    return pd.DataFrame({
        "station_name": [f"S{i:04d}" for i in range(n)],
        "x_index": x_indices,
        "y_index": y_indices,
        "x_coord": x_coords,
        "y_coord": y_coords,
    })


# ---------------------------------------------------------------------------
# Extraction methods
# ---------------------------------------------------------------------------


def extract_coord_mask(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Current hyve approach: coord → mask → apply_ufunc."""
    shape = (da[x_dim].shape[0], da[y_dim].shape[0])
    mask, dup_idx = create_mask_from_coords(
        df.rename(columns={"x_coord": "x_coord", "y_coord": "y_coord"}),
        da[x_dim].values,
        da[y_dim].values,
        shape,
    )
    masked_da = apply_mask(da, mask, x_dim, y_dim)
    ds = xr.Dataset({"var": masked_da})
    ds = ds.isel(index=dup_idx)
    ds = ds.rename({"index": "station"})
    ds["station"] = df["station_name"].values
    return ds


def extract_coord_sel(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Proposed approach: xr.sel with method='nearest'."""
    target_x = xr.DataArray(df["x_coord"].values, dims="station")
    target_y = xr.DataArray(df["y_coord"].values, dims="station")
    result = da.sel({x_dim: target_x, y_dim: target_y}, method="nearest")
    result = result.compute()
    ds = xr.Dataset({"var": result})
    ds["station"] = df["station_name"].values
    return ds


def extract_index_mask(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Current hyve approach: index → mask → apply_ufunc."""
    shape = (da[x_dim].shape[0], da[y_dim].shape[0])
    mask, dup_idx = create_mask_from_index(df, shape)
    masked_da = apply_mask(da, mask, x_dim, y_dim)
    ds = xr.Dataset({"var": masked_da})
    ds = ds.isel(index=dup_idx)
    ds = ds.rename({"index": "station"})
    ds["station"] = df["station_name"].values
    return ds


def extract_index_isel(
    da: xr.DataArray, df: pd.DataFrame, x_dim: str, y_dim: str
) -> xr.Dataset:
    """Proposed approach: xr.isel with vectorized DataArray indices."""
    xi = xr.DataArray(df["x_index"].values, dims="station")
    yi = xr.DataArray(df["y_index"].values, dims="station")
    result = da.isel({x_dim: xi, y_dim: yi})
    result = result.compute()
    ds = xr.Dataset({"var": result})
    ds["station"] = df["station_name"].values
    return ds


EXTRACTORS = {
    "coord-mask": extract_coord_mask,
    "coord-sel": extract_coord_sel,
    "index-mask": extract_index_mask,
    "index-isel": extract_index_isel,
}


def timed(label: str, fn, *args, **kwargs):
    """Run fn, return (result, elapsed_seconds)."""
    logger.info(f"[{label}] starting...")
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info(f"[{label}] done in {elapsed:.3f}s")
    return result, elapsed


def save_extracted_data(ds: xr.Dataset, method: str, data_dir: Path) -> None:
    """Save extracted dataset as a CSV (station × time)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    var_name = [v for v in ds.data_vars if v != "station"][0]
    df = ds[var_name].to_dataframe().reset_index()
    df.to_csv(data_dir / f"{method}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mask-based vs vectorized extraction"
    )
    parser.add_argument("--start-date", default="20200101", help="First hdate (YYYYMMDD)")
    parser.add_argument("--end-date", default="20200110", help="Last hdate (YYYYMMDD)")
    parser.add_argument("--n-stations", type=int, default=500, help="Number of random stations")
    parser.add_argument(
        "--methods", default="all",
        help="Comma-separated methods or 'all': coord-mask,coord-sel,index-mask,index-isel",
    )
    parser.add_argument("--materialize-early", action="store_true",
                        help="Call .compute() on DataArray before extraction")
    parser.add_argument("--output", default="benchmarks/results.csv", help="Timing CSV path")
    parser.add_argument("--data-dir", default="benchmarks/data", help="Dir for per-method data CSVs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    methods = ALL_METHODS if args.methods == "all" else args.methods.split(",")
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}. Choose from {ALL_METHODS}")

    data_dir = Path(args.data_dir)
    timings = []

    # --- Phase 1: Load from MARS ---
    request = build_mars_request(args.start_date, args.end_date)
    logger.info(f"MARS request hdate count: {len(request['hdate'])}")
    source, t_load = timed("load", load_data, request)
    timings.append({"phase": "load", "method": "-", "seconds": t_load})

    # --- Phase 2: Convert to xarray ---
    (da, var_name, x_dim, y_dim), t_xr = timed("to_xarray", to_xarray, source)
    timings.append({"phase": "to_xarray", "method": "-", "seconds": t_xr})

    # --- Phase 3: Optionally materialize early ---
    if args.materialize_early:
        da, t_mat = timed("materialize_early", da.compute)
        timings.append({"phase": "materialize_early", "method": "-", "seconds": t_mat})

    # --- Phase 4: Generate stations ---
    df, t_stations = timed(
        "generate_stations", generate_stations, da, x_dim, y_dim, args.n_stations, args.seed
    )
    timings.append({"phase": "generate_stations", "method": "-", "seconds": t_stations})

    logger.info(f"Grid: {x_dim}={da[x_dim].shape[0]}, {y_dim}={da[y_dim].shape[0]}")
    logger.info(f"Stations: {len(df)}")

    # --- Phase 5: Run extraction methods ---
    for method in methods:
        extractor_fn = EXTRACTORS[method]
        ds, t_extract = timed(f"extract:{method}", extractor_fn, da, df, x_dim, y_dim)
        timings.append({"phase": "extract", "method": method, "seconds": t_extract})
        save_extracted_data(ds, method, data_dir)
        logger.info(f"Saved {method} data to {data_dir / f'{method}.csv'}")

    # --- Write timing CSV ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "method", "seconds"])
        writer.writeheader()
        writer.writerows(timings)

    logger.info(f"Timings written to {output_path}")

    # --- Summary ---
    print("\n=== Benchmark Results ===")
    print(f"{'Phase':<25} {'Method':<15} {'Time (s)':>10}")
    print("-" * 52)
    for row in timings:
        print(f"{row['phase']:<25} {row['method']:<15} {row['seconds']:>10.3f}")


if __name__ == "__main__":
    main()
