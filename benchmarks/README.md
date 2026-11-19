<!--
SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)

SPDX-License-Identifier: Apache-2.0
-->
# Extraction benchmark

`bench_extraction.py` compares the legacy mask-based station extraction with
the vectorized one used by `hyve.extraction` now, on EFAS or GloFAS data
(retrieved from MARS) or on a generated synthetic array.

| Method       | What it does                                                        |
|--------------|---------------------------------------------------------------------|
| `coord-mask` | legacy: `sortby`, nearest cell from coordinates, boolean mask       |
| `index-mask` | legacy: `sortby`, cell from indices, boolean mask                   |
| `coord-sel`  | new: `da.sel(..., method="nearest")` with vectorized targets        |
| `index-isel` | new: `da.isel(...)` with vectorized indices                         |

## What is measured, and how it is kept fair

- Only the extraction step is timed. Loading from MARS, `--materialize-early`
  and station preparation are outside the timed sections.
- The legacy methods do what hyve on `main` did: they first run
  `da.sortby([x_dim, y_dim])`, and **that sort is part of their timed run**.
- Stations are drawn on the grid as loaded (unsorted). For `index-mask`, the
  station indices are remapped to the sorted grid (inverse of the sort
  permutation on each axis) in untimed station preparation, so all four
  methods read the same physical cells.
- No progress bar runs inside any timed section.
- Every method gets one untimed warm-up run, then `--repeats` timed runs
  (default 5). The order of the methods is shuffled in each round, seeded with
  `--seed`. The summary shows min, median and max per method.
- After timing, the output of every method is compared (same station order,
  equal values, NaN equal to NaN). A mismatch prints the differing stations
  and the script exits with code 1.

## Running on HPC

Use an environment with `hyve` dependencies and MARS access (the earthkit-data
version pinned in `pyproject.toml`). One EFAS field has about 13 M grid
points, so a long date range is large when loaded into memory: scale up with
care.

```bash
# EFAS reanalysis, 1 month, 1000 stations
python benchmarks/bench_extraction.py --dataset efas \
    --start-date 20200101 --end-date 20200131 --n-stations 1000

# GloFAS reanalysis
python benchmarks/bench_extraction.py --dataset glofas \
    --start-date 20200101 --end-date 20200131 --n-stations 1000

# Same, with all data loaded into memory before timing
python benchmarks/bench_extraction.py --dataset efas \
    --start-date 20200101 --end-date 20200131 --n-stations 1000 \
    --materialize-early --output-dir bench-output/efas-materialized
```

The requests are built from `--start-date/--end-date` (daily):

- `efas`: class=ce, expver=0001, stream=efcl, date=20230101, model=lisflood,
  domain=g, origin=ecmf, step=6, type=sfo, levtype=sfc, param=240023,
  time=00/06/12/18, `hdate` = the days.
- `glofas`: class=ce, expver=1, stream=wfcl, date=2022-10-01,
  model=global_lisflood, type=sfo, levtype=sfc, param=240024, time=00:00,
  `hdate` = the days.

Both use `hdate` for the list of days, with a fixed `date`. This was confirmed
by the first EFAS and GloFAS runs on HPC.

Dimension names: `--x-dim` and `--y-dim` set the two spatial dimensions and
default to `latitude` and `longitude` (x is the first axis of the station
index, as in the hyve grid config). The dimension names that earthkit-data
gives for real MARS output are not verified, so the defaults are a guess: if
the script says a dimension is missing, pass the right names. If the
dataset has several variables, pick one with `--variable` (by default the
only variable with 3 or more dimensions is used).

## `--materialize-early`

Without it, the DataArray stays lazy (dask-backed or an earthkit-data lazy
array, reading from the earthkit source) and every run reads data through the caches of earthkit-data, the
file system and the OS. With `--materialize-early` the script calls
`.compute()` on the whole DataArray first, so all data is in memory before
timing and the timings show the extraction only.

**Runs without `--materialize-early` measure warm caches, not cold I/O from
MARS or disk**: the warm-up run has already fetched the data, so the timed
runs read it from cache. They are not a measure of cold retrieval.

## Synthetic mode (no MARS needed)

```bash
python benchmarks/bench_extraction.py --synthetic 24,500,1000 \
    --descending-lat --chunks 8 --repeats 3
```

`--synthetic NT,NY,NX` generates a `(time, latitude, longitude)` array with
NY latitudes and NX longitudes on a 1/60° grid and about 1 % NaN.
`--descending-lat` stores latitude north to south (like EFAS),
`--descending-lon` does the same for longitude, and `--chunks N` makes a
dask array with chunks of N along time (without it the array is numpy).

## Output

Everything goes to `--output-dir` (default `bench-output/`, listed in
`.gitignore`; files of an earlier run in the same directory are overwritten):

- `timings.csv`: raw timing of every timed run (phase, round, method,
  seconds), plus the loading and `materialize_early` times.
- `summary.csv`: min, median, max per method and the median speed-up of each
  new method over its legacy counterpart.
- `manifest.json`: the MARS request or synthetic settings, grid shape, dtype,
  backend (numpy or dask) and chunks, `--materialize-early`, seed, method
  order of the warm-up and of every round, package versions, git revision (and
  whether the tree was dirty), dask scheduler and thread settings, and whether
  the outputs agreed.

Include `summary.csv` and `manifest.json` when you post numbers.
