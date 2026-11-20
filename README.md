<!--
SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)

SPDX-License-Identifier: Apache-2.0
-->

# Hyve

<p align="center">
    <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
        <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
    </a>
    <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
        <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
    </a>
    <a href="https://opensource.org/licenses/apache-2-0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
    </a>
    <a href="https://github.com/ecmwf/hyve/releases">
        <img src="https://img.shields.io/github/v/release/ecmwf/hyve?color=purple&label=Release" alt="Latest Release">
    </a>
</p>

<p align="center">
  <a href="#installation">Installation</a> *
  <a href="#development">Development</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Hyve is a library and toolkit for hydrological verification.

## Installation

Get the latest release from PyPI

```
pip install hydro-verification
```

or, for unreleased changes,

```
pip install git+https://github.com/ecmwf/hyve.git@main
```

### Development

After cloning locally, install the project with uv >= 0.13.5

```
# Create the virtual environment and install hyve with its development dependencies
uv sync

# Install the pre-commit hooks
uv run pre-commit install
```

Run tests
```
uv run pytest
```

## CLI tools

With a local installation, run one of the CLI tools

```
uv run hyve-extract-timeseries --help
```

or run a tool without installing hyve first (outside a local clone)

```
uv run --with git+https://github.com/ecmwf/hyve.git@main hyve-extract-timeseries --help
```

### `hyve-forecast-crps`

The variable `dis` of each input has exactly these dimensions (any order). The
spatial dimensions are chosen with `--spatial_dims` (default `station`, for
example `lat lon` or `y x`) or `--core_dim` for a single one.

| Input        | `dis` dimensions                                  |
| ------------ | ------------------------------------------------- |
| Reanalysis   | `time`, *spatial dims*                            |
| Reforecast   | `ensemble`, `time`, *spatial dims* (one file per date) |
| Climatology  | `ensemble`, `doy`, `issue_hour`, *spatial dims*   |

The reanalysis may not have any other dimension (such as `ensemble` or
`system`); otherwise the tool exits with a dimension mismatch error. The
reforecast grid is the reference and the other inputs are matched to it, see
[Troubleshooting](#hyve-forecast-crps-stations-or-grids-cannot-be-matched).

### `hyve-extract-timeseries`

`hyve-extract-timeseries` extracts a time series at each station from a gridded
source. The config says how each station is placed on the grid, using one of
these mappings:

- **Index mapping** (`station.index`): the station file gives the grid indices of
  each station. The indices refer to the grid as stored in the source; the grid is
  not sorted. Compute the indices on the same grid layout that is extracted, for
  example with the same latitude order (north-to-south or south-to-north).
- **Coordinate mapping** (`station.coords`): the station file gives coordinates, and
  each station is placed on the nearest grid point. The nearest point is chosen
  separately on each axis, so it does not matter in which order the grid is stored.

If a station lies exactly halfway between two grid points on an axis, it goes to
the grid point with the larger coordinate value. This holds whether the grid
coordinates ascend or descend. Earlier versions picked the smaller coordinate, so
a few stations on exact midpoints may now move by one grid cell.

`station.coords.tolerance` is optional. It is the largest offset allowed between a
station and its nearest grid point, checked separately on each axis, in the units
of the grid's coordinates (for example degrees). It must be zero or positive. If it
is not set (the default), there is no check and every station is placed on its nearest
grid point. If a station is further away than the tolerance, the extraction stops
with an error.

## Troubleshooting

### `hyve-forecast-crps`: stations or grids cannot be matched

`hyve-forecast-crps` matches the reanalysis, the reforecast files and the
(optional) climatology by their station IDs or grid coordinates. The reforecast
grid is the reference: reanalysis and climatology are relabelled to its
coordinate values (and output files carry them), and only points present in
both are scored. The latest reforecast file (by sorted name) defines the grid
and every other file must cover it. A mismatch exits with status 1 and a
one-line `Error: ...` message on stderr. The message names the dimension, both
datasets and, for float grids, the tolerance. There are three kinds:

- *Station mismatch* (`StationMismatchError`): station IDs of different kinds
  (integer vs text vs bytes, never converted), no ID in common, or a reforecast
  file lacking a reference station, or duplicate IDs. Fix the ID type or the
  duplicates upstream and rerun.
- *Grid mismatch* (`GridMismatchError`): the same problems for `lat`, `lon`,
  `y`, `x`, plus ambiguous tolerances, NaN/infinite coordinates and duplicate
  coordinates (of any type, integer or float).
- *Dimension mismatch* (`DimensionMismatchError`): a dataset has no `lat` (or
  other `--spatial_dims`) dimension, or the reanalysis has a dimension other
  than `time` and the spatial dimensions. Set `--spatial_dims` (or `--core_dim`) to
  the dimensions the files actually have.

Float coordinates (`lat`, `lon`, projected metres) match within a tolerance
relative to their precision, 4 x the machine epsilon of the coarser dtype, so a
float32 grid matches the same float64 grid. Integer coordinates (station IDs,
integer `y`/`x`) match exactly. If grids differ by more (for example a
rounded-coordinate file), pass an absolute tolerance in coordinate units:
`--coord_tolerance 0.001` (finite, >= 0).

*Ambiguous* means the tolerance is at least half the smallest grid spacing (or
two values snap to one point): a value could match two cells, so nothing is
matched. Lower `--coord_tolerance` or remove it, and check the coordinates.
The old name `hyve-crps-stations` still works but is deprecated.

## License
### License headers

Every file declares its copyright and license with [SPDX](https://spdx.dev) tags following
the [REUSE](https://reuse.software) specification, as set out in ECMWF's
[ADR-010](https://github.com/ecmwf/codex/blob/main/ADR/ADR-010-SPDX-License-Identifiers-and-REUSE-Headers.md).
Add the header to a new file with:

```
uvx reuse annotate \
    --copyright "European Centre for Medium-Range Weather Forecasts (ECMWF)" \
    --license Apache-2.0 \
    path/to/new_file.py
```

which writes, in the file's comment style and with the current year:

```
# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0
```

- `uvx` runs `reuse` without installing it; `pip install reuse` works too.
- If `reuse` does not recognise the file type, pick a comment style with `--style`, e.g. `--style python`.
- For a co-developed file, pass `--copyright` once per copyright holder.
- Files that cannot carry a comment (data, generated files) are covered in [REUSE.toml](REUSE.toml) instead.

The `reuse` pre-commit hook checks that every file is covered; run the same check by hand
with `uvx reuse lint`.

## License

Hyve is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE), which also carry ECMWF's intergovernmental notice.

© 2025-2026 European Centre for Medium-Range Weather Forecasts (ECMWF).
