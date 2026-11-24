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

```
pip install git+https://github.com/ecmwf/hyve.git@main
```

### Development

Install project with uv

```
# (Optional) Create and activate virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install hyve with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Inputs

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

### `hyve-forecast-crps`: memory

By default every input file is read whole. For large domains, read them lazily
in chunks over the spatial dimensions and fix the number of threads:

```shell
hyve-forecast-crps --reanalysis reanalysis.nc --reforecast forecasts/ \
    --climatology climatology.nc --spatial_dims point \
    --chunk_size 5000 --threads 4
```

`--chunk_size` is the number of spatial points per chunk; with several spatial
dimensions the first one is split into rows of whole slices of the others. The
climatology is also chunked by day of year, so a forecast reads only the days it
is scored against. Peak memory is close to the data held in memory plus
threads x 8 x chunk bytes, where a chunk holds every ensemble member and time
step of its points counted in float64 (the CRPS is computed in float64 even for
float32 files): 51 members, 40 steps and 5,000 points make 82 MB, so 4 threads
need about 2.6 GB on top of the outputs. The scores do not depend on
the chunk size. `--scheduler synchronous` runs everything in one thread, for
debugging.

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
