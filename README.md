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

## Troubleshooting

### `hyve-crps-stations`: station IDs cannot be matched

`hyve-crps-stations` matches stations between the reanalysis, the reforecast files
and the (optional) climatology by their IDs. It exits with status 1 and a one-line
`Error: ...` message on stderr in two cases:

- *Station IDs of different kinds*: one dataset has integer IDs (for example
  `int64`) and another has text IDs (for example `"1"`) or byte-string IDs. The message names both
  datasets, their dtypes, station counts and a few sample IDs.
- *No stations in common*: the datasets use the same kind of ID but no ID appears
  in both. This includes a climatology that shares no station with the reanalysis.

IDs are never converted, so integer `1` and text `"1"` are different stations.
To fix it, make the station ID type consistent upstream (in the extraction or
the dataset that holds the odd ID type) and rerun.

## License

See [LICENSE](LICENSE)

## Copyright

© 2025 ECMWF. All rights reserved.
