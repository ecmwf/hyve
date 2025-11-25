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
  <a href="#development">Development</a> *
</p>

# Hyve

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging)

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

## License

See [LICENSE](LICENSE)

## Copyright

© 2025 ECMWF. All rights reserved.
