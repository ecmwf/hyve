# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Climatology benchmark: compute a DOY-indexed percentile climatology
from a reanalysis dataset, shaped as a synthetic ensemble forecast
keyed by `(doy, issue_hour, ensemble, *space)`.
"""

from hyve.tools.clim_benchmark.cli import main

__all__ = ["main"]
