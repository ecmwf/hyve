#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Deprecated: use `hyve.tools.forecast_crps` and `hyve-forecast-crps`."""

import sys

from earthkit.meteo.score.array import crps as ensemble_crps  # noqa: F401

from hyve.tools.forecast_crps import (  # noqa: F401
    CoordinateMismatchError,
    DimensionMismatchError,
    GridMismatchError,
    StationMismatchError,
    check_same_station_kind,
    common_coords,
    common_stations,
    compute_score,
    coord_doy,
    forecast_crps,
)
from hyve.tools.forecast_crps import main as _main
from hyve.tools.forecast_crps import (  # noqa: F401
    persistence_crps,
    run,
    shift_dates,
    spatial_dims_from_args,
    station_id_kind,
)


def main():
    sys.stderr.write("hyve-crps-stations is deprecated; use hyve-forecast-crps\n")
    _main()


if __name__ == "__main__":
    main()
