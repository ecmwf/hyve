# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Forecast CRPS: CRPS and CRPSS of reforecasts against a reanalysis, with
persistence and (optionally) climatology benchmarks.
"""

from hyve.tools.forecast_crps.cli import main, run, spatial_dims_from_args
from hyve.tools.forecast_crps.crps import forecast_crps, persistence_crps
from hyve.tools.forecast_crps.errors import (
    CoordinateMismatchError,
    DimensionMismatchError,
    GridMismatchError,
    StationMismatchError,
)
from hyve.tools.forecast_crps.matching import (
    check_same_station_kind,
    common_coords,
    common_stations,
    match_coords,
    match_values,
    narrow,
    relabel,
    station_id_kind,
)
from hyve.tools.forecast_crps.scores import compute_score, coord_doy, shift_dates

__all__ = [
    "CoordinateMismatchError",
    "DimensionMismatchError",
    "GridMismatchError",
    "StationMismatchError",
    "check_same_station_kind",
    "common_coords",
    "common_stations",
    "compute_score",
    "coord_doy",
    "forecast_crps",
    "main",
    "match_coords",
    "match_values",
    "narrow",
    "persistence_crps",
    "relabel",
    "run",
    "shift_dates",
    "spatial_dims_from_args",
    "station_id_kind",
]
