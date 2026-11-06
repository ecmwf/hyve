# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexMapping(StrictBaseModel):
    x: str = "opt_x_index"
    y: str = "opt_y_index"


class CoordMapping(StrictBaseModel):
    x: str = "opt_x_coord"
    y: str = "opt_y_coord"


class GridCoords(StrictBaseModel):
    x: str = "lat"
    y: str = "lon"


class StationConfig(StrictBaseModel):
    file: str
    name: str
    index: Optional[IndexMapping] = None
    coords: Optional[CoordMapping] = None
    index_1d: Optional[str] = None
    ups_area: Optional[str] = None
    filter: Optional[str] = None

    @model_validator(mode="after")
    def check_mapping_exclusivity(self) -> "StationConfig":
        """Ensure exactly one location mapping is provided."""
        active = sum(
            [
                self.index is not None,
                self.coords is not None,
                self.index_1d is not None,
            ]
        )
        if active != 1:
            raise ValueError(
                "Station config must use exactly one of 'index', 'coords', or 'index_1d'."
            )
        return self


class EarthkitDataSource(StrictBaseModel):
    name: str
    options: Any
    variable: str | None = None

    @model_validator(mode="after")
    def check_name_and_options(self) -> "EarthkitDataSource":
        if self.name == "gribjump":
            if isinstance(self.options, dict):
                if not self.options.get("request"):
                    raise ValueError("Gribjump source requires 'request' in options.")
                if len(self.options) != 1:
                    raise ValueError(
                        "Gribjump source options must only contain one 'request' entry"
                    )
            else:
                raise ValueError("Gribjump source options must be a dictionary.")
        return self


class GridConfig(StrictBaseModel):
    source: EarthkitDataSource
    coords: GridCoords = GridCoords()
    to_xarray_options: dict[str, Any] = {}


class OutputConfig(StrictBaseModel):
    file: Path  # pydantic coerces str -> Path; xr.to_netcdf() accepts both


class ExtractorConfig(StrictBaseModel):
    station: StationConfig
    grid: GridConfig
    output: Optional[OutputConfig] = None
