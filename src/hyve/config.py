from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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


class GridConfig(StrictBaseModel):
    source: dict[str, Any]
    coords: GridCoords = GridCoords()
    to_xarray_options: dict[str, Any] = {}

    @field_validator("source")
    @classmethod
    def source_has_single_key(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) != 1:
            raise ValueError(
                f"'source' must have exactly one key (the earthkit-data source type), "
                f"got {len(v)}: {list(v.keys())}"
            )
        return v


class OutputConfig(StrictBaseModel):
    file: Path  # pydantic coerces str -> Path; xr.to_netcdf() accepts both


class ExtractorConfig(StrictBaseModel):
    station: StationConfig
    grid: GridConfig
    output: Optional[OutputConfig] = None
