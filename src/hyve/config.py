"""Pydantic configuration models for the extraction pipeline."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, model_validator


class IndexMapping(BaseModel):
    """Column names for 2D index-based station mapping."""

    x: str = "opt_x_index"
    y: str = "opt_y_index"


class CoordMapping(BaseModel):
    """Column names for coordinate-based station mapping.

    Notes
    -----
    ``tolerance`` is expressed in degrees and defaults to ~10 m at the equator
    (``1e-4``). Set to ``null`` in YAML to disable the tolerance guard.
    """

    x: str = "opt_x_coord"
    y: str = "opt_y_coord"
    tolerance: Optional[float] = 1e-4


class GridCoords(BaseModel):
    """Dimension names for grid coordinates."""

    x: str = "lat"
    y: str = "lon"


class StationConfig(BaseModel):
    """Configuration for station data input."""

    file: Path
    name: str
    filter: Optional[str] = None
    index: Optional[IndexMapping] = None
    coords: Optional[CoordMapping] = None
    index_1d: Optional[str] = None

    @model_validator(mode="after")
    def check_mapping_exclusivity(self) -> "StationConfig":
        if self.index_1d is not None:
            return self
        if self.index is not None and self.coords is not None:
            raise ValueError(
                "Station config must use either 'index' or 'coords', not both."
            )
        if self.index is None and self.coords is None:
            raise ValueError(
                "Station config must provide either 'index' or 'coords' "
                "for station mapping."
            )
        return self


class GridConfig(BaseModel):
    """Configuration for grid data input."""

    source: dict[str, Any]
    coords: GridCoords = GridCoords()
    to_xarray_options: dict[str, Any] = {}


class OutputConfig(BaseModel):
    """Configuration for output file."""

    file: Path


class ExtractorConfig(BaseModel):
    """Top-level configuration for the extraction pipeline."""

    station: StationConfig
    grid: GridConfig
    output: Optional[OutputConfig] = None
