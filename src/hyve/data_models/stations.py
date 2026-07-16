# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class StationsFileConfig(BaseModel):
    """Configuration for station metadata file.

    Attributes:
    -----------
        file (str): Path to the CSV file containing station metadata.
        station_id (str): Column name for the station ID.
        station_lat (str): Column name for the latitude of the station's official location.
        station_lon (str): Column name for the longitude of the station's official location.
        grid_x (str): Column name for the X coordinate in the grid.
        grid_y (str): Column name for the Y coordinate in the grid.

    Example
    -------
    ```yaml
        stations:
            file: "path/to/station_metadata.csv"
            station_id: "station_id"
            station_lat: "station_lat"
            station_lon: "station_lon"
            grid_x: "opt_x_coord"
            grid_y: "opt_y_coord"
    ```
    # TODO: add `filters` option accepting expressions for column-based filtering of table entries
    """

    file: str
    station_id: str
    station_lat: str
    station_lon: str
    grid_x: str
    grid_y: str
    ups_area: str | None = None
