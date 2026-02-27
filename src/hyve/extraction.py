import logging

import numpy as np
import pandas as pd
import xarray as xr

from hyve.config import ExtractorConfig, GridConfig, StationConfig
from hyve.core import load_da

logger = logging.getLogger(__name__)


def process_grid_inputs(
    grid: GridConfig,
) -> tuple[xr.DataArray, str, str, str, tuple[int, int]]:
    """Load gridded data and extract dimension metadata.

    Parameters
    ----------
    grid : GridConfig
        Grid configuration containing ``source`` and ``coords``.

    Returns
    -------
    da : xr.DataArray
        The loaded data array.
    var_name : str
        Name of the extracted variable.
    x_dim : str
        Name of the x (latitude) dimension.
    y_dim : str
        Name of the y (longitude) dimension.
    shape : tuple[int, int]
        Grid shape as ``(n_x, n_y)``.
    """
    source_dict = {
        "source": grid.source,
        "to_xarray_options": grid.to_xarray_options,
    }
    da, var_name = load_da(source_dict, 3)
    logger.info(f"Xarray created from source:\n{da}\n")
    x_dim = grid.coords.x
    y_dim = grid.coords.y
    shape = da[x_dim].shape[0], da[y_dim].shape[0]
    return da, var_name, x_dim, y_dim, shape


def _validate_index_bounds(df: pd.DataFrame, shape: tuple[int, int]) -> None:
    """Raise if any station index is outside the grid dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Station DataFrame with ``x_index`` and ``y_index`` columns.
    shape : tuple[int, int]
        Grid shape ``(n_x, n_y)``.

    Raises
    ------
    ValueError
        If any index is out of the grid bounds.
    """
    x_indices = df["x_index"].values
    y_indices = df["y_index"].values
    if (
        np.any(x_indices < 0)
        or np.any(x_indices >= shape[0])
        or np.any(y_indices < 0)
        or np.any(y_indices >= shape[1])
    ):
        raise ValueError(
            f"Station indices out of grid bounds. Grid shape={shape}, "
            f"x_index range=({int(x_indices.min())},{int(x_indices.max())}), "
            f"y_index range=({int(y_indices.min())},{int(y_indices.max())})"
        )


def parse_stations(station: StationConfig) -> pd.DataFrame:
    """Read, filter, and normalize station DataFrame to canonical column names.

    Parameters
    ----------
    station : StationConfig
        Validated station configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical columns (``station_name`` plus mapping cols).

    Raises
    ------
    ValueError
        If stations are empty after filtering, contain duplicate names,
        or have NaN in mapping columns.
    """
    logger.debug(f"Reading station file: {station.file}")
    df = pd.read_csv(station.file)
    if station.filter is not None:
        logger.debug(f"Applying filter: {station.filter}")
        df = df.query(station.filter)

    if len(df) == 0:
        raise ValueError("No stations found. Check station file or filter.")

    renames = {station.name: "station_name"}
    mapping_cols: list[str] = []

    if station.index is not None:
        renames[station.index.x] = "x_index"
        renames[station.index.y] = "y_index"
        mapping_cols.extend(["x_index", "y_index"])

    if station.coords is not None:
        renames[station.coords.x] = "x_coord"
        renames[station.coords.y] = "y_coord"
        mapping_cols.extend(["x_coord", "y_coord"])

    if station.index_1d is not None:
        renames[station.index_1d] = "index_1d"
        mapping_cols.append("index_1d")

    df_renamed = df.rename(columns=renames)

    # Validate mapping columns exist (catches typos in config column names)
    missing = [c for c in mapping_cols if c not in df_renamed.columns]
    if missing:
        raise ValueError(
            f"Station file is missing required columns after rename: {missing}. "
            f"Check that the column names in your config match the CSV headers."
        )

    # Validate no duplicate station names
    dupes = df_renamed["station_name"].duplicated()
    if dupes.any():
        n_dupes = dupes.sum()
        examples = df_renamed.loc[dupes, "station_name"].unique()[:5].tolist()
        raise ValueError(
            f"Duplicate station names found ({n_dupes} duplicates). "
            f"Examples: {examples}"
        )

    # Validate no NaN in mapping columns
    for col in mapping_cols:
        n_nan = df_renamed[col].isna().sum()
        if n_nan > 0:
            raise ValueError(
                f"Column '{col}' contains {n_nan} NaN value(s). "
                f"All station mapping values must be non-null."
            )

    return df_renamed


def _process_gribjump(grid: GridConfig, df: pd.DataFrame) -> xr.Dataset:
    """Handle extraction via gribjump 1-D indices."""
    if "index_1d" not in df.columns:
        raise ValueError("Gribjump source requires 'index_1d' in station config.")

    station_names = df["station_name"].values
    unique_indices, duplication_indexes = np.unique(
        df["index_1d"].values, return_inverse=True
    )  # type: ignore[call-overload]

    # Converting indices to ranges is currently faster than using indices
    # directly. This is a problem in the earthkit-data gribjump source and will
    # be fixed there.
    ranges = [(i, i + 1) for i in unique_indices]

    gribjump_config = {
        "source": {
            "gribjump": {
                **grid.source["gribjump"],
                "ranges": ranges,
            }
        },
        "to_xarray_options": grid.to_xarray_options,
    }

    masked_da, var_name = load_da(gribjump_config, 2)

    ds = xr.Dataset({var_name: masked_da})
    ds = ds.isel(index=duplication_indexes)
    ds = ds.rename({"index": "station"})
    ds["station"] = station_names
    return ds


def _process_regular(
    grid: GridConfig, df: pd.DataFrame, coord_tolerance: float | None
) -> xr.Dataset:
    """Handle extraction via 2-D index or coordinate mapping.

    Uses xarray vectorized indexing (``sel``/``isel``) to extract station
    timeseries directly, avoiding intermediate boolean masks.

    Parameters
    ----------
    grid : GridConfig
        Validated grid configuration.
    df : pd.DataFrame
        Station DataFrame with canonical column names.
    coord_tolerance : float | None
        Tolerance in degrees for coordinate matching. ``None`` disables the
        tolerance guard.

    Returns
    -------
    xr.Dataset
        Dataset with a ``station`` dimension.
    """
    station_names = df["station_name"].values
    da, var_name, x_dim, y_dim, shape = process_grid_inputs(grid)

    use_index = "x_index" in df.columns and "y_index" in df.columns

    if use_index:
        _validate_index_bounds(df, shape)
        xi = xr.DataArray(df["x_index"].values, dims="station")
        yi = xr.DataArray(df["y_index"].values, dims="station")
        logger.info("Extracting timeseries via index-based vectorized indexing")
        result = da.isel({x_dim: xi, y_dim: yi})
    else:
        target_x = xr.DataArray(df["x_coord"].values, dims="station")
        target_y = xr.DataArray(df["y_coord"].values, dims="station")
        logger.info("Extracting timeseries via coordinate-based vectorized indexing")
        sel_kwargs = {"method": "nearest"}
        if coord_tolerance is not None:
            sel_kwargs["tolerance"] = coord_tolerance
        try:
            result = da.sel({x_dim: target_x, y_dim: target_y}, **sel_kwargs)
        except KeyError as exc:
            raise ValueError(
                "Station coordinates outside tolerance of grid. "
                "Set station.coords.tolerance to null to disable."
            ) from exc

    result = result.compute()

    ds = xr.Dataset({var_name: result})
    ds["station"] = station_names
    return ds


def process_inputs(station: StationConfig, grid: GridConfig) -> xr.Dataset:
    """Parse stations and dispatch to the appropriate extraction backend.

    Parameters
    ----------
    station : StationConfig
        Validated station configuration.
    grid : GridConfig
        Validated grid configuration.

    Returns
    -------
    xr.Dataset
        Dataset with a ``station`` dimension.
    """
    df = parse_stations(station)
    if "gribjump" in grid.source:
        return _process_gribjump(grid, df)
    coord_tolerance = None
    if station.coords is not None:
        coord_tolerance = station.coords.tolerance
    return _process_regular(grid, df, coord_tolerance)


def extractor(config: ExtractorConfig) -> xr.Dataset:
    """Extract timeseries at station locations from gridded data.

    Parameters
    ----------
    config : ExtractorConfig
        Validated extraction configuration.

    Returns
    -------
    xr.Dataset
        Dataset with a ``station`` dimension.
    """
    ds = process_inputs(config.station, config.grid)
    if config.output is not None:
        logger.info(f"Saving output to {config.output.file}")
        ds.to_netcdf(str(config.output.file))
    return ds
