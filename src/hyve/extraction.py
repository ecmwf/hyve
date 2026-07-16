# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from hyve.config import ExtractorConfig, GridConfig, StationConfig
from hyve.core import load_da

logger = logging.getLogger(__name__)


def process_grid_inputs(
    grid: GridConfig,
) -> tuple[xr.DataArray, str, str, str, tuple[int, int]]:
    """Load a DataArray from the grid source and return spatial metadata.

    Parameters
    ----------
    grid : GridConfig
        Grid configuration specifying the earthkit-data source and coordinate names.

    Returns
    -------
    da : xr.DataArray
        The loaded and coordinate-sorted DataArray.
    var_name : str
        Name of the main variable in the dataset.
    x_dim : str
        Name of the x spatial dimension.
    y_dim : str
        Name of the y spatial dimension.
    shape : tuple[int, int]
        Grid shape (n_x, n_y).
    """
    da, var_name = load_da(
        {"source": grid.source, "to_xarray_options": grid.to_xarray_options}, 3
    )
    logger.info(f"Xarray created from source:\n{da}\n")
    x_dim = grid.coords.x
    y_dim = grid.coords.y
    da = da.sortby([x_dim, y_dim])
    shape = da[x_dim].shape[0], da[y_dim].shape[0]
    return da, var_name, x_dim, y_dim, shape


def construct_mask(x_indices, y_indices, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[x_indices, y_indices] = True

    flat_indices = np.ravel_multi_index((x_indices, y_indices), shape)
    _, duplication_indexes = np.unique(flat_indices, return_inverse=True)
    return mask, duplication_indexes


def create_mask_from_index(df, shape):
    logger.info(f"Creating mask {shape} from index")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
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
    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def create_mask_from_coords(df, gridx, gridy, shape):
    logger.info(f"Creating mask {shape} from coordinates")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    station_x = df["x_coord"].values
    station_y = df["y_coord"].values

    x_distances = np.abs(station_x[:, np.newaxis] - gridx)
    x_indices = np.argmin(x_distances, axis=1)
    y_distances = np.abs(station_y[:, np.newaxis] - gridy)
    y_indices = np.argmin(y_distances, axis=1)

    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def parse_stations(station: StationConfig) -> pd.DataFrame:
    """Read, filter, and normalize station CSV to canonical column names.

    Parameters
    ----------
    station : StationConfig
        Station configuration specifying the CSV path, station name column,
        location mapping (index, coords, or index_1d), and optional filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed to canonical names: ``station_name``
        plus ``x_index``/``y_index``, ``x_coord``/``y_coord``, or ``index_1d``
        depending on the configured mapping mode.

    Raises
    ------
    ValueError
        If the station file is empty after filtering, or if the configured
        mapping columns are not present in the CSV.
    """
    logger.debug(f"Reading station file, {station}")
    df = pd.read_csv(station.file)
    if station.filter is not None:
        logger.debug(f"Applying filters: {station.filter} to station DataFrame")
        df = df.query(station.filter)

    if len(df) == 0:
        raise ValueError("No stations found. Check station file or filter.")

    renames = {station.name: "station_name"}

    if station.index is not None:
        renames[station.index.x] = "x_index"
        renames[station.index.y] = "y_index"

    if station.coords is not None:
        renames[station.coords.x] = "x_coord"
        renames[station.coords.y] = "y_coord"

    if station.index_1d is not None:
        renames[station.index_1d] = "index_1d"

    df_renamed = df.rename(columns=renames)

    if station.index is not None and (
        "x_index" not in df_renamed.columns or "y_index" not in df_renamed.columns
    ):
        raise ValueError(
            "Station file missing required index columns."
            "Expected columns to map to 'x_index' and 'y_index'."
        )
    if station.coords is not None and (
        "x_coord" not in df_renamed.columns or "y_coord" not in df_renamed.columns
    ):
        raise ValueError(
            "Station file missing required coordinate columns."
            "Expected columns to map to 'x_coord' and 'y_coord'."
        )
    if station.index_1d is not None and "index_1d" not in df_renamed.columns:
        raise ValueError("Station file missing required 'index_1d' column.")

    return df_renamed


def _process_gribjump(grid: GridConfig, df: pd.DataFrame) -> xr.Dataset:
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
                # fetch_coords_from_fdb is currently very slow. Needs fix in
                # earthkit-data gribjump source.
                # "fetch_coords_from_fdb": True,
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


def _process_regular(grid: GridConfig, df: pd.DataFrame) -> xr.Dataset:
    station_names = df["station_name"].values
    da, var_name, x_dim, y_dim, shape = process_grid_inputs(grid)

    use_index = "x_index" in df.columns and "y_index" in df.columns

    if use_index:
        mask, duplication_indexes = create_mask_from_index(df, shape)
    else:
        mask, duplication_indexes = create_mask_from_coords(
            df, da[x_dim].values, da[y_dim].values, shape
        )

    logger.info("Extracting timeseries at selected stations")
    masked_da = apply_mask(da, mask, x_dim, y_dim)

    ds = xr.Dataset({var_name: masked_da})
    ds = ds.isel(index=duplication_indexes)
    ds = ds.rename({"index": "station"})
    ds["station"] = station_names
    return ds


def process_inputs(station: StationConfig, grid: GridConfig) -> xr.Dataset:
    """Parse station and grid inputs and route to the appropriate extraction path.

    Parameters
    ----------
    station : StationConfig
        Station configuration.
    grid : GridConfig
        Grid configuration. If ``source`` contains a ``gribjump`` key, the
        gribjump extraction path is used; otherwise the regular path is used.

    Returns
    -------
    xr.Dataset
        Dataset with a ``station`` dimension containing the extracted timeseries.
    """
    df = parse_stations(station)
    if "gribjump" in grid.source:
        return _process_gribjump(grid, df)
    return _process_regular(grid, df)


def mask_array_np(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[..., mask]


def apply_mask(
    da: xr.DataArray, mask: np.ndarray, coordx: str, coordy: str
) -> xr.DataArray:
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[(coordx, coordy), (coordx, coordy)],
        output_core_dims=[["index"]],
        output_dtypes=[da.dtype],
        exclude_dims={coordx, coordy},
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"index": int(mask.sum())},
            "allow_rechunk": True,
        },
    )
    with ProgressBar(dt=15):
        return task.compute()


def extractor(config: ExtractorConfig) -> xr.Dataset:
    """Run the full extraction pipeline from validated config.

    Parameters
    ----------
    config : ExtractorConfig
        Fully validated extraction configuration.

    Returns
    -------
    xr.Dataset
        Extracted timeseries dataset. Also written to ``config.output.file``
        if an output path is configured.
    """
    ds = process_inputs(config.station, config.grid)
    if config.output is not None:
        logger.info(f"Saving output to {config.output.file}")
        ds.to_netcdf(config.output.file)
    return ds
