# SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import earthkit.data as ekd
import xarray as xr


def find_main_var(ds: xr.Dataset, min_dim: int = 2) -> str:
    """
    Find the main variable in the dataset.

    Selects the single variable whose number of dimensions is at least
    ``min_dim``. Raises if zero or more than one variable qualifies.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to search for the main variable.
    min_dim : int, optional
        The minimum number of dimensions the variable must have. Default is 2.

    Returns
    -------
    str
        The name of the main variable.

    Raises
    ------
    ValueError
        If no variable or more than one variable with the required dimensions
        is found.
    """
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) >= min_dim]
    if len(variable_names) > 1:
        raise ValueError(
            f"More than one variable of dimension >= {min_dim} in dataset."
        )
    elif len(variable_names) == 0:
        raise ValueError(f"No variable of dimension >= {min_dim} in dataset.")
    else:
        return variable_names[0]


def load_da(
    ds_config: dict[str, Any], n_dims: int, variable: str | None = None
) -> tuple[xr.DataArray, str, dict[str, Any]]:
    """Load a DataArray from an earthkit source configuration.

    Parameters
    ----------
    ds_config : dict[str, Any]
        Must contain ``source`` (a single-key dict whose key is the earthkit
        source name and value is the source kwargs). Optionally contains
        ``to_xarray_options`` passed to ``source.to_xarray()``.
    n_dims : int
        Minimum number of dimensions to identify the main variable.
    variable : str, optional
        Name of the variable to select. If ``None``, the main variable is
        detected with :func:`find_main_var`.

    Returns
    -------
    da : xr.DataArray
        The loaded data array.
    var_name : str
        Name of the selected variable.
    ds_attrs : dict[str, Any]
        Global attributes of the loaded dataset.
    """
    src_name = list(ds_config["source"].keys())[0]
    source = ekd.from_source(src_name, **ds_config["source"][src_name])
    ds = source.to_xarray(**ds_config.get("to_xarray_options", {}))
    ds_attrs = ds.attrs
    var_name = variable or find_main_var(ds, n_dims)
    da = ds[var_name]
    return da, var_name, ds_attrs
