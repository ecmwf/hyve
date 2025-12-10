import earthkit.data as ekd


def find_main_var(ds, min_dim=2):
    """
    Find the main variable in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
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
        If no variable or more than one variable with the required dimensions is found.
    """
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) >= min_dim]
    if len(variable_names) > 1:
        raise ValueError("More than one variable of dimension >= {min_dim} in dataset.")
    elif len(variable_names) == 0:
        raise ValueError(f"No variable of dimension >= {min_dim} in dataset.")
    else:
        return variable_names[0]


def load_da(ds_config, n_dims):
    src_name = list(ds_config["source"].keys())[0]
    source = ekd.from_source(src_name, **ds_config["source"][src_name])
    ds = source.to_xarray(**ds_config.get("to_xarray_options", {}))
    var_name = find_main_var(ds, n_dims)
    da = ds[var_name]
    return da, var_name
