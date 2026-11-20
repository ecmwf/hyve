# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""CRPS of ensemble forecasts and absolute error of persistence."""

import numpy as np
import xarray as xr
from earthkit.meteo.score.array import crps as ensemble_crps

from hyve.tools.forecast_crps.errors import DimensionMismatchError


def check_dims(obj, name, required, optional=(), hint=""):
    """Raise DimensionMismatchError unless `obj` has exactly the allowed dimensions.

    `obj` must have every dimension in `required` and no dimension outside
    `required` + `optional`. The message names the missing and the unexpected
    dimensions and the allowed ones, followed by `hint`.
    """
    missing = [d for d in required if d not in obj.dims]
    extra = [d for d in obj.dims if d not in (*required, *optional)]
    if not missing and not extra:
        return
    problems = []
    if missing:
        problems.append("has no {} dimension".format(", ".join(map(repr, missing))))
    if extra:
        problems.append(
            "has unexpected {} {}".format(
                "dimensions" if len(extra) > 1 else "dimension",
                ", ".join(map(repr, extra)),
            )
        )
    allowed = ", ".join(map(repr, required))
    if optional:
        allowed += " and optionally " + ", ".join(map(repr, optional))
    raise DimensionMismatchError(
        "{} {} (its dimensions are {}; allowed: {}){}".format(
            name, " and ".join(problems), list(obj.dims), allowed, hint
        )
    )


def _crps_over_last_axis(x, y):
    """earthkit-meteo CRPS for an ensemble held on the last axis of `x`.

    `y` is broadcast against the non-ensemble axes of `x`, so any number of
    leading (time, spatial or other) axes is supported.
    """
    x = np.moveaxis(np.asarray(x), -1, 0)
    y = np.broadcast_to(np.asarray(y), x.shape[1:])
    return ensemble_crps(x, y, nan_policy="propagate")


def forecast_crps(x, y, core_dims=None, ensemble_dim="ensemble"):
    """CRPS of the ensemble `x` against `y`, broadcast over all other dimensions.

    Only `ensemble_dim` is reduced. Every other dimension of `x` and `y` is
    broadcast by name and aligned on its coordinates (inner join), so the
    computation does not depend on which spatial dimensions are used
    (``station``, ``lat``/``lon``, ``y``/``x``, ...) or on their order.

    `core_dims` names the spatial dimensions; by default every dimension of
    `x` other than `ensemble_dim` and ``time``. Points where the first member
    is missing at the first index of every non-spatial dimension (the first
    lead time) are masked to NaN everywhere. The result has the non-spatial
    dimensions first, then `core_dims`.

    Observation contract: `y` must have every dimension in `core_dims` and no
    dimension other than `core_dims` and an optional ``time``, even if `x`
    also has that dimension (``ensemble``, ``system``, ...); otherwise
    DimensionMismatchError is raised. An observation without ``time`` is
    broadcast to every step. Dimensions only `x` has (several forecast
    systems, say) are allowed and broadcast.
    """
    if core_dims is None:
        core_dims = [d for d in x.dims if d not in (ensemble_dim, "time")]
    core_dims = list(core_dims)
    check_dims(y, "observation", core_dims, optional=("time",))
    other_dims = [d for d in x.dims if d != ensemble_dim and d not in core_dims]

    first = {dim: 0 for dim in other_dims}
    first[ensemble_dim] = 0
    has_data = x.isel(first, drop=True).notnull()

    crps = xr.apply_ufunc(
        _crps_over_last_axis,
        x,
        y,
        input_core_dims=[[ensemble_dim], []],
        output_core_dims=[[]],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64],
        join="inner",
    )
    crps = crps.where(has_data)
    return crps.transpose(..., *core_dims)


def persistence_crps(reanalysis, persistence):
    """Absolute error of a persistence forecast, as a labelled DataArray.

    `persistence` is the reanalysis at the persistence date. Its singleton
    ``time`` dimension, if any, is dropped so that the field broadcasts over
    every step of `reanalysis`; all other dimensions are aligned on their
    coordinates, so the result does not depend on their order.
    """
    if "time" in persistence.dims:
        persistence = persistence.isel(time=0, drop=True)
    return np.fabs(reanalysis - persistence)
