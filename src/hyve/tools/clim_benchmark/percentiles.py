"""Compute percentile climatologies over pre-built (doy, issue) slots."""

from __future__ import annotations

import itertools
import logging

import dask
import numpy as np
import xarray as xr

from hyve.tools.clim_benchmark.sampling import Slot, gather

logger = logging.getLogger(__name__)


def _quantile_slot(da_slot: xr.DataArray, quantiles: np.ndarray) -> xr.DataArray:
    """Compute percentiles for a single (doy, issue) slot."""
    result = da_slot.quantile(q=quantiles, dim="time")
    # ``quantile`` names the new dim ``quantile`` with floating-point
    # coord values; rename to ``ensemble`` and replace the coord with
    # the integer percentile values so coord arithmetic is exact.
    percentiles = (quantiles * 100).round().astype(np.int64)
    return (
        result.rename({"quantile": "ensemble"})
        .assign_coords(ensemble=percentiles)
        .drop_vars([c for c in result.coords if c == "quantile"], errors="ignore")
    )


def chunked(iterable, size):
    it = iter(iterable)
    while batch := list(itertools.islice(it, size)):
        yield batch


def compute_climatology(
    da: xr.DataArray,
    slots: dict[Slot, np.ndarray],
    percentiles: list[int],
    worker_count: int,
) -> xr.DataArray:
    """Compute the full ``(doy, issue_hour, ensemble, *space)`` climatology.

    Parameters
    ----------
    da
        Input DataArray with a ``time`` dimension.
    slots
        Mapping ``{(doy, issue_hour): indices}`` produced by
        :func:`sampling.build_slots`.
    percentiles
        Integer percentiles in ``[0, 100]`` (e.g. ``[0, 10, ..., 100]``).
    """
    quantiles = np.asarray(percentiles, dtype=np.float64) / 100.0

    # Submit each slot as a delayed task so peak memory stays bounded by
    # a single slot's pool rather than the whole dataset.
    logger.info("Scheduling %d (doy, issue_hour) slots", len(slots))

    delayed_results: dict[Slot, xr.DataArray] = {}

    computed_list = []
    keys: list[Slot] = []
    total_slots = len(slots.items())
    # Only create 'worker_count' tasks at one time to limit memory usage
    for i, batch in enumerate(chunked(slots.items(), worker_count)):
        logger.info(
            "Calculating samples %d - %d of %d",
            i * worker_count,
            i * worker_count + worker_count,
            total_slots,
        )
        delayed_results: dict[Slot, xr.DataArray] = {}
        for slot_key, indices in batch:
            sub = gather(da, indices)
            delayed_results[slot_key] = dask.delayed(_quantile_slot)(sub, quantiles)
        batch_keys = list(delayed_results.keys())
        keys.extend(batch_keys)
        computed_list.extend(dask.compute(*(delayed_results[k] for k in batch_keys)))
    computed: dict[Slot, xr.DataArray] = dict(zip(keys, computed_list))

    # Build the full output array: (doy, issue_hour, ensemble, *space).
    issue_hours = sorted({ih for (_, ih) in slots.keys()})
    space_dims = [d for d in da.dims if d != "time"]
    space_sizes = {d: da.sizes[d] for d in space_dims}
    space_coords = {d: da.coords[d] for d in space_dims if d in da.coords}

    out_shape = (366, len(issue_hours), len(percentiles), *space_sizes.values())
    if computed:
        out_dtype = next(iter(computed.values())).dtype
    else:
        out_dtype = (
            np.float64
            if np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.bool_)
            else da.dtype
        )
    data = np.full(out_shape, np.nan, dtype=out_dtype)

    doy_index = {d: i for i, d in enumerate(range(1, 367))}
    issue_index = {h: i for i, h in enumerate(issue_hours)}
    for (doy, issue_hour), slot_da in computed.items():
        data[doy_index[doy], issue_index[issue_hour]] = slot_da.values

    coords = {
        "doy": np.arange(1, 367, dtype=np.int64),
        "issue_hour": np.asarray(issue_hours, dtype=np.int64),
        "ensemble": np.asarray(percentiles, dtype=np.int64),
    }
    coords.update(space_coords)

    return xr.DataArray(
        data,
        dims=("doy", "issue_hour", "ensemble", *space_dims),
        coords=coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
