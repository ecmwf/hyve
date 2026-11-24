# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Chunk layout for lazy reading over the spatial dimensions.

Every input is split into chunks of about `chunk_size` spatial points, with
every other dimension (ensemble, time) kept whole, so that each dask task
holds the full ensemble and all lead times of a few points. Peak memory is
then close to the data held in memory plus threads x 8 x chunk bytes.
"""

import math


def spatial_chunks(obj, spatial_dims, chunk_size, extra=None):
    """Chunk sizes of `obj` for about `chunk_size` points per chunk.

    With one spatial dimension the chunk holds `chunk_size` points. With
    several, the first spatial dimension is split into rows that hold whole
    slices of the others, never fewer than one row. `extra` adds fixed chunk
    sizes for other dimensions that `obj` has (``{"doy": 1}``); dimensions
    `obj` does not have are ignored.
    """
    spatial_dims = list(spatial_dims)
    sizes = dict(obj.sizes)
    chunks = {}
    first, others = spatial_dims[0], spatial_dims[1:]
    per_row = math.prod(sizes[dim] for dim in others)
    rows = max(1, chunk_size // per_row)
    chunks[first] = min(rows, sizes[first])
    for dim in others:
        chunks[dim] = sizes[dim]
    for dim, size in (extra or {}).items():
        if dim in sizes:
            chunks[dim] = size
    return chunks


def chunk_spatially(obj, spatial_dims, chunk_size, extra=None):
    """`obj` as dask arrays chunked by `spatial_chunks`; unchanged without a size.

    A lazily opened dataset stays lazy: each chunk is read when it is computed.
    """
    if chunk_size is None:
        return obj
    return obj.chunk(spatial_chunks(obj, spatial_dims, chunk_size, extra))
