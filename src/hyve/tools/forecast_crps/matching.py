# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Matching of station IDs and spatial coordinates between datasets.

Float coordinates are matched within a precision-aware tolerance (see
`hyve.tools.forecast_crps.snapping`); integer, text and bytes coordinates
(station IDs, integer y/x) are matched exactly. The first dataset is the
reference: `relabel` gives the other datasets its coordinate values.
"""

import numpy as np

from hyve.tools.forecast_crps.errors import DimensionMismatchError, mismatch_error_class
from hyve.tools.forecast_crps.snapping import check_unique, snap_floats, tolerance_text


def station_id_kind(values):
    """Kind of a coordinate (station ID) array.

    One of "integer", "float", "text" (str), "bytes", None (empty) or a dtype
    name. Bytes are never decoded, so they are a kind of their own. Object
    arrays are classified by the type of their elements.
    """
    values = np.asarray(values)
    if values.size == 0:
        return None
    if values.dtype.kind in "iu":
        return "integer"
    if values.dtype.kind == "f":
        return "float"
    if values.dtype.kind == "U":
        return "text"
    if values.dtype.kind == "S":
        return "bytes"
    if values.dtype.kind == "O":
        if all(isinstance(v, str) for v in values.ravel()):
            return "text"
        if all(isinstance(v, bytes) for v in values.ravel()):
            return "bytes"
        if all(isinstance(v, (int, np.integer)) for v in values.ravel()):
            return "integer"
    return str(values.dtype)


def _coordinate_words(dim):
    """Words for the values of `dim` in messages: (plural IDs, count, singular)."""
    if dim == "station":
        return "station IDs", "stations", "station ID"
    return f"{dim} values", f"{dim} values", f"{dim} value"


def _describe_stations(name, values, dim="station"):
    values = np.asarray(values)
    sample = values[:5].tolist()
    _, count, _ = _coordinate_words(dim)
    return f"{name} (dtype {values.dtype}, {values.size} {count}, e.g. {sample})"


def check_same_station_kind(a, b, a_name, b_name, dim="station"):
    """Raise a mismatch error if coordinates of `a` and `b` differ in kind.

    StationMismatchError for ``station``, GridMismatchError for other dims.

    Values are never converted: integer and text IDs are different things.
    """
    kind_a, kind_b = station_id_kind(a), station_id_kind(b)
    if kind_a is not None and kind_b is not None and kind_a != kind_b:
        ids, _, singular = _coordinate_words(dim)
        raise mismatch_error_class(dim)(
            "{} of different kinds ({} vs {}) cannot be matched: {} and {}."
            " Make the {} types consistent in the input datasets".format(
                ids,
                kind_a,
                kind_b,
                _describe_stations(a_name, a, dim),
                _describe_stations(b_name, b, dim),
                singular,
            )
        )


def _exact_positions(a, b, dim):
    """Positions ``(ia, ib)`` of the values shared exactly by `a` and `b`.

    Station IDs are returned sorted by value, other dimensions in the order of
    `a`. The comparison uses Python values, so no common dtype is imposed.
    """
    a_values = a.tolist()
    in_b = {}
    for j, value in enumerate(b.tolist()):
        in_b.setdefault(value, j)
    if dim == "station":
        in_a = {}
        for i, value in enumerate(a_values):
            if value in in_b:
                in_a.setdefault(value, i)
        keys = sorted(in_a)
        ia = [in_a[key] for key in keys]
    else:
        ia = [i for i, value in enumerate(a_values) if value in in_b]
        keys = [a_values[i] for i in ia]
    ib = [in_b[key] for key in keys]
    return np.array(ia, dtype=np.intp), np.array(ib, dtype=np.intp)


def match_values(a, b, a_name, b_name, dim="station", coord_tolerance=None):
    """Positions ``(ia, ib)`` such that ``a[ia]`` and ``b[ib]`` are the same points.

    `a` is the reference. Float axes (both sides) match within tolerance and
    keep the order of `a`; everything else matches exactly. Station IDs come
    out sorted. Raises StationMismatchError (``station``) or GridMismatchError
    (other dimensions) if the kinds differ, nothing matches or the tolerance
    is unsafe, or if either axis holds duplicate values.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    check_same_station_kind(a, b, a_name, b_name, dim)
    description = "{} and {}".format(
        _describe_stations(a_name, a, dim), _describe_stations(b_name, b, dim)
    )
    suffix = ""
    if a.dtype.kind == "f" and b.dtype.kind == "f" and a.size and b.size:
        context = f"{dim} values of {description}"
        ia, ib = snap_floats(a, b, dim, a_name, b_name, context, coord_tolerance)
        if dim == "station":
            order = np.argsort(a[ia], kind="stable")
            ia, ib = ia[order], ib[order]
        suffix = ", tolerance: " + tolerance_text(a, b, coord_tolerance)
    else:
        check_unique(a, _describe_stations(a_name, a, dim), dim)
        check_unique(b, _describe_stations(b_name, b, dim), dim)
        ia, ib = _exact_positions(a, b, dim)
    if ia.size == 0:
        _, count, _ = _coordinate_words(dim)
        raise mismatch_error_class(dim)(
            f"no {count} in common between {description}{suffix}"
        )
    return ia, ib


def common_stations(a, b, a_name, b_name, dim="station", coord_tolerance=None):
    """Coordinate values of `dim` present in both `a` and `b`, as `a` has them.

    Station IDs are returned sorted; values of any other dimension keep the
    order they have in `a`, so a descending latitude axis stays descending.
    Float values match within a tolerance (see `match_values`) and are
    returned with the values of `a`; other kinds match exactly, with no
    common dtype imposed (int64 and uint64 IDs are compared exactly, and bytes
    never equal str).

    Raises StationMismatchError (``station``) or GridMismatchError (other
    dimensions) if the values are of different kinds (integer vs float vs text
    vs bytes) or if the datasets share no value.
    """
    ia, _ = match_values(a, b, a_name, b_name, dim, coord_tolerance)
    return np.asarray(a)[ia].tolist()


def match_coords(a, b, spatial_dims, a_name, b_name, coord_tolerance=None):
    """Mapping of each spatial dimension to the positions ``(ia, ib)`` it shares.

    `a` is the reference (see `match_values`). Raises DimensionMismatchError if
    a spatial dimension is missing from either dataset.
    """
    for obj, name in ((a, a_name), (b, b_name)):
        missing = [dim for dim in spatial_dims if dim not in obj.dims]
        if missing:
            raise DimensionMismatchError(
                "{} has no {} dimension (its dimensions are {}). Set"
                " --spatial_dims to the spatial dimensions of the input"
                " datasets".format(
                    name, ", ".join(repr(d) for d in missing), list(obj.dims)
                )
            )
    return {
        dim: match_values(
            a[dim].values, b[dim].values, a_name, b_name, dim, coord_tolerance
        )
        for dim in spatial_dims
    }


def common_coords(a, b, spatial_dims, a_name, b_name, coord_tolerance=None):
    """Values of each spatial dimension present in both `a` and `b`.

    Returns a mapping of dimension name to the shared values, as `a` has them.
    Raises DimensionMismatchError if a spatial dimension is missing from
    either dataset, and StationMismatchError or GridMismatchError if the
    datasets share no value along a dimension.
    """
    matches = match_coords(a, b, spatial_dims, a_name, b_name, coord_tolerance)
    return {
        dim: np.asarray(a[dim].values)[ia].tolist() for dim, (ia, _) in matches.items()
    }


def narrow(reference, matches):
    """`reference` restricted to the matched points, in matched order."""
    return reference.isel({dim: ia for dim, (ia, _) in matches.items()})


def relabel(obj, matches, reference):
    """`obj` restricted to the matched points and labelled with `reference` values.

    `matches` comes from `match_coords` with `reference` as its first dataset.
    Label-based maths then aligns `obj` exactly with `narrow(reference, ...)`.
    """
    for dim, (ia, ib) in matches.items():
        values = np.asarray(reference[dim].values)[ia]
        obj = obj.isel({dim: ib}).assign_coords({dim: (dim, values, obj[dim].attrs)})
    return obj


def check_covers_reference(matches, ref_coords, name, ref_name):
    """Raise a mismatch error if `name` lacks a point of the reference grid.

    `matches` maps each spatial dimension to the positions ``(ia, ib)`` of the
    reference values (``ia``) matched in the file; `ref_coords` holds the
    reference values. StationMismatchError for ``station``, else
    GridMismatchError.
    """
    for dim, ref_values in ref_coords.items():
        covered = np.zeros(len(ref_values), dtype=bool)
        covered[matches[dim][0]] = True
        if not covered.all():
            missing = ref_values[~covered].tolist()
            raise mismatch_error_class(dim)(
                "{} lacks {} of the {} {!r} values of {} (e.g. {}). Every"
                " reforecast file must cover the grid of the reference file"
                " (the latest one)".format(
                    name, len(missing), len(ref_values), dim, ref_name, missing[:5]
                )
            )
