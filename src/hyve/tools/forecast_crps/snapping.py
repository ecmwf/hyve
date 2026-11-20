# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Tolerant matching of two float coordinate axes (snapping).

The nearest-value search (argsort + searchsorted, which works for ascending,
descending and unordered axes), the default
tolerance is relative to the float precision of the axes.
"""

import numpy as np

from hyve.tools.forecast_crps.errors import mismatch_error_class

RTOL_EPS_FACTOR = 4


def relative_tolerance(a, b):
    """4 x machine epsilon of the coarser (larger eps) float dtype of `a` and `b`."""
    return RTOL_EPS_FACTOR * max(np.finfo(a.dtype).eps, np.finfo(b.dtype).eps)


def tolerance_text(a, b, coord_tolerance=None):
    """Words describing the tolerance used to match `a` and `b`."""
    if coord_tolerance is not None:
        return f"absolute tolerance {coord_tolerance:g} (--coord_tolerance)"
    coarser = max(a.dtype, b.dtype, key=lambda dtype: np.finfo(dtype).eps)
    return "relative tolerance {:.3g} (4 x eps of {})".format(
        relative_tolerance(a, b), coarser
    )


def check_unique(values, description, dim):
    """Raise a mismatch error if `values` of `dim` repeat (any dtype).

    A repeated label cannot be matched to one point: matching would silently
    pick one of the cells. `description` names the dataset in the message.
    StationMismatchError for ``station``, GridMismatchError otherwise.
    """
    seen, repeated = set(), []
    for item in np.asarray(values).tolist():
        if item in seen and item not in repeated:
            repeated.append(item)
        seen.add(item)
    if repeated:
        raise mismatch_error_class(dim)(
            f"{description} has duplicate {dim} values ({len(repeated)} repeated,"
            f" e.g. {repeated[:5]}). Remove the duplicates from the dataset"
        )


def _min_spacing(values):
    """Smallest gap between values of an axis; infinite for a singleton axis."""
    if values.size < 2:
        return np.inf
    return np.diff(np.sort(values)).min()


def snap_floats(a, b, dim, a_name, b_name, context, coord_tolerance=None):
    """Positions ``(ia, ib)`` such that ``a[ia]`` matches ``b[ib]`` within tolerance.

    `a` is the reference axis: ``ia`` is increasing, so the order of `a` is
    kept. Two values match when ``|a - b| <= tol`` with ``tol = rtol *
    max(|a|, |b|)`` (see `relative_tolerance`), or ``tol = coord_tolerance``
    when given. Raises the error class of `dim` (GridMismatchError except for
    ``station``) for NaN/inf or duplicate values, an ambiguous tolerance (at
    least half the smallest spacing of either axis) and two values of `a`
    snapping to one value of `b`. `context` describes both datasets for the
    messages. Empty ``ia`` and ``ib`` mean that nothing matches.
    """
    text = tolerance_text(a, b, coord_tolerance)

    def fail(problem):
        raise mismatch_error_class(dim)(
            f"{problem}: {context}. Tolerance: {text}. Check the {dim} coordinates"
            " or set --coord_tolerance"
        )

    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    for values, name in ((a64, a_name), (b64, b_name)):
        if not np.isfinite(values).all():
            fail(f"{name} has NaN or infinite {dim} values")
        if np.unique(values).size != values.size:
            fail(f"{name} has duplicate {dim} values")

    order = np.argsort(b64, kind="stable")
    sorted_b = b64[order]
    # nearest value of b for every value of a
    pos = np.searchsorted(sorted_b, a64)
    right = np.clip(pos, 0, sorted_b.size - 1)
    left = np.clip(pos - 1, 0, sorted_b.size - 1)
    pick_left = np.abs(sorted_b[left] - a64) < np.abs(sorted_b[right] - a64)
    nearest = np.where(pick_left, left, right)
    distance = np.abs(sorted_b[nearest] - a64)
    if coord_tolerance is None:
        rtol = relative_tolerance(a, b)
        tol = rtol * np.maximum(np.abs(a64), np.abs(sorted_b[nearest]))
    else:
        tol = coord_tolerance
    matched = distance <= tol
    ia = np.flatnonzero(matched)
    ib = order[nearest[matched]]
    if np.unique(ib).size != ib.size:
        fail(f"two {dim} values of {a_name} snap to the same {dim} value of {b_name}")

    if coord_tolerance is None:
        largest = rtol * max(np.abs(a64).max(), np.abs(b64).max())
    else:
        largest = coord_tolerance
    if 2 * largest >= min(_min_spacing(a64), _min_spacing(b64)):
        fail(
            f"the tolerance of the {dim} axes is ambiguous: it is at least half"
            " the smallest spacing"
        )
    return ia, ib
