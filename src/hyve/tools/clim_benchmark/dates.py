"""Pure date/index logic for building DOY-indexed sampling pools.

The module is deliberately self-contained and side-effect free so it can
be unit-tested without any xarray/dask machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hyve.tools.clim_benchmark.config import STRIDE_DAYS


# Cumulative days before the start of each month on the canonical leap-year
# calendar used by the custom DOY mapping (month 1 -> 0, month 2 -> 31, ...).
_DAYS_BEFORE_MONTH_LEAP = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
_DAYS_BEFORE_MONTH_NOLEAP = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def is_leap_year(year: int | np.ndarray) -> bool | np.ndarray:
    """Vectorised proleptic-Gregorian leap-year test."""
    year = np.asarray(year)
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


def doy_of(time_index: pd.DatetimeIndex) -> np.ndarray:
    """Return the custom day-of-year (1..366) for each timestamp.

    The mapping preserves Feb 29 as its own slot (DOY 60 on leap years).
    On non-leap years, Mar 1 maps to DOY 60 and Dec 31 maps to DOY 365.
    The duplication of non-leap Dec 31 into the DOY 366 pool is handled
    later in :func:`build_doy_pools`.
    """
    ts = pd.DatetimeIndex(time_index)
    years = ts.year.to_numpy()
    months = ts.month.to_numpy()
    days = ts.day.to_numpy()
    leap = is_leap_year(years)
    dbm_leap = np.array(_DAYS_BEFORE_MONTH_LEAP)
    dbm_noleap = np.array(_DAYS_BEFORE_MONTH_NOLEAP)
    dbm = np.where(leap, dbm_leap[months - 1], dbm_noleap[months - 1])
    return (dbm + days).astype(np.int64)


def infer_timestep(time_index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the uniform timestep of a time index.

    Raises
    ------
    ValueError
        If the index has fewer than 2 timestamps, is not strictly
        increasing, contains duplicates, or has non-uniform spacing.
    """
    ts = pd.DatetimeIndex(time_index)
    if len(ts) < 2:
        raise ValueError("time_index must contain at least 2 timestamps")
    diffs = np.diff(ts.values)
    if np.any(diffs <= np.timedelta64(0, "ns")):
        raise ValueError(
            "time_index must be strictly increasing with no duplicates"
        )

    for i in range(0, len(diffs)):
        if diffs[i] != diffs[0]:
            if i > 1 and i < len(diffs):
                print(f'time_index {i} has irregular spacing: {time_index[i-1]} {time_index[i]} {time_index[i+1]}')
            else:
                print(f'time_index {i} has irregular spacing: {time_index[i]}')
            raise ValueError("time_index has irregular spacing")

    return pd.Timedelta(diffs[0])


def _stride_offsets(window_days: int, stride: str) -> np.ndarray:
    """Return the signed day offsets relative to the target DOY.

    For ``stride="daily"`` the offsets cover every day in the centered
    window.  For ``weekly``/``monthly`` they are symmetric multiples of
    the stride step that fit within the window half-width.
    """
    if window_days % 2 == 0:
        raise ValueError("window_days must be odd")
    half = window_days // 2
    step = STRIDE_DAYS[stride]
    if step == 1:
        return np.arange(-half, half + 1, dtype=np.int64)
    k_max = half // step
    return np.arange(-k_max, k_max + 1, dtype=np.int64) * step


def _doy_to_indices(time_index: pd.DatetimeIndex) -> dict[int, np.ndarray]:
    """Map each DOY in 1..366 to the positional indices contributing to
    it.  Non-leap Dec 31 samples are added to BOTH DOY 365 and 366."""
    ts = pd.DatetimeIndex(time_index)
    doys = doy_of(ts)
    years = ts.year.to_numpy()
    months = ts.month.to_numpy()
    days = ts.day.to_numpy()

    buckets: dict[int, list[int]] = {d: [] for d in range(1, 367)}
    for i, d in enumerate(doys):
        buckets[int(d)].append(i)
    # Duplicate non-leap Dec 31 into DOY 366.
    non_leap_dec31 = np.where(
        (~is_leap_year(years)) & (months == 12) & (days == 31)
    )[0]
    for i in non_leap_dec31:
        buckets[366].append(int(i))

    return {d: np.asarray(sorted(ix), dtype=np.int64) for d, ix in buckets.items()}


def build_doy_pools(
    time_index: pd.DatetimeIndex,
    window_days: int,
    stride: str,
) -> dict[int, np.ndarray]:
    """Build a mapping ``{doy: indices}`` covering every DOY 1..366.

    ``indices`` are positional indices into ``time_index`` of every
    timestamp contributing to the climatological sample for that DOY
    under the given centered window and stride.

    The window wraps at the year boundary on a 366-day canonical
    calendar; the DOY-366 duplication rule is applied so every
    non-leap Dec 31 sample contributes to both DOY 365 and DOY 366.
    """
    if stride not in STRIDE_DAYS:
        raise ValueError(f"unknown stride {stride!r}")

    doy_to_idx = _doy_to_indices(time_index)
    offsets = _stride_offsets(window_days, stride)

    pools: dict[int, np.ndarray] = {}
    for target_doy in range(1, 367):
        parts: list[np.ndarray] = []
        for off in offsets:
            src_doy = ((target_doy - 1 + int(off)) % 366) + 1
            parts.append(doy_to_idx[src_doy])
        if parts:
            pools[target_doy] = np.concatenate(parts)
        else:  # pragma: no cover - _stride_offsets always returns >=1 offset
            pools[target_doy] = np.asarray([], dtype=np.int64)
    return pools


def split_pool_by_issue(
    pool_idx: np.ndarray,
    time_index: pd.DatetimeIndex,
    issue_frequency_hours: int,
) -> dict[int, np.ndarray]:
    """Split a DOY pool into ``{issue_hour: indices}`` slots.

    Each source timestamp is assigned to exactly one issue hour ``H``
    such that its hour-of-day lies in ``[H, H + issue_frequency_hours)``.
    Issue hours span ``range(0, 24, issue_frequency_hours)``.
    """
    if 24 % issue_frequency_hours != 0:
        raise ValueError(
            f"issue_frequency_hours must divide 24, got {issue_frequency_hours}"
        )
    ts = pd.DatetimeIndex(time_index)
    hours = ts[pool_idx].hour.to_numpy()
    issue_hours = range(0, 24, issue_frequency_hours)
    slots: dict[int, np.ndarray] = {}
    for ih in issue_hours:
        mask = (hours >= ih) & (hours < ih + issue_frequency_hours)
        slots[int(ih)] = pool_idx[mask]
    return slots


def doy_to_month_day(doy: int) -> tuple[int, int]:
    """Return the canonical ``(month, day)`` for a DOY in 1..366 using
    the leap-year calendar (DOY 60 == Feb 29)."""
    if doy < 1 or doy > 366:
        raise ValueError(f"doy must be in 1..366, got {doy}")
    # Use the 2020 leap year as the reference calendar.
    ref = pd.Timestamp("2020-01-01") + pd.Timedelta(days=doy - 1)
    return int(ref.month), int(ref.day)
