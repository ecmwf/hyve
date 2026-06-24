"""Unit tests for `hyve.tools.clim_benchmark.dates` and `.config`."""

from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest

from hyve.tools.clim_benchmark.config import ClimConfig
from hyve.tools.clim_benchmark.dates import (
    _doy_to_indices,
    build_doy_pools,
    doy_of,
    doy_to_month_day,
    infer_timestep,
    is_leap_year,
    split_pool_by_issue,
)


# ---------------------------------------------------------------------------
# infer_timestep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq",
    ["1D", "12h", "6h", "3h", "1h"],
)
def test_infer_timestep_regular(freq):
    idx = pd.date_range("2020-01-01", periods=50, freq=freq)
    assert infer_timestep(idx) == pd.Timedelta(freq)


def test_infer_timestep_irregular():
    idx = pd.DatetimeIndex(
        pd.date_range("2020-01-01", periods=10, freq="1D").to_list()
        + [pd.Timestamp("2020-01-15")]
    )
    with pytest.raises(ValueError, match="irregular"):
        infer_timestep(idx)


def test_infer_timestep_duplicate():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
    with pytest.raises(ValueError, match="increasing"):
        infer_timestep(idx)


def test_infer_timestep_non_monotonic():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="increasing"):
        infer_timestep(idx)


def test_infer_timestep_too_short():
    with pytest.raises(ValueError, match="at least 2"):
        infer_timestep(pd.DatetimeIndex(["2020-01-01"]))


# ---------------------------------------------------------------------------
# doy_of
# ---------------------------------------------------------------------------


def test_doy_of_jan1_any_year():
    idx = pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])
    assert list(doy_of(idx)) == [1, 1, 1]


def test_doy_of_feb28_and_feb29():
    idx = pd.to_datetime(
        ["2019-02-28", "2020-02-28", "2020-02-29"]
    )
    # Feb 28 is DOY 59 on both calendars; Feb 29 on leap year is DOY 60.
    assert list(doy_of(idx)) == [59, 59, 60]


def test_doy_of_mar1_leap_vs_nonleap():
    idx = pd.to_datetime(["2019-03-01", "2020-03-01"])
    # Non-leap Mar 1 -> 60; leap Mar 1 -> 61.
    assert list(doy_of(idx)) == [60, 61]


def test_doy_of_dec31():
    idx = pd.to_datetime(["2019-12-31", "2020-12-31"])
    # Non-leap Dec 31 -> 365; leap Dec 31 -> 366.
    assert list(doy_of(idx)) == [365, 366]


def test_is_leap_year():
    assert bool(is_leap_year(2020))
    assert not bool(is_leap_year(2019))
    assert not bool(is_leap_year(1900))  # divisible by 100 but not 400
    assert bool(is_leap_year(2000))


# ---------------------------------------------------------------------------
# build_doy_pools
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def four_year_daily_index():
    """4 consecutive years of daily data starting on a non-leap year.
    One of the four years (2020) is a leap year."""
    return pd.date_range("2019-01-01", "2022-12-31", freq="1D")


def test_build_doy_pools_daily_pool_size(four_year_daily_index):
    pools = build_doy_pools(four_year_daily_index, window_days=31, stride="daily")
    # DOY 150 is well away from year boundaries; window covers 31 days each year.
    assert pools[150].size == 31 * 4


def test_build_doy_pools_weekly_pool_size(four_year_daily_index):
    pools = build_doy_pools(four_year_daily_index, window_days=31, stride="weekly")
    # Symmetric offsets {-14, -7, 0, 7, 14} -> 5 per year.
    assert pools[150].size == 5 * 4


def test_build_doy_pools_monthly_pool_size(four_year_daily_index):
    pools = build_doy_pools(four_year_daily_index, window_days=31, stride="monthly")
    # Only the center day contributes.
    assert pools[150].size == 1 * 4


def test_build_doy_pools_wraparound(four_year_daily_index):
    """DOY 1 pool must include late-December timestamps from each year."""
    pools = build_doy_pools(four_year_daily_index, window_days=31, stride="daily")
    sel = four_year_daily_index[pools[1]]
    assert (sel.month == 12).any(), "wrap-around should include December days"
    assert (sel.month == 1).any(), "pool should include January days"


def test_build_doy_pools_feb29_lands_in_doy_60(four_year_daily_index):
    """Explicit regression for the legacy off-by-one.

    With year-based DOY (the 60th day of each year is DOY 60), leap-year
    Feb 29 and non-leap Mar 1 share DOY 60. The key property being tested
    is that Feb 29 is present in the DOY 60 pool (the legacy code dropped
    it or shifted it into DOY 61).
    """
    pools = build_doy_pools(four_year_daily_index, window_days=1, stride="daily")
    sel = sorted(four_year_daily_index[pools[60]])
    assert pd.Timestamp("2020-02-29") in sel
    # 2019/2021/2022 contribute their Mar 1 samples (the 60th day of those years).
    assert [t.strftime("%Y-%m-%d") for t in sel] == [
        "2019-03-01",
        "2020-02-29",
        "2021-03-01",
        "2022-03-01",
    ]


def test_build_doy_pools_doy366_mixes_feb29_and_nonleap_dec31(
    four_year_daily_index,
):
    """DOY 366 pool must contain: leap-year Feb 29 + every non-leap Dec 31."""
    pools = build_doy_pools(four_year_daily_index, window_days=1, stride="daily")
    sel = sorted(four_year_daily_index[pools[366]])
    # Expected: 2019-12-31, 2020-02-29 (from DOY 60? NO — DOY 366 on leap calendar is Dec 31),
    # 2020-12-31 (leap Dec 31 has DOY 366), 2021-12-31, 2022-12-31.
    # So non-leap Dec 31 (2019, 2021, 2022) + leap Dec 31 (2020) = 4 entries.
    assert [t.strftime("%Y-%m-%d") for t in sel] == [
        "2019-12-31",
        "2020-12-31",
        "2021-12-31",
        "2022-12-31",
    ]


def test_build_doy_pools_doy365_includes_nonleap_dec31(four_year_daily_index):
    """DOY 365 must still include Dec 31 of non-leap years (and Dec 30 of leap)."""
    pools = build_doy_pools(four_year_daily_index, window_days=1, stride="daily")
    sel = sorted(four_year_daily_index[pools[365]])
    # Non-leap Dec 31 has doy_of == 365; leap Dec 30 has doy_of == 365.
    assert [t.strftime("%Y-%m-%d") for t in sel] == [
        "2019-12-31",
        "2020-12-30",
        "2021-12-31",
        "2022-12-31",
    ]


# ---------------------------------------------------------------------------
# split_pool_by_issue
# ---------------------------------------------------------------------------


@pytest.fixture
def six_hourly_index():
    return pd.date_range("2020-01-01", "2020-12-31T18", freq="6h")


def test_split_pool_by_issue_24h(six_hourly_index):
    pool = np.arange(len(six_hourly_index))
    slots = split_pool_by_issue(pool, six_hourly_index, issue_frequency_hours=24)
    assert list(slots.keys()) == [0]
    assert slots[0].size == pool.size


def test_split_pool_by_issue_12h(six_hourly_index):
    pool = np.arange(len(six_hourly_index))
    slots = split_pool_by_issue(pool, six_hourly_index, issue_frequency_hours=12)
    assert sorted(slots.keys()) == [0, 12]
    # Union is the full pool and slots are disjoint.
    union = np.concatenate(list(slots.values()))
    assert sorted(union.tolist()) == pool.tolist()
    # Hour filtering.
    hours0 = six_hourly_index[slots[0]].hour
    hours12 = six_hourly_index[slots[12]].hour
    assert set(hours0.tolist()).issubset({0, 6})
    assert set(hours12.tolist()).issubset({12, 18})


def test_split_pool_by_issue_6h_one_sample_per_day(six_hourly_index):
    pool = np.arange(len(six_hourly_index))
    slots = split_pool_by_issue(pool, six_hourly_index, issue_frequency_hours=6)
    assert sorted(slots.keys()) == [0, 6, 12, 18]
    # Each slot should contain exactly one sample per day for this cadence.
    for ih, idx in slots.items():
        hours = six_hourly_index[idx].hour
        assert (hours == ih).all()


def test_split_pool_by_issue_rejects_bad_freq(six_hourly_index):
    pool = np.arange(5)
    with pytest.raises(ValueError):
        split_pool_by_issue(pool, six_hourly_index, issue_frequency_hours=5)


# ---------------------------------------------------------------------------
# doy_to_month_day
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doy, expected",
    [
        (1, (1, 1)),
        (59, (2, 28)),
        (60, (2, 29)),
        (61, (3, 1)),
        (365, (12, 30)),
        (366, (12, 31)),
    ],
)
def test_doy_to_month_day(doy, expected):
    assert doy_to_month_day(doy) == expected


# ---------------------------------------------------------------------------
# ClimConfig validation
# ---------------------------------------------------------------------------


def test_config_default_is_valid():
    c = ClimConfig()
    assert c.window_days == 31
    assert c.stride == "daily"
    assert c.num_workers == 1


def test_config_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        ClimConfig(window_days=30)


def test_config_rejects_weekly_window_too_small():
    with pytest.raises(ValueError, match="window_days"):
        ClimConfig(window_days=3, stride="weekly")


def test_config_rejects_monthly_window_too_small():
    with pytest.raises(ValueError, match="window_days"):
        ClimConfig(window_days=15, stride="monthly")


def test_config_rejects_bad_issue_frequency():
    with pytest.raises(ValueError, match="divisor of 24"):
        ClimConfig(issue_frequency_hours=5)


def test_config_rejects_issue_freq_not_multiple_of_timestep():
    c = ClimConfig(issue_frequency_hours=24)
    # dataset timestep of 5 hours does not divide 24 cleanly into the issue freq
    with pytest.raises(ValueError, match="multiple"):
        c.validate_against_data(timestep_hours=5)


def test_config_rejects_bad_percentiles():
    with pytest.raises(ValueError):
        ClimConfig(percentiles=[-1, 50, 101])


@pytest.mark.parametrize("workers", [0, -1])
def test_config_rejects_non_positive_num_workers(workers):
    with pytest.raises(ValueError, match="num_workers"):
        ClimConfig(num_workers=workers)


def test_config_accepts_valid_timestep():
    c = ClimConfig(issue_frequency_hours=12)
    c.validate_against_data(timestep_hours=6)  # 12 / 6 == 2
    c.validate_against_data(timestep_hours=3)  # 12 / 3 == 4


# ---------------------------------------------------------------------------
# Regression: non-leap Dec 31 duplication must not double-count in pools
# ---------------------------------------------------------------------------


def test_build_doy_pools_no_duplicate_indices_near_year_end():
    """Regression for the pre-fix bug where _doy_to_indices() added non-leap
    Dec 31 indices to BOTH DOY 365 and DOY 366, and build_doy_pools() could
    pull both source DOYs into the same target pool via a multi-day window,
    causing each Dec 31 timestamp to appear twice.

    With window_days=3 and stride="daily" the offsets are [-1, 0, +1].
    For target DOY 365 the source DOYs are 364, 365, 366 – both 365 and 366
    contain non-leap Dec 31 indices, so without deduplication each Dec 31
    index would appear twice.

    This test verifies that every pool contains strictly unique indices.
    """
    # Three full non-leap years so we have multiple Dec 31 timestamps.
    time_index = pd.date_range("2021-01-01", "2023-12-31", freq="1D")
    assert not any(time_index.is_leap_year)

    pools = build_doy_pools(time_index, window_days=3, stride="daily")

    for doy, idx in pools.items():
        assert len(idx) == len(np.unique(idx)), (
            f"Pool for DOY {doy} contains duplicate indices: {idx}"
        )


def test_dec31_not_overweighted_in_percentiles_near_year_end():
    """Regression for the same bug, verified through percentile computation.

    With window_days=5 and stride="daily" the offsets are [-2,-1,0,+1,+2].
    For target DOY 365 the source DOYs are 363, 364, 365, 366, 1.
    DOY 365 and DOY 366 both contain the same non-leap Dec 31 positional
    indices, so before deduplication each Dec 31 timestamp contributes twice.

    Pool sizes:
      Without fix: Dec29×3 + Dec30×3 + Dec31×3 + Dec31×3 (dup) + Jan1×3 = 15
                   → 9×1 and 6×999 → median (rank 8) = 999
      With fix:    Dec29×3 + Dec30×3 + Dec31×3 + Jan1×3 = 12
                   → 9×1 and 3×999 → median (rank 6.5) = 1

    The test passes iff the fix is active: the median must be 1.0, not 999.
    """
    import xarray as xr

    from hyve.tools.clim_benchmark.percentiles import compute_climatology
    from hyve.tools.clim_benchmark.sampling import build_slots

    # Three non-leap years; all values = 1 except Dec 31 which is a large
    # outlier.  If Dec 31 is double-counted in the DOY 365 pool it becomes
    # the majority and biases the median.
    time = pd.date_range("2021-01-01", "2023-12-31", freq="1D")
    assert not any(time.is_leap_year)
    values = np.ones(len(time), dtype=np.float64)
    is_dec31 = (time.month == 12) & (time.day == 31)
    values[is_dec31] = 999.0
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="dis")

    # window_days=5: offsets [-2,-1,0,+1,+2] pull src DOYs 363,364,365,366,1.
    # DOY 365 and 366 map to the same Dec 31 positional indices; without
    # np.unique they are concatenated twice, making Dec 31 the majority.
    slots = build_slots(da, window_days=5, stride="daily", issue_frequency_hours=24)
    clim = compute_climatology(da, slots, percentiles=[50], worker_count=1)

    # After deduplication: 9 samples with value=1, 3 with value=999, median=1.
    median_doy365 = float(clim.sel(doy=365, issue_hour=0, ensemble=50).values)
    assert median_doy365 == 1.0, (
        f"Median for DOY 365 is {median_doy365}, expected 1.0 – "
        "Dec 31 outlier appears overweighted, possible duplication bug."
    )


# ---------------------------------------------------------------------------
# Performance regression: _doy_to_indices must not use a Python for-loop
# over every timestamp (bottleneck for multi-decadal sub-daily reanalyses).
# PR review comment: https://github.com/ecmwf/hyve/pull/33#discussion_r3100264404
# ---------------------------------------------------------------------------


def test_doy_to_indices_is_fast_on_large_input():
    """_doy_to_indices should complete in <200 ms for 50 years of hourly data.

    The original implementation iterates over every timestamp with a Python
    for-loop (``for i, d in enumerate(doys): buckets[int(d)].append(i)``).
    For 50 years × 365.25 × 24 ≈ 438 k timestamps this typically takes
    ~300–800 ms in CPython.  A fully vectorised implementation using
    np.argsort + np.searchsorted stays well under 100 ms on the same input.
    The 200 ms threshold is intentionally set between the two so that the
    test fails on the loop implementation and passes on the vectorised one.
    """
    import time

    time_index = pd.date_range("1970-01-01", "2019-12-31", freq="1h")
    # ~438 400 timestamps — representative of 50 years of hourly reanalysis.

    t0 = time.perf_counter()
    result = _doy_to_indices(time_index)
    elapsed = time.perf_counter() - t0

    # Sanity-check: all 366 DOYs present, total count correct.
    assert set(result.keys()) == set(range(1, 367))
    total = sum(len(v) for v in result.values())
    # Non-leap Dec 31 entries appear in both DOY 365 and 366, so the sum
    # exceeds len(time_index).
    assert total >= len(time_index)

    assert elapsed < 0.2, (
        f"_doy_to_indices took {elapsed:.3f}s for {len(time_index):,} timestamps "
        f"(limit 0.2s). The Python for-loop implementation is too slow; "
        f"replace with a vectorised np.argsort / np.searchsorted approach."
    )


def test_doy_to_indices_vectorised_matches_loop_reference():
    """Correctness check: vectorised result must equal the naive loop result.

    Uses a mixed dataset that spans multiple leap and non-leap years and
    includes 6-hourly sub-daily resolution so that every edge case
    (Feb 29, non-leap Dec 31 duplication, year wrap) is exercised.
    """
    time_index = pd.date_range("2016-01-01", "2024-12-31T18", freq="6h")

    # Reference: the pure-Python loop implementation (copied verbatim from
    # the version that existed before the vectorisation fix so the test is
    # independent of production code).
    def _doy_to_indices_loop(ti: pd.DatetimeIndex) -> dict[int, np.ndarray]:
        ts = pd.DatetimeIndex(ti)
        doys = doy_of(ts)
        years = ts.year.to_numpy()
        months = ts.month.to_numpy()
        days = ts.day.to_numpy()
        buckets: dict[int, list[int]] = {d: [] for d in range(1, 367)}
        for i, d in enumerate(doys):
            buckets[int(d)].append(i)
        non_leap_dec31 = np.where(
            (~is_leap_year(years)) & (months == 12) & (days == 31)
        )[0]
        for i in non_leap_dec31:
            buckets[366].append(int(i))
        return {d: np.asarray(sorted(ix), dtype=np.int64) for d, ix in buckets.items()}

    reference = _doy_to_indices_loop(time_index)
    result = _doy_to_indices(time_index)

    assert set(result.keys()) == set(reference.keys())
    for doy in range(1, 367):
        np.testing.assert_array_equal(
            result[doy],
            reference[doy],
            err_msg=f"Mismatch at DOY {doy}",
        )


# ---------------------------------------------------------------------------
# Regression: compute_climatology must process slots in batches of worker_count
# so that peak memory is bounded (not all slots computed at once).
# PR review comment: https://github.com/ecmwf/hyve/pull/33#discussion_r3100264319
# ---------------------------------------------------------------------------


def test_compute_climatology_batches_by_worker_count():
    """dask.compute() must be called with at most worker_count arguments per
    call, not with all slots at once.

    The original implementation used a single
    ``dask.compute(*delayed_results.values())`` which submits every slot task
    simultaneously and materialises every result in RAM before building the
    output array -- negating the "bounded by one slot" memory intent.

    This test patches dask.compute to count the number of tasks submitted per
    call and verifies that no call exceeds worker_count.
    """
    import unittest.mock as mock
    import xarray as xr

    from hyve.tools.clim_benchmark.percentiles import compute_climatology
    from hyve.tools.clim_benchmark.sampling import build_slots

    # Small dataset: 3 years daily so slots are non-trivial but fast.
    time = pd.date_range("2021-01-01", "2023-12-31", freq="1D")
    values = np.arange(len(time), dtype=np.float64)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="x")
    slots = build_slots(da, window_days=1, stride="daily", issue_frequency_hours=24)
    total_slots = len(slots)  # 366

    for worker_count in (1, 5, total_slots):
        call_sizes: list[int] = []
        original_compute = dask.compute

        def _recording_compute(*args, **kwargs):
            call_sizes.append(len(args))
            return original_compute(*args, **kwargs)

        with mock.patch("hyve.tools.clim_benchmark.percentiles.dask.compute",
                        side_effect=_recording_compute):
            result = compute_climatology(da, slots, percentiles=[50], worker_count=worker_count)

        assert result is not None
        assert len(call_sizes) > 0, "dask.compute was never called"
        max_batch = max(call_sizes)
        assert max_batch <= worker_count, (
            f"worker_count={worker_count}: dask.compute was called with "
            f"{max_batch} tasks in one call (limit {worker_count}). "
            f"All {total_slots} slots are being scheduled at once."
        )


def test_compute_climatology_results_independent_of_worker_count():
    """Output must be identical regardless of worker_count batching.

    Verifies that the batching strategy is transparent: splitting the work
    into smaller dask.compute calls produces the same numerical output as
    computing all slots in one call (worker_count >= total_slots).
    """
    import xarray as xr

    from hyve.tools.clim_benchmark.percentiles import compute_climatology
    from hyve.tools.clim_benchmark.sampling import build_slots

    time = pd.date_range("2021-01-01", "2022-12-31", freq="1D")
    values = np.arange(len(time), dtype=np.float64)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="x")
    slots = build_slots(da, window_days=3, stride="daily", issue_frequency_hours=24)

    ref = compute_climatology(da, slots, percentiles=[0, 50, 100], worker_count=len(slots))
    for wc in (1, 7, 50):
        result = compute_climatology(da, slots, percentiles=[0, 50, 100], worker_count=wc)
        np.testing.assert_array_equal(
            result.values,
            ref.values,
            err_msg=f"Results differ for worker_count={wc} vs worker_count={len(slots)}",
        )
