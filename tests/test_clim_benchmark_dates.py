"""Unit tests for `hyve.tools.clim_benchmark.dates` and `.config`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyve.tools.clim_benchmark.config import ClimConfig
from hyve.tools.clim_benchmark.dates import (
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
