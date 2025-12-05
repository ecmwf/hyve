"""Unit tests for stat_calc."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.hydrostats.stat_calc import stat_calc


@pytest.fixture
def sim_discharge():
    """Sim: stations [A, B, C], times [Jan 1-3], values = station_idx + time_idx * 0.1."""
    return xr.Dataset(
        {"discharge": (["time", "station"], [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
        ])},
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "station": ["A", "B", "C"],
        },
    )


@pytest.fixture
def obs_discharge():
    """Obs: stations [B, C, D], times [Jan 2-4], values = sim + 0.5 at intersection."""
    return xr.Dataset(
        {"dis": (["time", "station"], [
            [1.6, 2.6, 3.6],
            [1.7, 2.7, 3.7],
            [1.8, 2.8, 3.8],
        ])},
        coords={
            "time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "station": ["B", "C", "D"],
        },
    )


def test_stat_calc_partial_overlap(sim_discharge, obs_discharge):
    """Test stat_calc computes MAE and correlation on station/time intersection."""
    config = {
        "sim": {
            "source": {"netcdf": {"path": "sim.nc"}},
            "var": "discharge",
            "coords": {"s": "station", "t": "time"},
        },
        "obs": {
            "source": {"netcdf": {"path": "obs.nc"}},
            "var": "discharge",
            "coords": {"s": "station", "t": "time"},
        },
        "stats": ["mae", "correlation"],
        "output": {"coords": {"s": "station", "t": "time"}},
    }

    sim_source = MagicMock()
    sim_source.to_xarray.return_value = sim_discharge
    obs_source = MagicMock()
    obs_source.to_xarray.return_value = obs_discharge

    with patch(
        "earthkit.data.from_source", side_effect=[sim_source, obs_source]
    ) as mock_from_source:
        result = stat_calc(config)

    assert mock_from_source.call_count == 2
    mock_from_source.assert_any_call("netcdf", path="sim.nc")
    mock_from_source.assert_any_call("netcdf", path="obs.nc")

    assert set(result["station"].values) == {"B", "C"}
    np.testing.assert_allclose(result["mae"].values, [0.5, 0.5])
    np.testing.assert_allclose(result["correlation"].values, [1.0, 1.0])
