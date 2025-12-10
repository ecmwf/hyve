import numpy as np
import pytest

from hyve.hydrostats.stat_calc import stat_calc


def _wrap_lod(dicts: list[dict]) -> dict:
    """Simple wrapper to turn a list of dicts into an earthkit-data source config."""
    return {"list-of-dicts": {"list_of_dicts": dicts}}


@pytest.fixture
def simulation_source_config():
    """earthkit-data list-of-dicts source config for simulation data."""
    return _wrap_lod(
        [
            {"date": 20240101, "time": 0, "number": 1, "param": "dis", "values": [1.0]},
            {"date": 20240102, "time": 0, "number": 1, "param": "dis", "values": [2.0]},
            {"date": 20240101, "time": 0, "number": 2, "param": "dis", "values": [1.0]},
            {"date": 20240102, "time": 0, "number": 2, "param": "dis", "values": [2.0]},
            # Stations that only exist in sim should be ignored
            {"date": 20240101, "time": 0, "number": 3, "param": "dis", "values": [8.0]},
            {"date": 20240102, "time": 0, "number": 3, "param": "dis", "values": [9.0]},
            # Times that only exist in sim should be ignored
            {"date": 20240101, "time": 6, "number": 1, "param": "dis", "values": [7.0]},
            {"date": 20240101, "time": 6, "number": 2, "param": "dis", "values": [4.0]},
            {"date": 20240101, "time": 6, "number": 3, "param": "dis", "values": [5.0]},
        ]
    )


@pytest.fixture
def observation_source_config():
    """earthkit-data list-of-dicts source config for observation data."""
    return _wrap_lod(
        [
            {"date": 20240101, "time": 0, "number": 1, "param": "d", "values": [1.0]},
            {"date": 20240102, "time": 0, "number": 1, "param": "d", "values": [2.0]},
            {"date": 20240101, "time": 0, "number": 2, "param": "d", "values": [2.0]},
            {"date": 20240102, "time": 0, "number": 2, "param": "d", "values": [1.0]},
            # Stations that only exist in obs should be ignored
            {"date": 20240101, "time": 0, "number": 4, "param": "d", "values": [9.0]},
            {"date": 20240102, "time": 0, "number": 4, "param": "d", "values": [9.0]},
            # Times that only exist in obs should be ignored
            {"date": 20240102, "time": 6, "number": 1, "param": "d", "values": [1.0]},
            {"date": 20240102, "time": 6, "number": 2, "param": "d", "values": [5.0]},
            {"date": 20240102, "time": 6, "number": 4, "param": "d", "values": [9.0]},
        ]
    )


def test_stat_calc(simulation_source_config, observation_source_config):
    """Test stat_calc with synthetic data.

    Simulated and observed data each have two stations and two times in common.
    Station 1 has identical values in sim and obs, leading to MAE=0 and correlation=1.
    Station 2 has inverted values in sim and obs, leading to MAE=1 and correlation=-1.

    The test checks that only the common stations and times are considered, and that
    the calculated statistics match the expected results.

    Albeit a bit hacky, this test uses the list-of-dicts data source for
    simplicity and uses the ensemble member dimension to represent stations.
    """
    config = {
        "sim": {
            "source": simulation_source_config,
            "coords": {"s": "number", "t": "forecast_reference_time"},
        },
        "obs": {
            "source": observation_source_config,
            "to_xarray_options": {"time_dim_mode": "valid_time"},
            "coords": {"s": "number", "t": "valid_time"},
        },
        "stats": ["mae", "correlation"],
        "output": {"coords": {"s": "station", "t": "time"}},
    }

    result = stat_calc(config)

    assert set(result["station"].values) == {1, 2}
    np.testing.assert_allclose(result["mae"].values.squeeze(), [0.0, 1.0])
    np.testing.assert_allclose(result["correlation"].values.squeeze(), [1.0, -1.0])


def test_stat_calc_with_nans():
    """Test common NaN handling scenarios in stat_calc.

    NOTE: This test documents current behavior, not necessarily desired behavior.

    This test creates simulated and observed data for three stations over two time points,
    with NaN values in various configurations:
    - Station 1: Overlapping valid data points, only one data point is NaN in obs
    - Station 2: No overlapping valid data points (one NaN in sim, one NaN in obs)
    - Station 3: One station with only NaN values in sim and one nan in obs
    """
    sim_source = _wrap_lod(
        [
            {"date": 20240101, "time": 0, "number": 1, "param": "q", "values": [1.0]},
            {"date": 20240102, "time": 0, "number": 1, "param": "q", "values": [-1.0]},
            {"date": 20240101, "time": 0, "number": 2, "param": "q", "values": [1.0]},
            {
                "date": 20240102,
                "time": 0,
                "number": 2,
                "param": "q",
                "values": [np.nan],
            },
            {
                "date": 20240101,
                "time": 0,
                "number": 3,
                "param": "q",
                "values": [np.nan],
            },
            {
                "date": 20240102,
                "time": 0,
                "number": 3,
                "param": "q",
                "values": [np.nan],
            },
        ]
    )
    obs_source = _wrap_lod(
        [
            {
                "date": 20240101,
                "time": 0,
                "number": 1,
                "param": "q",
                "values": [1.0],
            },
            {
                "date": 20240102,
                "time": 0,
                "number": 1,
                "param": "q",
                "values": [np.nan],
            },
            {
                "date": 20240101,
                "time": 0,
                "number": 2,
                "param": "q",
                "values": [np.nan],
            },
            {"date": 20240102, "time": 0, "number": 2, "param": "q", "values": [2.0]},
            {
                "date": 20240101,
                "time": 0,
                "number": 3,
                "param": "q",
                "values": [np.nan],
            },
            {"date": 20240102, "time": 0, "number": 3, "param": "q", "values": [1.0]},
        ]
    )
    config = {
        "sim": {
            "source": sim_source,
            "coords": {"s": "number", "t": "forecast_reference_time"},
        },
        "obs": {
            "source": obs_source,
            "coords": {"s": "number", "t": "forecast_reference_time"},
        },
        "stats": ["mae", "br"],
        "output": {"coords": {"s": "station", "t": "time"}},
    }

    result = stat_calc(config)

    # Station 1: sim=[1.0, -1.0], obs=[1.0, nan] -> mae=0.0, br=0.0
    np.testing.assert_allclose(result["mae"].sel(station=1).values.squeeze(), 0.0)
    np.testing.assert_allclose(result["br"].sel(station=1).values.squeeze(), 0.0)

    # Station 2: sim=[1.0, nan], obs=[nan, 2.0] -> mae=nan, br=0.5
    assert np.isnan(result["mae"].sel(station=2).values.squeeze())
    np.testing.assert_allclose(result["br"].sel(station=2).values.squeeze(), 0.5)

    # Station 3: sim=[nan, nan], obs=[nan, 1.0] -> mae=nan, br=nan
    assert np.isnan(result["mae"].sel(station=3).to_numpy())
    assert np.isnan(result["br"].sel(station=3).to_numpy())
