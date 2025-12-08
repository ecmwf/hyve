import numpy as np
import pytest

from hyve.hydrostats.stat_calc import stat_calc


@pytest.fixture
def sim_source_config():
    """Sim: station 1 = [1, 2], station 2 = [1, 2]."""
    return {
        "list-of-dicts": {
            "list_of_dicts": [
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 1,
                    "param": "dis",
                    "values": [1.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 1,
                    "param": "dis",
                    "values": [2.0],
                },
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 2,
                    "param": "dis",
                    "values": [1.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 2,
                    "param": "dis",
                    "values": [2.0],
                },
                # Stations that only exist in sim should be ignored
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 3,
                    "param": "dis",
                    "values": [999.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 3,
                    "param": "dis",
                    "values": [999.0],
                },
                # Times that only exist in sim should be ignored
                {
                    "date": 20240101,
                    "time": 6,
                    "number": 1,
                    "param": "dis",
                    "values": [100.0],
                },
                {
                    "date": 20240101,
                    "time": 6,
                    "number": 2,
                    "param": "dis",
                    "values": [500.0],
                },
                {
                    "date": 20240101,
                    "time": 6,
                    "number": 3,
                    "param": "dis",
                    "values": [999.0],
                },
            ]
        }
    }


@pytest.fixture
def obs_source_config():
    """Obs: station 1 = [1, 2] (identical), station 2 = [2, 1] (inverted)."""
    return {
        "list-of-dicts": {
            "list_of_dicts": [
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 1,
                    "param": "discharge",
                    "values": [1.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 1,
                    "param": "discharge",
                    "values": [2.0],
                },
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 2,
                    "param": "discharge",
                    "values": [2.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 2,
                    "param": "discharge",
                    "values": [1.0],
                },
                # Stations that only exist in obs should be ignored
                {
                    "date": 20240101,
                    "time": 0,
                    "number": 4,
                    "param": "discharge",
                    "values": [999.0],
                },
                {
                    "date": 20240102,
                    "time": 0,
                    "number": 4,
                    "param": "discharge",
                    "values": [999.0],
                },
                # Times that only exist in obs should be ignored
                {
                    "date": 20240102,
                    "time": 6,
                    "number": 1,
                    "param": "discharge",
                    "values": [100.0],
                },
                {
                    "date": 20240102,
                    "time": 6,
                    "number": 2,
                    "param": "discharge",
                    "values": [500.0],
                },
                {
                    "date": 20240102,
                    "time": 6,
                    "number": 4,
                    "param": "discharge",
                    "values": [999.0],
                },
            ]
        }
    }


def test_stat_calc(sim_source_config, obs_source_config):
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
            "source": sim_source_config,
            "coords": {"s": "number", "t": "forecast_reference_time"},
        },
        "obs": {
            "source": obs_source_config,
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
