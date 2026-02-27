"""Unit tests for the extractor function."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.config import ExtractorConfig, StationConfig
from hyve.extraction import extractor, parse_stations


def _extract(config_dict: dict) -> xr.Dataset:
    """Test helper: build ExtractorConfig from dict and run extractor."""
    return extractor(ExtractorConfig(**config_dict))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_grid_data():
    """4x5 grid, 2 timesteps, ascending lats, temperature variable."""
    lats = np.array([40.0, 41.0, 42.0, 43.0])
    lons = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    temperature_values = np.array(
        [
            # t=0: 2024-01-01
            [
                [10.0, 11.0, 12.0, 13.0, 14.0],  # lat=40
                [15.0, 16.0, 17.0, 18.0, 19.0],  # lat=41
                [20.0, 21.0, 22.0, 23.0, 24.0],  # lat=42
                [25.0, 26.0, 27.0, 28.0, 29.0],  # lat=43
            ],
            # t=1: 2024-01-02
            [
                [30.0, 31.0, 32.0, 33.0, 34.0],  # lat=40
                [35.0, 36.0, 37.0, 38.0, 39.0],  # lat=41
                [40.0, 41.0, 42.0, 43.0, 44.0],  # lat=42
                [45.0, 46.0, 47.0, 48.0, 49.0],  # lat=43
            ],
        ]
    )

    list_of_dicts = []

    list_of_dicts.append(
        {
            "values": temperature_values[0].flatten(),
            "param": "temperature",
            "date": 20240101,
            "time": 0,
            "distinctLatitudes": lats.tolist(),
            "distinctLongitudes": lons.tolist(),
        }
    )

    list_of_dicts.append(
        {
            "values": temperature_values[1].flatten(),
            "param": "temperature",
            "date": 20240102,
            "time": 0,
            "distinctLatitudes": lats.tolist(),
            "distinctLongitudes": lons.tolist(),
        }
    )

    return list_of_dicts


@pytest.fixture
def descending_lat_grid_data():
    """4x3 grid, 2 timesteps, DESCENDING latitudes (north-to-south).

    Layout (per timestep):
        lat=43: [30, 31, 32]  (index 0)
        lat=42: [20, 21, 22]  (index 1)
        lat=41: [10, 11, 12]  (index 2)
        lat=40: [ 0,  1,  2]  (index 3)
    """
    lats_desc = [43.0, 42.0, 41.0, 40.0]
    lons = [10.0, 11.0, 12.0]

    vals_t0 = np.array([30, 31, 32, 20, 21, 22, 10, 11, 12, 0, 1, 2], dtype=float)
    vals_t1 = vals_t0 + 100

    return [
        {
            "values": vals_t0,
            "param": "temperature",
            "date": 20240101,
            "time": 0,
            "distinctLatitudes": lats_desc,
            "distinctLongitudes": lons,
        },
        {
            "values": vals_t1,
            "param": "temperature",
            "date": 20240102,
            "time": 0,
            "distinctLatitudes": lats_desc,
            "distinctLongitudes": lons,
        },
    ]


@pytest.fixture
def station_dataframe():
    """2 stations with both index and coordinate columns."""
    return pd.DataFrame(
        {
            "station_id": ["STATION_A", "STATION_B"],
            "opt_x_index": [1, 2],
            "opt_y_index": [2, 3],
            "opt_x_coord": [41.00005, 42.00005],  # within tolerance
            "opt_y_coord": [12.00005, 13.00005],
        }
    )


@pytest.fixture
def station_csv_file(station_dataframe, tmp_path):
    """Write station DataFrame to temporary CSV."""
    csv_path = tmp_path / "stations.csv"
    station_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.mark.parametrize(
    "mapping_config",
    [
        {"index": {"x": "opt_x_index", "y": "opt_y_index"}},
        {"coords": {"x": "opt_x_coord", "y": "opt_y_coord"}},
    ],
    ids=["index", "coords"],
)
def test_extractor_with_temperature(dummy_grid_data, station_csv_file, mapping_config):
    """Test extraction with both index and coords mapping."""
    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            **mapping_config,
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
    }

    result_ds = _extract(config)

    assert isinstance(result_ds, xr.Dataset)
    assert "temperature" in result_ds.data_vars
    assert "station" in result_ds.dims
    assert len(result_ds.station) == 2
    assert list(result_ds.station.values) == ["STATION_A", "STATION_B"]

    time_dim = "time" if "time" in result_ds.dims else "forecast_reference_time"
    assert len(result_ds[time_dim]) == 2

    # Station A: indices [1,2] -> values [17.0, 37.0]
    # Station B: indices [2,3] -> values [23.0, 43.0]
    np.testing.assert_allclose(
        result_ds["temperature"].sel(station="STATION_A").values, [17.0, 37.0]
    )
    np.testing.assert_allclose(
        result_ds["temperature"].sel(station="STATION_B").values, [23.0, 43.0]
    )


def test_extractor_with_station_filter(dummy_grid_data, tmp_path):
    """Test station filtering."""
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S2", "S3"],
            "opt_x_index": [1, 2, 1],
            "opt_y_index": [2, 3, 3],
            "network": ["primary", "secondary", "primary"],
        }
    )
    csv_path = tmp_path / "stations.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "filter": "network == 'primary'",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
    }

    result_ds = _extract(config)

    assert len(result_ds.station) == 2
    assert list(result_ds.station.values) == ["S1", "S3"]
    assert result_ds["temperature"].sel(station="S1").values[0] == 17.0
    assert result_ds["temperature"].sel(station="S3").values[0] == 18.0


def test_extractor_rejects_both_index_and_coords(dummy_grid_data, station_csv_file):
    """Providing both index and coords raises ValidationError."""
    from pydantic import ValidationError

    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
            "coords": {"x": "opt_x_coord", "y": "opt_y_coord"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(
        ValidationError, match="must use either 'index' or 'coords', not both"
    ):
        ExtractorConfig(**config)


def test_extractor_with_output_file(dummy_grid_data, station_csv_file, tmp_path):
    """Test output file writing."""
    output_file = tmp_path / "output.nc"

    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
        "output": {
            "file": str(output_file),
        },
    }

    result_ds = _extract(config)

    assert output_file.exists()

    loaded_ds = xr.open_dataset(output_file)
    assert "temperature" in loaded_ds.data_vars
    assert "station" in loaded_ds.dims
    assert len(loaded_ds.station) == 2

    xr.testing.assert_allclose(result_ds["temperature"], loaded_ds["temperature"])
    xr.testing.assert_equal(result_ds.station, loaded_ds.station)

    loaded_ds.close()


def test_extractor_with_empty_stations(dummy_grid_data, tmp_path):
    """Test that extractor raises clear error for empty station list."""
    empty_csv = tmp_path / "empty_stations.csv"
    pd.DataFrame(columns=["station_id", "opt_x_index", "opt_y_index"]).to_csv(
        empty_csv, index=False
    )

    config = {
        "station": {
            "file": str(empty_csv),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValueError, match="No stations found"):
        _extract(config)


@patch("earthkit.data.from_source")
def test_extractor_gribjump(mock_from_source, tmp_path):
    """Gribjump path: ranges, earthkit call, deduplication, and data values."""
    mock_source = Mock()
    mock_source.to_xarray.return_value = xr.Dataset(
        {
            "temperature": xr.DataArray(
                [[15.0, 25.0], [35.0, 45.0]], dims=["index", "time"]
            )
        }
    )
    mock_from_source.return_value = mock_source

    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1", "S2", "S3"],
            "idx": [100, 200, 100],  # S1 and S3 share index 100
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {
            "source": {
                "gribjump": {
                    "request": {"class": "od", "expver": "0001", "stream": "oper"}
                }
            }
        },
    }

    result = _extract(config)

    mock_from_source.assert_called_once_with(
        "gribjump",
        request={"class": "od", "expver": "0001", "stream": "oper"},
        ranges=[(100, 101), (200, 201)],
    )

    assert len(result.station) == 3
    assert list(result.station.values) == ["S1", "S2", "S3"]

    # S1 and S3 share index 100 → unique pos 0 → row [15, 25]
    np.testing.assert_allclose(
        result["temperature"].sel(station="S1").values, [15.0, 25.0]
    )
    np.testing.assert_allclose(
        result["temperature"].sel(station="S3").values, [15.0, 25.0]
    )
    # S2: index 200 → unique pos 1 → row [35, 45]
    np.testing.assert_allclose(
        result["temperature"].sel(station="S2").values, [35.0, 45.0]
    )


# ---------------------------------------------------------------------------
# test-sort-bug: Regression test for Issue #5
# ---------------------------------------------------------------------------


def test_index_extraction_with_descending_latitudes(descending_lat_grid_data, tmp_path):
    """Index-based extraction must work when grid latitudes are descending.

    The station indices [0, 0] and [2, 1] were computed against the original
    descending grid:
        index 0 → lat=43, index 2 → lat=41
    The extractor must not sort the grid before applying these indices.
    """
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["HIGH_LAT", "MID_LAT"],
            "opt_x_index": [0, 2],
            "opt_y_index": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": descending_lat_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    # HIGH_LAT: index [0, 0] on descending grid → lat=43, lon=10 → values 30, 130
    np.testing.assert_allclose(
        result["temperature"].sel(station="HIGH_LAT").values, [30.0, 130.0]
    )
    # MID_LAT: index [2, 1] on descending grid → lat=41, lon=11 → values 11, 111
    np.testing.assert_allclose(
        result["temperature"].sel(station="MID_LAT").values, [11.0, 111.0]
    )


def test_coord_extraction_with_descending_latitudes(descending_lat_grid_data, tmp_path):
    """Coord-based extraction must work regardless of grid lat ordering."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["HIGH_LAT", "MID_LAT"],
            "lat": [43.0, 41.0],
            "lon": [10.0, 11.0],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "coords": {"x": "lat", "y": "lon"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": descending_lat_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    # lat=43, lon=10 → values 30, 130
    np.testing.assert_allclose(
        result["temperature"].sel(station="HIGH_LAT").values, [30.0, 130.0]
    )
    # lat=41, lon=11 → values 11, 111
    np.testing.assert_allclose(
        result["temperature"].sel(station="MID_LAT").values, [11.0, 111.0]
    )


# ---------------------------------------------------------------------------
# test-happy-path-gaps: Fill gaps in existing happy-path tests
# ---------------------------------------------------------------------------


def test_shared_grid_cell_regular_path(dummy_grid_data, tmp_path):
    """Two stations mapping to the same grid cell get identical values."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "opt_x_index": [1, 1],
            "opt_y_index": [2, 2],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    assert len(result.station) == 2
    # Both stations at [1, 2] → lat=41, lon=12 → values 17, 37
    np.testing.assert_allclose(
        result["temperature"].sel(station="S1").values, [17.0, 37.0]
    )
    np.testing.assert_allclose(
        result["temperature"].sel(station="S2").values, [17.0, 37.0]
    )


def test_default_station_column_names(dummy_grid_data, tmp_path):
    """Index config with no explicit x/y uses default column names."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["S1"],
            "opt_x_index": [0],
            "opt_y_index": [0],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {},  # uses defaults: opt_x_index, opt_y_index
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    # [0, 0] → lat=40, lon=10 → values 10, 30
    np.testing.assert_allclose(
        result["temperature"].sel(station="S1").values, [10.0, 30.0]
    )


def test_no_mapping_raises_error(dummy_grid_data, tmp_path):
    """Station config with neither index nor coords nor index_1d raises."""
    from pydantic import ValidationError

    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["S1"], "x": [0], "y": [0]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {"file": str(csv_path), "name": "station_id"},
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(
        ValidationError, match="must provide either 'index' or 'coords'"
    ):
        ExtractorConfig(**config)


# ---------------------------------------------------------------------------
# test-edge-cases
# ---------------------------------------------------------------------------


def test_single_station_extraction(dummy_grid_data, tmp_path):
    """Extraction works with a single station."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {"station_id": ["ONLY"], "opt_x_index": [3], "opt_y_index": [4]}
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    assert len(result.station) == 1
    # [3, 4] → lat=43, lon=14 → values 29, 49
    np.testing.assert_allclose(
        result["temperature"].sel(station="ONLY").values, [29.0, 49.0]
    )


def test_non_uniform_grid_spacing(tmp_path):
    """Coord-based extraction works on a grid with non-uniform spacing."""
    lats = np.array([40.0, 41.0, 43.0, 47.0])  # gaps: 1, 2, 4
    lons = np.array([10.0, 10.5, 12.0])  # gaps: 0.5, 1.5

    da = xr.DataArray(
        np.arange(24, dtype=float).reshape(2, 4, 3),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            "latitude": lats,
            "longitude": lons,
        },
    )

    csv_path = tmp_path / "stations.csv"
    # Station near (43.0, 10.5) → nearest lat=43 (idx 2), nearest lon=10.5 (idx 1)
    pd.DataFrame({"station_id": ["S1"], "lat": [43.00005], "lon": [10.50005]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "coords": {"x": "lat", "y": "lon"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": []}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with patch("hyve.extraction.load_da", return_value=(da, "tp")):
        result = _extract(config)

    # lat=43 (idx 2), lon=10.5 (idx 1) → t=0: 2*3+1=7, t=1: 1*12+2*3+1=19
    np.testing.assert_allclose(result["tp"].sel(station="S1").values, [7.0, 19.0])


# ---------------------------------------------------------------------------
# test-dask-lazy: Verify dask-chunked input works
# ---------------------------------------------------------------------------


def test_dask_chunked_coord_extraction(dummy_grid_data, tmp_path):
    """Coord-based extraction produces correct results with dask-backed input."""
    lats = np.array([40.0, 41.0, 42.0, 43.0])
    lons = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    t0 = np.array(
        [
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
        ],
        dtype=float,
    )
    t1 = t0 + 20
    import dask.array as dask_array

    # Use full-size chunks to avoid expensive rechunking on tiny arrays
    da_chunked = xr.DataArray(
        dask_array.from_array(np.stack([t0, t1]), chunks=-1),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            "latitude": lats,
            "longitude": lons,
        },
    )
    assert da_chunked.chunks is not None

    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "lat": [41.00005, 42.99995],
            "lon": [12.00005, 13.00005],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "coords": {"x": "lat", "y": "lon"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": []}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with patch("hyve.extraction.load_da", return_value=(da_chunked, "temperature")):
        result = _extract(config)

    # S1: nearest (41, 12) → [17, 37], S2: nearest (43, 13) → [28, 48]
    np.testing.assert_allclose(
        result["temperature"].sel(station="S1").values, [17.0, 37.0]
    )
    np.testing.assert_allclose(
        result["temperature"].sel(station="S2").values, [28.0, 48.0]
    )


# ---------------------------------------------------------------------------
# test-validation-errors
# ---------------------------------------------------------------------------


def test_duplicate_station_names_raises(dummy_grid_data, tmp_path):
    """Duplicate station names must raise ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["DUP", "DUP", "UNIQUE"],
            "opt_x_index": [0, 1, 2],
            "opt_y_index": [0, 1, 2],
        }
    ).to_csv(csv_path, index=False)

    station_config = StationConfig(
        file=str(csv_path),
        name="station_id",
        index={"x": "opt_x_index", "y": "opt_y_index"},
    )

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        parse_stations(station_config)


def test_nan_in_station_indices_raises(dummy_grid_data, tmp_path):
    """NaN in station index columns must raise ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "opt_x_index": [0, np.nan],
            "opt_y_index": [1, 2],
        }
    ).to_csv(csv_path, index=False)

    station_config = StationConfig(
        file=str(csv_path),
        name="station_id",
        index={"x": "opt_x_index", "y": "opt_y_index"},
    )

    with pytest.raises(ValueError, match="NaN|nan|missing"):
        parse_stations(station_config)


def test_nan_in_station_coords_raises(dummy_grid_data, tmp_path):
    """NaN in station coordinate columns must raise ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "lat": [41.0, np.nan],
            "lon": [12.0, 13.0],
        }
    ).to_csv(csv_path, index=False)

    station_config = StationConfig(
        file=str(csv_path),
        name="station_id",
        coords={"x": "lat", "y": "lon"},
    )

    with pytest.raises(ValueError, match="NaN|nan|missing"):
        parse_stations(station_config)


def test_missing_station_config_key_raises(dummy_grid_data, tmp_path):
    """Missing 'station' key in config raises ValidationError."""
    from pydantic import ValidationError

    config = {
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
        },
    }

    with pytest.raises(ValidationError, match="station"):
        ExtractorConfig(**config)


def test_missing_grid_source_raises(tmp_path):
    """Missing 'source' in grid config raises ValidationError."""
    from pydantic import ValidationError

    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["S1"], "opt_x_index": [0], "opt_y_index": [0]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {},
    }

    with pytest.raises(ValidationError, match="source"):
        ExtractorConfig(**config)


def test_missing_station_name_key_raises(dummy_grid_data, tmp_path):
    """Missing 'name' in station config raises ValidationError."""
    from pydantic import ValidationError

    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"sid": ["S1"], "opt_x_index": [0], "opt_y_index": [0]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {
            "file": str(csv_path),
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValidationError, match="name"):
        ExtractorConfig(**config)


def test_index_out_of_bounds_raises(dummy_grid_data, tmp_path):
    """Index exceeding grid dimensions raises ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {"station_id": ["S1"], "opt_x_index": [99], "opt_y_index": [0]}
    ).to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValueError, match="out of grid bounds"):
        _extract(config)


def test_coord_outside_grid_domain_raises(dummy_grid_data, tmp_path):
    """Station coordinates outside grid bounding box raises ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["S1"], "lat": [90.0], "lon": [12.0]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "coords": {"x": "lat", "y": "lon"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValueError, match="outside tolerance"):
        _extract(config)


def test_coord_tolerance_can_be_disabled(dummy_grid_data, tmp_path):
    """Disabling tolerance allows far-away stations to map to nearest grid."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["S1"], "lat": [90.0], "lon": [12.0]}).to_csv(
        csv_path, index=False
    )

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "coords": {"x": "lat", "y": "lon", "tolerance": None},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    result = _extract(config)

    # lat=90, lon=12 -> nearest lat=43, lon=12 -> values [27, 47]
    np.testing.assert_allclose(
        result["temperature"].sel(station="S1").values, [27.0, 47.0]
    )


def test_wrong_column_name_in_config_raises(tmp_path):
    """Config referencing a non-existent CSV column raises ValueError."""
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["S1"], "lat": [41.0], "lon": [12.0]}).to_csv(
        csv_path, index=False
    )

    station_config = StationConfig(
        file=str(csv_path),
        name="station_id",
        coords={"x": "WRONG_NAME", "y": "lon"},
    )

    with pytest.raises(ValueError, match="missing required columns"):
        parse_stations(station_config)
