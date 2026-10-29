# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from pydantic import ValidationError

from hyve.config import ExtractorConfig


def test_gribjump_requires_request_option(tmp_path):
    """Test that gribjump source without 'request' in options raises ValueError."""
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1"],
            "idx": [100],
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {"source": {"name": "gribjump", "options": {}}},  # Missing 'request'
    }

    with pytest.raises(
        ValidationError, match="Gribjump source requires 'request' in options."
    ):
        ExtractorConfig(**config)


def test_gribjump_options_must_be_dict(tmp_path):
    """Test that gribjump source options must be a dictionary."""
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1"],
            "idx": [100],
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {
            "source": {
                "name": "gribjump",
                "options": "not-a-dict",  # Invalid options type
            }
        },
    }

    with pytest.raises(
        ValidationError, match="Gribjump source options must be a dictionary."
    ):
        ExtractorConfig(**config)


def test_gribjump_options_must_only_contain_request(tmp_path):
    """Test that gribjump source options must only contain one 'request' entry."""
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1"],
            "idx": [100],
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {
            "source": {
                "name": "gribjump",
                "options": {
                    "request": {"class": "od", "expver": "0001", "stream": "oper"},
                    "extra": "not-allowed",
                },
            }
        },
    }

    with pytest.raises(
        ValidationError,
        match="Gribjump source options must only contain one 'request' entry",
    ):
        ExtractorConfig(**config)


def test_gribjump_valid_request(tmp_path):
    """Test that gribjump source with a valid 'request' passes validation."""
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1"],
            "idx": [100],
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {
            "source": {
                "name": "gribjump",
                "options": {
                    "request": {"class": "od", "expver": "0001", "stream": "oper"}
                },
            }
        },
    }

    # Should not raise any exception
    ExtractorConfig(**config)


def test_gribjump_accepts_list_of_requests(tmp_path):
    """Test that gribjump source accepts a list of 'request' entries."""
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1"],
            "idx": [100],
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {
            "source": {
                "name": "gribjump",
                "options": {
                    "request": [
                        {"class": "od", "expver": "0001", "stream": "oper"},
                        {"class": "od", "expver": "0002", "stream": "oper"},
                    ]
                },
            }
        },
    }

    # Should not raise any exception
    ExtractorConfig(**config)
