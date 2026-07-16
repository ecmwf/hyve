# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from hyve.cli import extractor_cli


def test_cli_validation_error_exits_cleanly(tmp_path, capsys):
    """Missing required field produces exact formatted error on stderr and exits 1."""
    config = {
        "station": {"file": "s.csv", "index": {}},  # missing name
        "grid": {"source": {"list-of-dicts": {}}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(config))

    with pytest.raises(SystemExit) as exc:
        extractor_cli([str(path)])

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        "Error: invalid configuration\n  station -> name: Field required\n"
    )


def test_cli_empty_config_exits_cleanly(tmp_path, capsys):
    """An empty config file exits 1 with a clean validation error."""
    path = tmp_path / "empty.yaml"
    path.write_text("")

    with pytest.raises(SystemExit) as exc:
        extractor_cli([str(path)])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: invalid configuration\n")


def test_cli_file_not_found_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        extractor_cli(["/nonexistent/config.yaml"])

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        "Error: config file not found: /nonexistent/config.yaml\n"
    )
