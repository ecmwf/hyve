# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys

import yaml
from pydantic import ValidationError

from hyve.config import ExtractorConfig
from hyve.extraction import extractor
from hyve.hydrostats.stat_calc import stat_calc

logger = logging.getLogger(__name__)


def commandlineify(func):
    def wrapper(args=None):
        if args is None:
            args = sys.argv[1:]
        parser = argparse.ArgumentParser(description="Run tool with YAML config")
        parser.add_argument("config", help="Path to the YAML config file")
        args = parser.parse_args(args)
        confpath = args.config
        with open(confpath, "r") as file:
            config = yaml.safe_load(file)
        func(config)

    return wrapper


def extractor_cli(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run hyve extractor with YAML config")
    parser.add_argument("config", help="Path to the YAML config file")
    parsed = parser.parse_args(args)

    try:
        with open(parsed.config, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        sys.stderr.write(f"Error: config file not found: {parsed.config}\n")
        sys.exit(1)

    try:
        extractor(ExtractorConfig(**(raw or {})))
    except ValidationError as e:
        sys.stderr.write("Error: invalid configuration\n")
        for err in e.errors():
            path = " -> ".join(str(x) for x in err["loc"])
            sys.stderr.write(f"  {path}: {err['msg']}\n")
        sys.exit(1)


stat_calc_cli = commandlineify(stat_calc)


if __name__ == "__main__":
    from importlib.metadata import entry_points

    eps = entry_points().select(group="console_scripts")
    tools = {ep.name: ep.load() for ep in eps if ep.module.startswith("hyve.")}
    tool_name = sys.argv[1]
    if tool_name in tools:
        tools[tool_name](sys.argv[2:])
    else:
        logger.error(
            f"Tool '{tool_name}' not found. Available tools: {', '.join(tools.keys())}"
        )
        sys.exit(1)
