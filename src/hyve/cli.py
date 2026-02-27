import argparse
import logging
import sys

import yaml

from hyve.config import ExtractorConfig
from hyve.extraction import extractor
from hyve.hydrostats.stat_calc import stat_calc

logger = logging.getLogger(__name__)


def extractor_cli(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Extract timeseries at station locations from gridded data"
    )
    parser.add_argument("config", help="Path to the YAML config file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(parsed.config, "r") as f:
        config = yaml.safe_load(f)
    extractor(ExtractorConfig(**config))


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
