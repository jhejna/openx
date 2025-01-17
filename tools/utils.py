"""
Utils for running sweeps.
Note that this is designed to be able to be used outside of the core environment.
Thus, it only has default python dependencies (hence no absl)
"""

import argparse
import itertools
import json
import os
from typing import Dict, Iterable, List, Tuple

# Default configuration for running jobs
REPO_PATH = os.path.dirname(os.path.dirname(__file__))
STORAGE_ROOT = os.path.dirname(REPO_PATH)
ENV_SETUP_SCRIPT = os.path.join(REPO_PATH, "setup_shell.sh")
TMP_DIR = os.path.join(STORAGE_ROOT, "tmp")

DEFAULT_ENTRY_POINT = "scripts/train.py"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", type=str, default=None)
    parser.add_argument(
        "--arguments",
        metavar="KEY=VALUE",
        nargs="+",
        help="Set kv pairs used as args for the entry point script.",
    )
    parser.add_argument("--sweep", type=str, default=None, help="Path to a json file containing a sweep.")
    return parser


def parse_var(s: str) -> Tuple[str]:
    """
    Parse a key, value pair, separated by '='
    """
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = "=".join(items[1:])
    return (key, value)


def parse_vars(items: Iterable) -> Dict:
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def get_scripts(args: argparse.Namespace) -> List[Tuple[str, Dict]]:
    if args.entry_point is None:
        args.entry_point = DEFAULT_ENTRY_POINT

    scripts = []
    parsed_args = parse_vars(args.arguments)
    assert not (
        args.entry_point == DEFAULT_ENTRY_POINT and "config" in parsed_args and args.sweep is not None
    ), "Cannot have both a config and a sweep"
    if args.entry_point == DEFAULT_ENTRY_POINT and parsed_args.get("config", "").endswith(".json"):
        args.sweep = parsed_args.pop("config")  # Remove the config, and pass it to sweep.

    if args.sweep is not None:
        for config, name in zip(*load_sweep(args.sweep), strict=False):
            script_args = parsed_args.copy()
            if args.entry_point == DEFAULT_ENTRY_POINT:
                # Special handling for ML Collections. This gets passed as a config string.
                script_args["config"] = config.pop("config") + ":" + ",".join(config.values())
                script_args["name"] = '"' + name + '"'
                script_args["include_timestamp"] = "false"
            else:
                script_args.update(config)  # Update the default config with the sweep
                if "path" in script_args:
                    script_args["path"] = os.path.join(script_args["path"], name)  # Add the name.
            scripts.append(script_args)
    else:
        scripts.append(parsed_args)

    return scripts


def _format_name(s):
    s = str(s)
    if os.path.exists(s):
        s = os.path.basename(os.path.normpath(s))
        s = os.path.splitext(s)[0]
    return s.replace("/", "_")


def load_sweep(path):
    assert path.endswith(".json"), "Must be a json file"
    with open(path, "r") as f:
        sweep = json.load(f)
    # convert all to strings if not already
    sweep = {k: [str(vv) for vv in v] for k, v in sweep.items()}
    # Get all values
    configs, names = [], []
    for config_values in itertools.product(*(v for v in sweep.values())):
        config = {}
        name = []
        # We only add the first part of the name if there are multiple
        for k, v in zip(sweep.keys(), config_values, strict=True):
            for kk, vv in zip(k.split(","), v.split(","), strict=True):
                config[kk] = vv
                name.append(kk + "-" + _format_name(vv))
        configs.append(config)
        names.append("_".join(name))

    return configs, names
