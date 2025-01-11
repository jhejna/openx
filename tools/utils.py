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
DEFAULT_REQUIRED_ARGS = ["path", "config"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", type=str, default=None)
    parser.add_argument(
        "--arguments",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Set kv pairs used as args for the entry point script.",
    )
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
    for arguments in args.arguments:
        parsed_args = parse_vars(arguments)
        if args.entry_point == DEFAULT_ENTRY_POINT:
            for arg_name in DEFAULT_REQUIRED_ARGS:
                assert arg_name in parsed_args

        # Handle case where we have a sweep
        if args.entry_point == DEFAULT_ENTRY_POINT and parsed_args["config"].endswith(".json"):
            assert "name" not in parsed_args, "Name populated automatically for sweeps."
            # We have a sweep
            for config_str, name in zip(*load_ml_collections_sweep(parsed_args["config"]), strict=False):
                script_args = parsed_args.copy()
                script_args.update(dict(name='"' + name + '"', config=config_str))
                scripts.append((args.entry_point, script_args))
        else:
            scripts.append((args.entry_point, parsed_args))

    return scripts


def _format_name(s):
    if os.path.exists(s):
        s = os.path.basename(os.path.normpath(s))
        s = os.path.splitext(s)[0]
    return s.replace("/", "_")


def load_ml_collections_sweep(path):
    assert path.endswith(".json"), "Must be a json file"
    with open(path, "r") as f:
        sweep = json.load(f)
    assert "config" in sweep, "json did not have a 'config' field"
    config = sweep.pop("config")
    assert all(isinstance(v, list) for v in sweep.values())
    # NOTE: we care about order
    sweep = {k: [str(vv) for vv in v] for k, v in sweep.items()}
    config_strs = list(itertools.product(*(v for v in sweep.values())))
    names = [
        "_".join([k + "-" + _format_name(v) for k, v in zip(sweep.keys(), config_str, strict=False)])
        for config_str in config_strs
    ]
    config_strs = [",".join(config_str) for config_str in config_strs]
    config_strs = [config + ":" + config_str for config_str in config_strs]
    return config_strs, names
