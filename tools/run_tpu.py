"""
A script for running on TPUs.
NOTE: repo must be in the same region and run on TPUs.
"""

import datetime
import itertools
import math
import os
import re
import subprocess
import tempfile

import utils

TPU_LOG_DEFAULT = os.path.join(utils.STORAGE_ROOT, "tpu_logs")
TPU_RUN_COMMAND = "gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --project {project} --command='{command}'"
STATE_COMMAND = "gcloud alpha compute tpus queued-resources list --project {project} --zone {zone} --quiet"


def write_tpu_header(f):
    nl = "\n"
    f.write("#!/bin/bash" + nl)
    f.write(nl)

    f.write(
        'alias TPU0="TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"'  # noqa: E501
        + nl
    )
    f.write(
        'alias TPU1="TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477"'  # noqa: E501
        + nl
    )
    f.write(
        'alias TPU2="TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"'  # noqa: E501
        + nl
    )
    f.write(
        'alias TPU3="TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479"'  # noqa: E501
        + nl
    )
    f.write(
        'alias TPU01="TPU_VISIBLE_DEVICES=0,1 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"'  # noqa: E501
        + nl
    )
    f.write(
        'alias TPU23="TPU_VISIBLE_DEVICES=2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"'  # noqa: E501
        + nl
    )

    f.write(nl)
    f.write(". " + utils.ENV_SETUP_SCRIPT)
    f.write(nl)

    f.write("sudo chmod -R 777 " + str(utils.REPO_PATH))
    f.write(nl)


def expand_string_ranges(s):
    """
    Expands strings with ranges in square brackets into a list of individual strings.

    Args:
        s (str): Input string with ranges, e.g., "tpu-v3-8-p-[0-1]" or "tpu-v3-8-p-[3,5]".

    Returns:
        list: List of expanded strings, e.g., ["tpu-v3-8-p-0", "tpu-v3-8-p-1"].
    """
    # Regular expression to find ranges in square brackets
    pattern = re.compile(r"\[(.*?)\]")

    # Find all ranges in the input string
    matches = pattern.findall(s)

    if not matches:
        return [s]  # No ranges to expand

    # Extract the first range and split by comma or hyphen
    range_part = matches[0]

    if "," in range_part:
        # Handle comma-separated values (e.g., "3,5")
        values = range_part.split(",")
    elif "-" in range_part:
        # Handle range values (e.g., "0-1")
        start, end = map(int, range_part.split("-"))
        values = [str(i) for i in range(start, end + 1)]
    else:
        return [s]  # No valid range format

    # Replace the range in the original string with each value
    return [s.replace(f"[{range_part}]", v) for v in values]


if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument("--tpus", type=str, nargs="+", required=True)
    parser.add_argument("--scripts-per-vm", type=int, default=1, help="number of scripts to run per slurm job.")
    parser.add_argument("--zone", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)

    args = parser.parse_args()
    scripts = utils.get_scripts(args)

    assert args.scripts_per_vm in {1, 2, 4}

    tpus = []
    for tpu in args.tpus:
        tpus.extend(expand_string_ranges(tpu))
    tpus = ["tpu-" + args.zone + "-" + tpu for tpu in tpus]

    # Check to see if some TPUs are down.
    state_stdout = subprocess.check_output(STATE_COMMAND.format(project=args.project, zone=args.zone), shell=True)
    lines = state_stdout.decode("utf-8").split("\n")
    status_map = {node[0]: node[-1] for node in [line.split() for line in lines[1:-1]]}
    alive_set = {k for k, v in status_map.items() if v not in ["FAILED", "SUSPENDED", "SUSPENDING"]}

    # If some specified TPUs are dead raise a warning.
    wait_for_confirm = False
    for tpu in tpus:
        if tpu not in alive_set:
            print("[WARNING] tpu", tpu, "is not alive.")
            wait_for_confirm = True

    if not wait_for_confirm or input("Continue (y)? ") == "y":
        pass
    else:
        exit()

    # Subset to only the alive tpus
    tpus = [tpu for tpu in tpus if tpu in alive_set]

    # See if we should run multiple per TPU.
    tpus = list(itertools.chain(*[[tpu] * args.scripts_per_vm for tpu in tpus]))

    # Determine the number of scripts per TPU.
    chunk_size = math.ceil(len(scripts) / len(tpus))

    for idx, tpu in enumerate(tpus):
        chunk_scripts = scripts[: min(chunk_size, len(scripts))]
        scripts = scripts[min(chunk_size, len(scripts)) :]

        if args.scripts_per_vm == 2:
            prefix = ["TPU01", "TPU23"][idx % 2]
        elif args.scripts_per_vm == 4:
            prefix = ["TPU0", "TPU1", "TPU2", "TPU3"][idx % 4]
        else:
            prefix = ""

        _, script_file = tempfile.mkstemp(
            text=True, dir=utils.TMP_DIR, prefix="job_{idx}_".format(idx=str(idx)), suffix=".sh"
        )

        with open(script_file, "w+") as f:
            write_tpu_header(f)

            for script_args in chunk_scripts:
                # After we write the header, write the command
                command_str = ["PYTHONPATH=.", "python", args.entry_point]
                if prefix != "":
                    command_str = [prefix, *command_str]
                for arg_name, arg_value in script_args.items():
                    if arg_name == "path" or arg_name == "output":
                        # If we have a path make sure we have permissions.
                        f.write("sudo chmod -R 777 " + arg_value + "\n")
                    command_str.append("--" + arg_name + "=" + str(arg_value))
                command_str = " ".join(command_str) + "\n"
                f.write(command_str)
                f.write("\n")

        # Now launch the job
        print("Launching job", script_file, "on tpu", tpu)
        session = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Construct the run command
        cmd = 'tmux new -d -s {session} "bash -i {script}; tmux kill-session -t {session}"; tmux pipe-pane -t {session} "cat >> {log_dir}/{session}.log"'.format(  # noqa: E501
            session=session, script=script_file, log_dir=TPU_LOG_DEFAULT
        )
        # Construct the gcp command
        cmd = TPU_RUN_COMMAND.format(tpu=tpu, command=cmd, zone=args.zone, project=args.project)

        # NOTE: the following is specific to our TPU launcher. Be careful using without it.
        # check to determine if we have a pod
        if "pod" in tpu or all(k not in tpu for k in ("v4-8", "v3-8", "v2-8")):
            cmd += " --worker=all"

        print(cmd)
        print()

        # TPU launches are serial.
        subprocess.run(cmd, shell=True, check=False)
