import argparse
import copy
import itertools
import os
import subprocess
import tempfile
from typing import TextIO

import utils

SLURM_LOG_DEFAULT = os.path.join(utils.STORAGE_ROOT, "slurm_logs")

SLURM_ARGS = {
    "partition": {"type": str, "required": True},
    "time": {"type": str, "default": "72:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "required": True},
    "gpus": {"type": str, "required": False, "default": None},
    "mem": {"type": str, "required": True},
    "output": {"type": str, "default": SLURM_LOG_DEFAULT},
    "error": {"type": str, "default": SLURM_LOG_DEFAULT},
    "job-name": {"type": str, "required": True},
    "exclude": {"type": str, "required": False, "default": None},
    "nodelist": {"type": str, "required": False, "default": None},
    "account": {"type": str, "required": False, "default": None},
}

SLURM_NAME_OVERRIDES = {"gpus": "gres", "cpus": "cpus-per-task"}

MANUAL_SWEEP = None


def write_slurm_header(f: TextIO, args: argparse.Namespace) -> None:
    # Make a copy of the args to prevent corruption
    args = copy.deepcopy(args)
    # Modify everything in the name space to later write it all at once
    for key in SLURM_ARGS:
        assert key.replace("-", "_") in args, "Key " + key + " not found."

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(args.error):
        os.makedirs(args.error)

    args.output = os.path.join(args.output, args.job_name + "_%A.out")
    args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus) if args.gpus is not None else args.gpus

    nl = "\n"
    f.write("#!/bin/bash" + nl)
    f.write(nl)
    for arg_name in SLURM_ARGS:
        arg_value = vars(args)[arg_name.replace("-", "_")]
        if arg_value is not None:
            f.write("#SBATCH --" + SLURM_NAME_OVERRIDES.get(arg_name, arg_name) + "=" + str(arg_value) + nl)

    f.write(nl)
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID' + nl)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + nl)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + nl)
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES' + nl)
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR' + nl)
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR' + nl)
    f.write(nl)
    f.write(". " + utils.ENV_SETUP_SCRIPT)
    f.write(nl)


if __name__ == "__main__":
    parser = utils.get_parser()
    # Add Slurm Arguments
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)
    parser.add_argument(
        "--remainder",
        default="split",
        choices=["split", "new"],
        help="Whether or not to spread out jobs that don't divide evently, or place them in a new job",
    )
    parser.add_argument("--scripts-per-job", type=int, default=1, help="number of scripts to run per slurm job.")
    args = parser.parse_args()

    if MANUAL_SWEEP is not None:
        scripts = []
        configs = list(itertools.product(*MANUAL_SWEEP.values()))
        keys = list(itertools.chain(*(k if isinstance(k, tuple) else (k,) for k in MANUAL_SWEEP)))
        for config in configs:
            values = itertools.chain(*(v if isinstance(v, tuple) else (v,) for v in config))
            script_args = {k: v for k, v in zip(keys, values, strict=False)}
            if "path" not in script_args:
                # Add a path based on the parameters.
                path = script_args.pop("path")
                name = "_".join([utils._format_name(v) for v in script_args.values()])
                script_args["path"] = os.path.join(path, name)
            scripts.append(script_args)
    else:
        scripts = utils.get_scripts(args)

    # Call python subprocess to launch the slurm jobs.
    num_slurm_calls = len(scripts) // args.scripts_per_job
    remainder_scripts = len(scripts) - num_slurm_calls * args.scripts_per_job
    scripts_per_call = [args.scripts_per_job for _ in range(num_slurm_calls)]
    if args.remainder == "split":
        for i in range(remainder_scripts):
            scripts_per_call[i] += 1  # Add the remainder jobs to spread them out as evenly as possible.
    elif args.remainder == "new":
        scripts_per_call.append(remainder_scripts)
    else:
        raise ValueError("Invalid job remainder specification.")
    assert sum(scripts_per_call) == len(scripts)
    script_index = 0
    procs = []
    for i, num_scripts in enumerate(scripts_per_call):
        current_scripts = scripts[script_index : script_index + num_scripts]
        script_index += num_scripts

        _, slurm_file = tempfile.mkstemp(text=True, prefix="job_", suffix=".sh")

        with open(slurm_file, "w+") as f:
            write_slurm_header(f, args)
            f.write("sleep " + str(2 * i) + " \n")  # Add a sleep to prevent all jobs from starting at the same time.
            # Now that we have written the header we can launch the jobs.
            for script_args in current_scripts:
                command_str = ["python", args.entry_point]
                for arg_name, arg_value in script_args.items():
                    command_str.append("--" + arg_name + "=" + str(arg_value))
                if len(current_scripts) != 1:
                    command_str.append("&")
                command_str = " ".join(command_str) + "\n"
                f.write(command_str)
            if len(current_scripts) != 1:
                f.write("wait")

        # Now launch the job
        # print(command_str)
        print("Launching job with slurm configuration:", slurm_file)
        proc = subprocess.Popen(["sbatch", slurm_file])
        procs.append(proc)

    exit_codes = [p.wait() for p in procs]
