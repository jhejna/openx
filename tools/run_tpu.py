"""
A script for running on TPUs.
NOTE: repo must be in the same region.
"""

import datetime
import itertools
import os
import tempfile

import utils

TPU_LOG_DEFAULT = os.path.join(utils.STORAGE_ROOT, "tpu_logs")
TPU_RUN_COMMAND = "gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --project {project} --command='{command}'"


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


if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument("--tpus", type=str, nargs="+", required=True)
    parser.add_argument("--scripts-per-vm", type=int, default=1, help="number of scripts to run per slurm job.")
    parser.add_argument("--zone", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    scripts = utils.get_scripts(args)

    assert len(scripts) <= len(args.tpus), "More scripts than TPUs."
    assert args.scripts_per_vm in {1, 2, 4}

    tpus = itertools.chain(*[[tpu] * args.scripts_per_vm for tpu in args.tpus])

    for idx, (script, tpu) in enumerate(zip(scripts, args.tpus, strict=False)):
        if args.scripts_per_vm == 2:
            prefix = ["TPU01", "TPU23"][idx % 2]
        elif args.scripts_per_vm == 4:
            prefix = ["TPU0", "TPU1", "TPU2", "TPU3"][idx % 4]
        else:
            prefix = ""

        _, script_file = tempfile.mkstemp(text=True, prefix="job_", suffix=".sh")

        with open(script_file, "w+") as f:
            write_tpu_header(f)
            # After we write the header, write the command
            entry_point, script_args = script
            command_str = ["PYTHONPATH=.", "python", entry_point]
            if prefix != "":
                command_str = [prefix, *command_str]
            for arg_name, arg_value in script_args.items():
                command_str.append("--" + arg_name)
                command_str.append(str(arg_value))
            command_str = " ".join(command_str) + "\n"
            f.write(command_str)

        # Now launch the job
        print("Launching job", script_file, "on tpu", tpu)
        session = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Construct the run command
        cmd = 'tmux new -d -s {session} "bash -i {script}; tmux kill-session -t {session}"; tmux pipe-pane -t {session} "cat >> {log_dir}/{session}.log"'.format(  # noqa: E501
            session=session, script=script_file, log_dir=TPU_LOG_DEFAULT
        )
        # Construct the gcp command
        cmd = TPU_RUN_COMMAND.format(tpu=tpu, command=cmd, zone=args.zone, project=args.project)

        # NOTE: the following is specific to the iliad TPU launcher. Be careful using without it.
        # check to determine if we have a pod
        if "pod" in tpu or all(k not in tpu for k in ("v4-8", "v3-8", "v2-8")):
            cmd += " --worker=all"

        # TPU launches are serial.
        # subprocess.run(cmd, shell=True)
