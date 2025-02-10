import os
import subprocess

from absl import app, flags

flags.DEFINE_string("path", None, "The path to the checkpoint step.", required=True)
flags.DEFINE_string("output", None, "The path to save the checkpoint", required=True)

FLAGS = flags.FLAGS


def main(_):
    path = FLAGS.path
    base_cmd = "gsutil cp" if path.startswith("gs://") else "scp"

    path = path[:-1] if path.endswith("/") else path
    assert os.path.basename(path).isdigit()
    step = int(os.path.basename(path))
    path = os.path.dirname(path)
    dest = os.path.join(FLAGS.output, os.path.basename(path))
    os.makedirs(dest, exist_ok=True)

    files = ["dataset_statistics.json", "config.json", "example_batch.msgpack"]
    for file in files:
        cmd = base_cmd + " " + os.path.join(path, file) + " " + dest
        subprocess.run(cmd, shell=True, check=False)

    # Now download the checkpoint
    cmd = base_cmd + " -r " + os.path.join(path, str(step)) + " " + dest
    subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    app.run(main)
