import tensorflow_datasets as tfds
from absl import app, flags
from ml_collections import config_flags

from openx.data.dataloader import make_dataloader

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to a config file", lock_config=True)

"""
Benchmark a TFDS dataset.
"""


def main(_):
    # Take the dataloader config
    dataloader_config = FLAGS.config.dataloader.to_dict()

    # Do not load any val datasets.
    for ds in dataloader_config["datasets"]:
        if "val_split" in dataloader_config["datasets"][ds]:
            del dataloader_config["datasets"][ds]["val_split"]

    ds, _, _, _ = make_dataloader(**dataloader_config, structure=FLAGS.config.structure.to_dict(), split_for_jax=False)

    statistics = tfds.benchmark(ds, batch_size=dataloader_config["batch_size"], num_iter=1000)

    print(statistics)


if __name__ == "__main__":
    app.run(main)
