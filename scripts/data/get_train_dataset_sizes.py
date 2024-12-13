import pprint

import tensorflow as tf
from absl import app, flags
from ml_collections import config_flags

from openx.data.core import load_dataset
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to a config file", lock_config=True)

"""
A simple script for getting the exact size of the train split of datasets.
"""


def main(_):
    dataset_sizes = {}
    for dataset_name, config in FLAGS.config.dataloader.datasets.to_dict().items():
        if not dataset_name.startswith("nyu"):
            continue
        assert "path" in config and "transform" in config
        transform_fn = ModuleSpec.instantiate(config["transform"])
        filter_fn = ModuleSpec.instantiate(config["filter"]) if config.get("filter", None) is not None else None
        path = config["path"]

        dataset = load_dataset(path, config["train_split"], standardization_transform=transform_fn, filter_fn=filter_fn)

        def _reduce_fn(state, ep):
            ep_len = tf.shape(tf.nest.flatten(ep)[0])[0] - 1
            return (state[0] + 1, state[1] + ep_len)

        num_ep, num_steps = dataset.reduce((0, 0), _reduce_fn)
        dataset_sizes[dataset_name] = dict(num_ep=num_ep.numpy(), num_steps=num_steps.numpy())

    pprint.pprint(dataset_sizes)


if __name__ == "__main__":
    app.run(main)
