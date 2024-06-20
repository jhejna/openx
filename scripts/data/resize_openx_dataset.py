"""
Code for parallel resizing of tfds datasets.

Please make sure to pip install the rlds_dataset_mod repo from Karl Pertsch
https://github.com/kpertsch/rlds_dataset_mod.git
"""

import os
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from rlds_dataset_mod.multithreaded_adhoc_tfds_builder import MultiThreadedAdhocDatasetBuilder

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "/Users/jhejna/tensorflow_datasets/bridge/1.0.0", "Path to raw datasets")
flags.DEFINE_string("data_dir", "test", "Path to save dataset")
flags.DEFINE_string("mods", "", "String controlling extra mods, specifically success filter and image channels")
flags.DEFINE_integer("num_workers", 2, "Number of parallel works")
flags.DEFINE_integer("chunk_size", 4, "Number of episodes per worker. Controls mem usage")

RES = 256


def is_depth_feature(name, feature):
    return is_image_feature(feature) and "depth" in name


def is_image_feature(feature):
    if len(feature.shape) != 2 and len(feature.shape) != 3:
        return False
    if feature.shape[0] < 64 or feature.shape[1] < 64:
        return False
    return True


def preprocess_features(features: tfds.features.FeaturesDict) -> tfds.features.FeaturesDict:
    """
    Does four things on the features dict.
    1. Filter by success
    2. Removes Depth
    3. Resizes Images
    4. Flips image channels if needed
    This order in particular should be the fastest since we eliminate the most ops
    """
    # 1. Filter by success: No Op
    obs_features = features["steps"]["observation"]
    # 2. Remove Depth
    obs_features = {k: v for k, v in obs_features.items() if not is_depth_feature(k, v)}
    # 3. Resize images
    image_keys = [k for k, v in obs_features.items() if is_image_feature(v)]
    for k in image_keys:
        # preserve the aspect ratio
        h, w = obs_features[k].shape[:2]
        new_shape = (RES, int(RES / h * w)) if w > h else (int(RES / w * h), RES)
        new_shape = new_shape + obs_features[k].shape[2:]
        # JPEG encode EVERYTHING.
        assert obs_features[k].dtype == tf.uint8
        obs_features[k] = tfds.features.Image(
            shape=new_shape, dtype=tf.uint8, encoding_format="jpeg", doc=obs_features[k].doc
        )
    # 4. Flip Image channels: No Op

    # Return the modified feature dict
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(obs_features),
                    **{k: features["steps"][k] for k in features["steps"].keys() if k != "observation"},
                }
            ),
            **{k: features[k] for k in features.keys() if k != "steps"},
        }
    )


def preprocess_dataset(
    ds: tf.data.Dataset,
    filter_success: bool = False,
    flip_image_channels: bool = False,
    flip_wrist_image_channels: bool = False,
) -> tf.data.Dataset:
    # 1. Filter by success
    if filter_success:
        ds = ds.filter(lambda e: e["success"])

    # This does everything in one parallel call instead of multiple.
    def _preprocess(step):
        # 2. Remove Depth by filtering to just the keys we care about
        observation = {k: v for k, v in step["observation"].items() if not is_depth_feature(k, v)}
        image_keys = [k for k, v in observation.items() if is_image_feature(v)]

        for k in image_keys:
            image = observation[k]
            # 3. Resize images
            h, w = image.shape[:2]
            new_shape = (RES, int(RES / h * w)) if w > h else (int(RES / w * h), RES)
            assert image.dtype == tf.uint8
            image = tf.image.resize(image, new_shape, method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)

            # 4. Flip channels if desired.
            if (flip_image_channels and k == "image") or (
                flip_wrist_image_channels and k in ("wrist_image", "hand_image")
            ):
                image = image[..., ::-1]  # BGR to RGB
            observation[k] = image

        step["observation"] = observation
        return step

    def _ep_map(ep):
        ep["steps"] = ep["steps"].map(_preprocess)
        return ep

    return ds.map(_ep_map)


def dataset_generator(builder, split, **kwargs):
    """Modifies dataset features."""
    ds = builder.as_dataset(split=split)
    ds = preprocess_dataset(ds, **kwargs)
    for episode in tfds.core.dataset_utils.as_numpy(ds):
        yield episode


def main(_):
    # Preprocess all of the openx data
    builder = tfds.builder_from_directory(builder_dir=FLAGS.dataset_path)
    features = preprocess_features(builder.info.features)

    # Grab the mods from the extra mod string.
    mods = FLAGS.mods.split(",")
    kwargs = dict(
        filter_success="filter_success" in mods,
        flip_image_channels="flip_image_channels" in mods,
        flip_wrist_image_channels="flip_wrist_image_channels" in mods,
    )

    dataset_path = os.path.normpath(FLAGS.dataset_path)
    name = os.path.basename(os.path.dirname(dataset_path)) + "_preprocessed"

    tf.io.gfile.makedirs(os.path.join(FLAGS.data_dir, name))

    mod_dataset_builder = MultiThreadedAdhocDatasetBuilder(
        name=name,
        version=builder.version,
        features=features,
        split_datasets={split: builder.info.splits[split] for split in builder.info.splits},
        config=builder.builder_config,
        data_dir=FLAGS.data_dir,
        description=builder.info.description,
        generator_fcn=partial(dataset_generator, builder=builder, **kwargs),
        n_workers=FLAGS.num_workers,
        max_episodes_in_memory=FLAGS.chunk_size,
    )
    mod_dataset_builder.download_and_prepare()


if __name__ == "__main__":
    app.run(main)
