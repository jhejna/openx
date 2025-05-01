import imageio
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from openx.data.dataloader import make_dataloader

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "vis.mp4", "The path to save the results if desired.", required=False)
flags.DEFINE_string("dataset", "libero_90", "The name of the dataset we are visualizing", required=False)
flags.DEFINE_string("image_key", "agent", "The name of the image to save viz of", required=False)
flags.DEFINE_integer("frame_skip", 2, "The level of frame skip", required=False)
flags.DEFINE_integer("num", 100, "The number of samples we are visualizing", required=False)
config_flags.DEFINE_config_file(
    "config", None, "File path to the training hyperparameter configuration.", lock_config=True
)


def main(_):
    dataloader_config = FLAGS.config.dataloader.to_dict()
    assert FLAGS.dataset in dataloader_config["datasets"]
    dataloader_config["datasets"] = {FLAGS.dataset: dataloader_config["datasets"][FLAGS.dataset]}
    dataloader_config["shuffle_size"] = 0
    dataloader_config["repeat"] = False
    dataloader_config["n_obs"] = 1
    dataloader_config["drop_remainder"] = False
    if "augment_kwargs" in dataloader_config:
        dataloader_config["augment_kwargs"]["train"] = False
    # We are only visualizing, so set dummy statistics here.
    if dataloader_config.get("global_dataset_statistics") is not None:
        dataloader_config["global_dataset_statistics"] = [FLAGS.dataset]

    ds, _, _, _ = make_dataloader(**dataloader_config, structure=FLAGS.config.structure.to_dict(), split_for_jax=True)
    all_imgs = []
    for batch in tqdm.tqdm(ds.take(FLAGS.num), total=FLAGS.num):
        images = batch["observation"]["image"][FLAGS.image_key].numpy()
        # Cut the images down temporally.
        images = images[:: FLAGS.frame_skip, -1]
        images = (images * 255).astype(np.uint8)
        all_imgs.append(images)

    all_imgs = np.concatenate(all_imgs, axis=0)

    imageio.mimsave(FLAGS.path, all_imgs)


if __name__ == "__main__":
    app.run(main)
