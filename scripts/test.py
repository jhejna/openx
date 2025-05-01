import functools
import os
import pprint

import gymnasium as gym
import imageio
import jax
import numpy as np
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache

from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy, load_checkpoint
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "/tmp/", "Path to the checkpoint.")
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step to load.")
flags.DEFINE_integer("n_eval_proc", 1, "Number of eval processes")
flags.DEFINE_integer("num_ep", 10, "Number of episodes")
flags.DEFINE_string("image_key", None, "The image key for saving gifs.")
flags.DEFINE_string("path", None, "Path to save videos.")


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.checkpoint, FLAGS.checkpoint_step)
    rng = jax.random.key(config.seed)

    ### Define the Predict Function ###
    jitted_predict = jax.jit(alg.predict)

    ### Setup Eval Envs ###
    if config.get("envs", None) is not None and len(config.envs) > 0:
        structure = config.structure.to_dict()
        n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
        augment_kwargs = config.dataloader.to_dict().get("augment_kwargs", dict())
        exec_horizon = config.exec_horizon

        def _make_env(fn, stats):
            env = fn()
            return wrap_env(
                env,
                structure=structure,
                dataset_statistics=stats,
                n_obs=n_obs,
                n_action=n_action,
                exec_horizon=exec_horizon,
                augment_kwargs=augment_kwargs,
            )

        for env_name, env_spec in config.envs.to_dict().items():
            env_fn = functools.partial(
                _make_env, fn=ModuleSpec.instantiate(env_spec), stats=dataset_statistics[env_name]
            )
            vec_env_cls = (
                functools.partial(gym.vector.AsyncVectorEnv, context="spawn", shared_memory=True)
                if FLAGS.n_eval_proc > 1
                else gym.vector.SyncVectorEnv
            )
            env = vec_env_cls([env_fn for _ in range(FLAGS.n_eval_proc)])
            eval_metrics, videos = eval_policy(
                env, functools.partial(jitted_predict, state), rng, num_ep=FLAGS.num_ep, image_key=FLAGS.image_key
            )
            # Save Videos
            if len(videos) > 0:
                path = FLAGS.path if FLAGS.path is not None else "."
                # Parse the checkpoint name
                ckpt_name = FLAGS.checkpoint
                ckpt_name = ckpt_name[:-1] if ckpt_name.endswith("/") else ckpt_name
                if os.path.basename(ckpt_name).isdigit():
                    ckpt_name = "_".join(ckpt_name.split("/")[-2:])
                else:
                    ckpt_name = ckpt_name.split("/")[-1]
                tf.io.gfile.makedirs(os.path.join(path, ckpt_name))
                for i, video in enumerate(videos):
                    imageio.mimsave(os.path.join(path, ckpt_name, f"ep_{i}.gif"), video)

            eval_metrics["num_ep"] = next(iter(eval_metrics.values())).shape[0]
            print("#########", env_name, "#########")
            eval_metrics = jax.tree.map(lambda x: np.mean(x), eval_metrics)
            pprint.pprint(eval_metrics)


if __name__ == "__main__":
    app.run(main)
