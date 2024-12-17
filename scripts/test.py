import functools
import json
import os
import pprint

import flax
import gymnasium as gym
import jax
import optax
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache
from ml_collections import ConfigDict
from orbax import checkpoint

from openx.data.core import load_dataset_statistics
from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy
from openx.utils.spec import ModuleSpec, recursively_instantiate

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "/tmp/", "Path to save logs and checkpoints.")
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step to load.")
flags.DEFINE_integer("n_eval_proc", 1, "Number of eval processes")
flags.DEFINE_integer("num_ep", 10, "Number of episodes")


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Load the example batch
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.path, "example_batch.msgpack"), "rb") as f:
        example_batch = flax.serialization.msgpack_restore(f.read())

    # Load the dataset statistics
    dataset_statistics = load_dataset_statistics(FLAGS.path, "dataset_statistics.json")

    # Load the config
    with tf.io.gfile.GFile(tf.io.gfile.join(FLAGS.path, "config.json"), "r") as f:
        config = json.load(f)
        config = ConfigDict(config)

    # Instantiate the model
    alg = recursively_instantiate(config.alg.to_dict())
    tx = optax.set_to_zero()  # Dummy optimizer without state.
    rng = jax.random.key(config.seed)

    state = alg.init(example_batch, tx, rng)
    checkpointer = checkpoint.CheckpointManager(FLAGS.path, checkpoint.PyTreeCheckpointer())
    step = FLAGS.checkpoint_step if FLAGS.checkpoint_step is not None else checkpointer.latest_step()
    params = checkpointer.restore(step, state.params)
    state = state.replace(params=params)

    ### Define the Predict Function ###
    jitted_predict = jax.jit(alg.predict)

    ### Setup Eval Envs ###
    if config.get("envs", None) is not None and len(config.envs) > 0:
        structure = config.structure.to_dict()
        n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
        scale_range = config.dataloader.augment_kwargs.get("scale_range", None)
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
                scale_range=scale_range,
            )

        for env_name, env_spec in config.envs.to_dict().items():
            env_fn = functools.partial(
                _make_env, fn=ModuleSpec.instantiate(env_spec), stats=dataset_statistics[env_name]
            )
            vec_env_cls = gym.vector.AsyncVectorEnv if FLAGS.n_eval_proc > 1 else gym.vector.SyncVectorEnv
            env = vec_env_cls([env_fn for _ in range(FLAGS.n_eval_proc)], context="spawn", shared_memory=True)
            eval_metrics = eval_policy(env, functools.partial(jitted_predict, state), rng, num_ep=FLAGS.num_ep)
            eval_metrics["num_ep"] = next(iter(eval_metrics.values())).shape[0]

            print("#########", env_name, "#########")
            pprint.pprint(eval_metrics)


if __name__ == "__main__":
    app.run(main)
