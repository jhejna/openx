import functools
import os
import pprint

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache

from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy, load_checkpoint
from openx.utils.spec import ModuleSpec

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

    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.path, FLAGS.checkpoint_step)
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
            vec_env_cls = gym.vector.AsyncVectorEnv if FLAGS.n_eval_proc > 1 else gym.vector.SyncVectorEnv
            env = vec_env_cls([env_fn for _ in range(FLAGS.n_eval_proc)], context="spawn", shared_memory=True)
            eval_metrics = eval_policy(env, functools.partial(jitted_predict, state), rng, num_ep=FLAGS.num_ep)
            eval_metrics["num_ep"] = next(iter(eval_metrics.values())).shape[0]
            print("#########", env_name, "#########")
            eval_metrics = jax.tree.map(lambda x: np.mean(x), eval_metrics)
            pprint.pprint(eval_metrics)


if __name__ == "__main__":
    app.run(main)
