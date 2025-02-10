import json
import os
from typing import Callable, Dict

import flax
import gymnasium as gym
import jax
import numpy as np
import optax
import tensorflow as tf
from ml_collections import ConfigDict
from orbax import checkpoint

from openx.data.core import load_dataset_statistics
from openx.utils.spec import recursively_instantiate


def load_checkpoint(path: str, step: int | None = None, sharding: jax.sharding.Sharding | None = None):
    if not path.startswith("gs://"):
        path = os.path.abspath(path)
    path = path[:-1] if path.endswith("/") else path
    if os.path.basename(path).isdigit():
        assert step is None, "Provided a checkpoint step, but it was already present in the path."
        # The checkpoint step is included in the path, so get the path from there
        step = int(os.path.basename(path))
        path = os.path.dirname(path)

    with tf.io.gfile.GFile(tf.io.gfile.join(path, "example_batch.msgpack"), "rb") as f:
        example_batch = flax.serialization.msgpack_restore(f.read())

    # Load the dataset statistics
    dataset_statistics = load_dataset_statistics(path)

    # Load the config
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "config.json"), "r") as f:
        config = json.load(f)
        config = ConfigDict(config)

    # Instantiate the model
    alg = recursively_instantiate(config.alg.to_dict())
    tx = optax.set_to_zero()  # Dummy optimizer without state.
    rng = jax.random.key(config.seed)

    state = alg.init(example_batch, tx, rng)
    if sharding is not None:
        # If sharding is supplied, shard the state and add restore args for every item.
        state = jax.tree.map(lambda x: jax.device_put(x, sharding), state)
        restore_kwargs = {
            "restore_args": checkpoint.checkpoint_utils.construct_restore_args(
                state.params, jax.tree.map(lambda _: sharding, state.params)
            )
        }
    else:
        restore_kwargs = {}

    checkpointer = checkpoint.CheckpointManager(path, checkpoint.PyTreeCheckpointer())
    step = step if step is not None else checkpointer.latest_step()
    params = checkpointer.restore(step, state.params, restore_kwargs=restore_kwargs)
    state = state.replace(params=params)

    return alg, state, dataset_statistics, config


def eval_policy(
    env: gym.Env,
    predict: Callable,
    rng: jax.random.PRNGKey,
    num_ep: int = 10,
) -> Dict:
    if not isinstance(env, gym.vector.VectorEnv):
        env = gym.vector.SyncVectorEnv(lambda: env)
    num_envs = env.num_envs

    rewards, lengths, successes = [], [], []
    ep_length = np.zeros((num_envs,), dtype=np.int32)
    ep_reward = np.zeros((num_envs,), dtype=np.float32)
    ep_success = np.zeros((num_envs,), dtype=np.bool_)
    obs, info = env.reset()
    steps = 0

    while len(rewards) < num_ep:
        steps += 1
        rng = jax.random.fold_in(rng, steps)
        batch = dict(observation=obs)
        action = predict(batch, rng=rng)
        action = np.asarray(action)  # Must convert away from jax tensor.
        obs, reward, done, trunc, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        if "success" in info:
            ep_success = np.logical_or(ep_success, info["success"])

        # Determine if we are done.
        for i in range(num_envs):
            if done[i] or trunc[i]:
                rewards.append(ep_reward[i])
                lengths.append(ep_length[i])
                # Need to manually check for success
                success = ep_success[i]
                if "final_info" in info:
                    success = success or info["final_info"][i]["success"]
                successes.append(success)
                ep_reward[i] = 0.0
                ep_length[i] = 0
                ep_success[i] = False

    return dict(reward=np.array(rewards), success=np.array(successes), length=np.array(lengths))
