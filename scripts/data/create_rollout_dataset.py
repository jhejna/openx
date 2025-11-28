import functools
import os

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from jax.experimental import compilation_cache

from openx.data.utils import StateEncoding
from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import load_checkpoint
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None, "Path to the checkpoint", required=True)
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step to load.")
flags.DEFINE_string("path", None, "Path to save the dataset", required=True)
flags.DEFINE_enum("dataset_type", None, ["robomimic"], "Type of dataset to create.")
flags.DEFINE_integer("ep_idx", None, "The beginning of the ep_idx value. Defaults to size of the previous dataset.")
flags.DEFINE_integer("max_ep_per_file", 50, "Max number of episodes per record.")
flags.DEFINE_string("env", None, "Environment to load.")
flags.DEFINE_integer("n_eval_proc", 1, "Number of eval processes")
flags.DEFINE_integer("num_ep", 10, "Number of episodes")


def _make_step_robomimic(obs, action, reward, is_first: bool, is_last: bool, is_terminal: bool):
    # Define the action
    action = np.concatenate(
        (
            action["desired_delta"][StateEncoding.EE_POS],
            action["desired_delta"][StateEncoding.EE_EULER],
            action["desired_absolute"][StateEncoding.GRIPPER],
        ),
        axis=-1,
    )
    # We need to pad gripper qpos to two values
    observation = dict(
        agent_image=obs["image"]["agent"],
        wrist_image=obs["image"]["wrist"],
        state=dict(
            ee_pos=obs["state"][StateEncoding.EE_POS].astype(np.float32),
            ee_quat=obs["state"][StateEncoding.EE_QUAT].astype(np.float32),
            gripper_qpos=np.concatenate(
                (obs["state"][StateEncoding.GRIPPER], obs["state"][StateEncoding.GRIPPER]), axis=-1
            ).astype(np.float32),
            joint_pos=obs["state"][StateEncoding.JOINT_POS].astype(np.float32),
            joint_vel=obs["state"][StateEncoding.JOINT_VEL].astype(np.float32),
            object=obs["state"][StateEncoding.MISC].astype(np.float32),
        ),
    )
    return dict(
        observation=observation,
        action=action,
        discount=1.0,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        language_instruction="",
    )


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # TODO: handle sharding. Currently does not.
    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.checkpoint, FLAGS.checkpoint_step)
    rng = jax.random.key(config.seed)

    # Get the environment
    if FLAGS.env is None:
        if len(config.envs) > 1:
            raise ValueError(
                "Trained on multiple datasets, must provide a dataset, choose one of: " + ", ".join(config.envs.keys())
            )
        env_name = next(iter(config.envs.keys()))
    else:
        env_name = FLAGS.env

    dataset_path = config.dataloader.datasets.to_dict()[env_name]["path"]

    # Define the dataset spec
    if isinstance(dataset_path, list):
        builder = tfds.builder_from_directories(builder_dir=dataset_path)
    else:
        builder = tfds.builder_from_directory(builder_dir=dataset_path)

    tf.io.gfile.makedirs(FLAGS.path)

    ds_identity = tfds.core.dataset_info.DatasetIdentity(
        name=builder.info.name + "_rollouts", version=tfds.core.Version("1.0.0"), data_dir=FLAGS.path, module_name=""
    )

    ds_info = tfds.core.DatasetInfo(
        builder=ds_identity,
        description=builder.info.description,
        features=builder.info.features,
        supervised_keys=builder.info.supervised_keys,
        homepage=builder.info.homepage,
        citation=builder.info.citation,
        metadata=builder.info.metadata,
    )

    sequential_writer = tfds.core.SequentialWriter(
        ds_info,
        FLAGS.max_ep_per_file,
        overwrite=False,
        file_format=builder.info.file_format,
    )
    sequential_writer.initialize_splits(["train"], fail_if_exists=False)

    ep_idx = (
        sum(builder.info.splits[split].num_examples for split in builder.info.splits)
        if FLAGS.ep_idx is None
        else FLAGS.ep_idx
    )

    ### Define the Predict Function ###
    jitted_predict = jax.jit(alg.predict)

    ### Setup Eval Envs ###
    structure = config.structure.to_dict()
    n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
    augment_kwargs = config.dataloader.to_dict().get("augment_kwargs", dict())
    exec_horizon = config.exec_horizon
    config.envs[env_name].kwargs.terminate_early = True  # Make sure that we terminate early.

    env_fn = ModuleSpec.instantiate(config.envs[env_name])
    stats = dataset_statistics[env_name]

    def _make_env():
        return wrap_env(
            env_fn(),
            structure=structure,
            dataset_statistics=stats,
            n_obs=n_obs,
            n_action=n_action,
            exec_horizon=exec_horizon,  # For now only allow an exec horizon of 1 to simplify logic.
            augment_kwargs=augment_kwargs,
            add_raw=True,  # Add the original observations and actions back.
        )

    vec_env_cls = (
        functools.partial(gym.vector.AsyncVectorEnv, context="spawn", shared_memory=True)
        if FLAGS.n_eval_proc > 1
        else gym.vector.SyncVectorEnv
    )
    env = vec_env_cls([_make_env for _ in range(FLAGS.n_eval_proc)])

    steps = {i: [] for i in range(env.num_envs)}
    obs, info = env.reset()

    ep_success = np.zeros((env.num_envs,), dtype=np.bool_)
    invalid = np.zeros((env.num_envs,), dtype=bool)  # Starts at all false

    # If you want to make a rollout dataset, this must be properly defined.
    # It might be different for each environment.
    make_step_fn = {
        "robomimic": _make_step_robomimic
    }[FLAGS.dataset_type]

    count, num_ep = 0, 0
    while num_ep < FLAGS.num_ep:
        count += 1
        rng = jax.random.fold_in(rng, count)
        batch = dict(observation=obs)
        action = jitted_predict(state, batch, rng=rng)
        action = np.asarray(action)  # Must convert away from jax tensor.

        first_raw_obs = [info["raw_obs"][i][-1] for i in range(env.num_envs)]
        next_obs, reward, term, trunc, info = env.step(action)
        raw_actions = info["raw_action"]
        # Because of the RHC wrapper we actually have lists of things! Merge next_obs with first.
        raw_obses = [[first_raw_obs[i], *info["raw_obs"][i]] for i in range(env.num_envs)]

        # Track the success rate of the episode.
        ep_success = np.logical_or(ep_success, info["success"])

        # Add all of the steps for (obs, action, reward)
        for i in range(env.num_envs):
            if invalid[i]:
                continue

            # The observation list should always be one longer than the action list.
            assert len(raw_obses[i]) == len(raw_actions[i]) + 1

            for o, a in zip(raw_obses[i], raw_actions[i], strict=False):
                # If not invalid from a reset, make the step
                step = make_step_fn(
                    o,
                    a,
                    reward=reward[i],  # NOTE: Rewards are broken due to RHCWrapper.
                    is_first=len(steps[i]) == 0,
                    is_last=False,
                    is_terminal=False,
                )
                steps[i].append(step)

            if term[i] or trunc[i]:
                # Add the final step.
                final_step = make_step_fn(
                    raw_obses[i][-1],
                    raw_actions[i][-1],
                    reward=reward[i],  # NOTE: Rewards are broken due to RHCWrapper.
                    is_first=False,
                    is_last=True,
                    is_terminal=term[i],
                )
                steps[i].append(final_step)

                if ep_success[i]:
                    episode = dict(
                        steps=steps[i],
                        episode_metadata=dict(
                            operator="",
                            ep_idx=ep_idx,
                            quality_score=-1,
                            file_path="",
                        ),
                    )

                    sequential_writer.add_examples({"train": [episode]})
                    num_ep += 1
                    ep_idx += 1

                    print("Finished episode", num_ep)

                # Reset this stream.
                steps[i] = []

        invalid = np.logical_or(term, trunc)
        obs = next_obs

    sequential_writer.close_all()  # Close the writer.


if __name__ == "__main__":
    app.run(main)