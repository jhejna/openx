"""
Script for evaluation on a Franka Panda robot.
Designed for use with the robot-lightning repository on the panda branch:

https://github.com/jhejna/robot-lightning/tree/panda
"""

import os
import pickle
from datetime import datetime

import cv2
import imageio
import jax
import numpy as np
import tensorflow as tf
import yaml
from absl import app, flags
from jax.experimental import compilation_cache

from openx.envs.franka import FrankaEnv
from openx.envs.wrappers import preprocess_goal, wrap_env
from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Path to checkpoint folder.")
flags.DEFINE_string("checkpoint_step", None, "Checkpoint step to load.")
flags.DEFINE_string("robot_config_path", None, "Path to robot config")
flags.DEFINE_string("goal_path", None, "Path to a goal state")
flags.DEFINE_bool("show_image", True, "Whether or not to display the robot images.")
flags.DEFINE_string("video_save_path", None, "whether or not to save videos.")
flags.DEFINE_string("task", None, "description of current task")
flags.DEFINE_integer("max_steps", 100, "Maximum number of steps to run the robot before terminating.")


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Load
    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.path, FLAGS.checkpoint_step)
    rng = jax.random.PRNGKey(config.seed)

    ### Define the Predict Function ###
    @jax.jit
    def predict(obs, goal, rng):
        batch = dict(observation=obs, goal=goal)
        batch = jax.tree.map(lambda x: x[None], batch)
        action = alg.predict(state, batch, rng)
        return jax.tree.map(lambda x: x[0], action)

    ### Setup Eval Envs ###
    structure = config.structure.to_dict()
    n_obs, n_action = config.dataloader.n_obs, config.dataloader.n_action
    scale_range = config.dataloader.augment_kwargs.get("scale_range", None)
    # Determine if we are using the Octo resized dataset

    # set up the franka client
    with open(FLAGS.robot_config_path, "r") as f:
        robot_config = yaml.load(f, Loader=yaml.Loader)
    env = FrankaEnv(**robot_config)

    env = wrap_env(
        env,
        structure=structure,
        dataset_statistics=dataset_statistics,
        n_obs=n_obs,
        n_action=n_action,
        exec_horizon=max(1, n_action // 2),
        scale_range=scale_range,
    )

    if FLAGS.goal_path is not None:
        with open(FLAGS.goal_path, "rb") as f:
            goal = pickle.load(f)
    else:
        input("Press [Enter] when ready for taking the goal image. ")
        goal, _ = env.unwrapped.reset()
        # save the goal
        goal_name = input("Enter a name for the goal:\n")
        with open(f"goal_states/{goal_name}.pkl", "wb") as f:
            pickle.dump(goal, f)

    goal = preprocess_goal(goal, structure, dataset_statistics, scale_range)

    obs, _ = env.reset()
    while input("Quit?\n") != "q":
        image = (255 * obs["image"]["agent"][-1]).astype(np.uint8)
        steps = 0
        done, trunc = False, False

        images = [image]
        try:
            while not done and not trunc:
                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(10)

                rng = jax.random.fold_in(rng, steps)
                action = predict(obs, goal, rng)
                obs, reward, done, trunc, info = env.step(action)
                image = (255 * obs["image"]["agent"][-1]).astype(np.uint8)
                images.append(image)

                steps += 1
                trunc = trunc or steps == FLAGS.max_steps
        except KeyboardInterrupt:
            print("Ended early")

        if FLAGS.video_save_path is not None:
            checkpoint_name = (
                (FLAGS.path).split("/")[-2] if len((FLAGS.path).split("/")[-1]) < 2 else (FLAGS.path).split("/")[-1]
            )
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            os.makedirs(FLAGS.video_save_path + "/" + checkpoint_name + "/" + FLAGS.task, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                checkpoint_name,
                FLAGS.task,
                f"{curr_time}.mp4",
            )
            video = np.stack(images)
            imageio.mimsave(save_path, video, fps=1.0 / 0.1 * 3)
            obs, _ = env.reset()


if __name__ == "__main__":
    app.run(main)
