"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils

from openx.data.datasets.libero import LIBERO_TASK_IDS
from openx.data.utils import StateEncoding


class LiberoEnv(gym.Env):
    def __init__(
        self,
        benchmark: str,
        task: str,
        horizon: Optional[int] = None,
        terminate_early: bool = False,
    ):
        self.terminate_early = terminate_early
        benchmark = libero_benchmark.get_benchmark_dict()[benchmark]()
        task_idx = benchmark.get_task_names().index(task)
        self.env_lang = benchmark.get_task(task_idx).language
        self.task_id = LIBERO_TASK_IDS[task]

        env_args = {
            "bddl_file_name": benchmark.get_task_bddl_file_path(task_idx),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        if horizon is not None:
            env_args["horizon"] = horizon
        self.env = OffScreenRenderEnv(**env_args)

        # Only enable observables we will use.
        observables = {
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        }
        for ob_name in self.env.env.observation_names:
            modifier = ob_name in observables
            self.env.env.modify_observable(observable_name=ob_name, attribute="enabled", modifier=modifier)
            self.env.env.modify_observable(observable_name=ob_name, attribute="active", modifier=modifier)

        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.JOINT_POS: gym.spaces.Box(shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
                image=gym.spaces.Dict(
                    dict(
                        agent=gym.spaces.Box(shape=(128, 128, 3), dtype=np.uint8, low=0, high=255),
                        wrist=gym.spaces.Box(shape=(128, 128, 3), dtype=np.uint8, low=0, high=255),
                    )
                ),
                task_id=gym.spaces.Discrete(n=len(LIBERO_TASK_IDS)),
                language_instruction=gym.spaces.Text(max_length=500),
            )
        )

        low, high = self.env.env.action_spec
        self.action_space = gym.spaces.Dict(
            dict(
                desired_delta=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=low[:3], high=high[:3], dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(
                            shape=(3,), low=low[3:6], high=high[3:6], dtype=np.float32
                        ),
                    }
                ),
                desired_absolute=gym.spaces.Dict(
                    {StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=low[-1:], high=high[-1:], dtype=np.float32)}
                ),
            )
        )

    def step(self, action):
        action = np.concatenate(
            (
                action["desired_delta"][StateEncoding.EE_POS],
                action["desired_delta"][StateEncoding.EE_EULER],
                action["desired_absolute"][StateEncoding.GRIPPER],
            ),
            axis=-1,
        )

        low, high = self.env.env.action_spec
        action = np.clip(action, a_min=low, a_max=high)
        # NOTE: libero returns `check_success` instead of actually returning done *facepalm*
        # https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py#L807
        obs, reward, success, info = self.env.step(action)
        done = self.env.env.done
        if self.terminate_early and success:
            done = True
        info["success"] = success
        # Never terminate robot envs, but do truncate them.
        return self._format_obs(obs), reward, False, done, info

    def render(self, mode="human", height=None, width=None, camera_name=None):
        camera_name = "agentview"
        return self.env.sim.render(height=128, width=128, camera_name=camera_name)[::-1]

    def _format_obs(self, obs):
        return dict(
            state={
                StateEncoding.EE_POS: obs["robot0_eef_pos"],
                StateEncoding.EE_EULER: transform_utils.quat2axisangle(obs["robot0_eef_quat"]),
                StateEncoding.GRIPPER: obs["robot0_gripper_qpos"][..., :1],
                StateEncoding.JOINT_POS: obs["robot0_joint_pos"],
            },
            image=dict(agent=np.flipud(obs["agentview_image"]), wrist=np.flipud(obs["robot0_eye_in_hand_image"])),
            language_instruction=self.env_lang,
            task_id=self.task_id,
        )

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        return self._format_obs(obs), dict(success=False)
