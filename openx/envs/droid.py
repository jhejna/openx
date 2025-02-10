"""
This is a minimal & hackable script meant meant to implement a gym environment
equivalent to the DROID client with minimal dependencies.

INSTALLATION:
- Follow the instructions from this section:
  https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-laptopworkstation

STEPS:
1) Start the DROID server on the NUC - this has not changed

2) Discover ZED cameras to get their serial numbers (only need to do this once) and can be done with ZED_Explorer

"""

import time
from copy import deepcopy
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tft
from PIL import Image

from openx.data.utils import StateEncoding

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")

try:
    import zerorpc
except ModuleNotFoundError:
    print("WARNING: ZeroRPC not installed, can only be used in debug mode.")


class ZedCamera(object):
    def __init__(self, serial_number: str, mode: str, **kwargs):
        assert mode in {"left", "right", "both"}
        self._cam = sl.Camera()
        self._frame = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self.model = {"left": sl.VIEW.LEFT, "right": sl.VIEW.RIGHT, "both": sl.VIEW.SIDE_BY_SIDE}

        sl_params = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            camera_fps=60,
            camera_image_flip=sl.FLIP_MODE.OFF,
            depth_stabilization=False,
        )
        sl_params.set_from_serial_number(int(serial_number))
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")

    def get_frame(self):
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        # We will read at the full raw resolution as was done during data collection.
        self._cam.retrieve_image(self._frame, sl.VIEW.LEFT, resolution=sl.Resolution(0, 0))
        return deepcopy(self._frame.get_data())[..., :3]  # Remove Depth

    def __del__(self):
        self._cam.close()


class DebugZedCamera(object):
    def __init__(*args, **kwargs):
        pass

    def get_frame(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)


def rotation_6d_to_euler(d6: tf.Tensor) -> tf.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / tf.linalg.norm(a1, ord=2, axis=-1)
    b2 = a2 - tf.reduce_sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / tf.linalg.norm(b2, ord=2, axis=-1)
    b3 = tf.linalg.cross(b1, b2)
    mat = tf.stack((b1, b2, b3), axis=-2)
    return tft.euler.from_rotation_matrix(mat)


class DebugRobotServer(object):
    """
    This is a dummy class that implements the same methods as the droid server but
    returns dummy values. Designed for debugging pipelines.
    """

    def update_command(
        self, action, action_space="cartesian_position", gripper_action_space="position", blocking=False
    ):
        pass

    def update_gripper(self, action, velocity=False, blocking=False):
        pass

    def update_joints(self, joints, velocity=False, blocking=True, cartesian_noise=None):
        pass

    def get_robot_state(self):
        return {
            "cartesian_position": list(range(6)),
            "joint_positions": list(range(8)),
            "gripper_position": 0.5,
        }, None


class DroidEnv(gym.Env):
    def __init__(
        self,
        robot_action_space,
        gripper_action_space,
        cameras: Optional[Dict[str, str]] = None,
        resolution: Tuple[int, int] = (180, 320),
        ip_address="127.0.0.1",
        debug: bool = False,
    ):
        assert robot_action_space in [
            "cartesian_position",
            "cartesian_velocity",
        ], "Joint space control currently not supported"
        assert gripper_action_space in ["position"], "Does not currently support `velociy`."
        super().__init__()
        # Robot Configuration
        self.robot_action_space = robot_action_space
        self.gripper_action_space = gripper_action_space
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.control_hz = 15
        self._time = None

        # Set action space to match openx.datasets.oxe.droid_dataset_transform
        self.action_space = gym.spaces.Dict(
            dict(
                desired_delta=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
                desired_absolute=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_ROT6D: gym.spaces.Box(shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32),
                    }
                ),
            )
        )

        # Set the observation space to match. Allow the user to specify image keys via the `cameras` dict.
        image_spaces = dict()
        self.cameras = dict()
        self.resolution = resolution
        for name, serial_number in cameras.items():
            self.cameras[name] = ZedCamera(serial_number, mode="left") if not debug else DebugZedCamera()
            image_spaces[name] = gym.spaces.Box(shape=(*resolution, 3), dtype=np.uint8, low=0, high=255)

        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32),
                        StateEncoding.JOINT_POS: gym.spaces.Box(shape=(8,), low=-np.inf, high=np.inf, dtype=np.float32),
                    },
                ),
                image=gym.spaces.Dict(image_spaces),
            )
        )

        # Connect to the robot
        if debug:
            self.server = DebugRobotServer()
        else:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect("tcp://" + ip_address + ":4242")
            for i in range(2, 0, -1):
                try:
                    self.server.launch_controller()
                    self.server.launch_robot()
                except zerorpc.exceptions.RemoteError as err:
                    last_attempt = i == 0
                    if last_attempt:
                        raise err
                    time.sleep(0.1)

    def _get_observation(self):
        # the original code reads the state first, so we will do that here as well
        state, _ = self.server.get_robot_state()
        # We have to parse it out as is done in the RLDS converter
        state = {
            StateEncoding.EE_POS: np.array(state["cartesian_position"][:3], dtype=np.float32),
            StateEncoding.EE_EULER: np.array(state["cartesian_position"][3:6], dtype=np.float32),
            StateEncoding.JOINT_POS: np.array(state["joint_positions"], dtype=np.float32),
            StateEncoding.GRIPPER: np.array([state["gripper_position"]], dtype=np.float32),
        }
        images = {}
        for name, camera in self.cameras.items():
            # For consistency, we use the resize function from the RLDS processor
            # See https://github.com/kpertsch/droid_dataset_builder/blob/main/droid/droid.py
            # This, however, is one of the slowest resize operations :( -- cv2 is much faster.
            image = Image.fromarray(camera.get_frame())
            image = np.array(image.resize(self.resolution, resample=Image.BICUBIC))
            images[name] = image[..., ::-1]  # Flip the images as done in RLDS converter

        return dict(state=state, image=images)

    def step(self, action):
        if self.robot_action_space == "cartesian_position":
            if StateEncoding.EE_ROT6D in action["desired_absolute"]:
                # Handle the case of R6 actions separately.
                rot = rotation_6d_to_euler(action["desired_absolute"][StateEncoding.EE_ROT6D])
            else:
                rot = action["desired_absolute"][StateEncoding.EE_EULER]
            pos = action["desired_absolute"][StateEncoding.EE_POS]
        elif self.robot_action_space == "cartesian_velocity":
            pos = action["desired_delta"][StateEncoding.EE_POS]
            rot = action["desired_delta"][StateEncoding.EE_EULER]
        else:
            raise ValueError("Invalid robot action space specified.")

        if self.gripper_action_space == "position":
            gripper = action["desired_absolute"][StateEncoding.GRIPPER]
        else:
            raise ValueError("Invalid gripper action space specified")

        # Step the environment #
        action = np.concatenate((pos, rot, gripper), axis=-1)
        # Calling the args EXACTLY like this is extremely important
        self.server.update_command(action.tolist(), self.robot_action_space, False)

        # Regularize Control Frequency #
        comp_time = time.time() - self._time
        sleep_left = (1 / self.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        # TODO: consider moving the timing point to after the obs is read
        self._time = time.time()
        # Return obs, reward, done, trunc, info
        return self._get_observation(), 0.0, False, False, {}

    def reset(self, seed=None, options=None, randomize=False):
        super().reset(seed=seed)
        self.server.update_gripper(0, velocity=False, blocking=True)
        noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high).tolist() if randomize else None

        # Calling the args EXACTLY like this is extremely important
        self.server.update_joints(self.reset_joints.tolist(), False, True, noise)

        time.sleep(1.5)
        obs = self._get_observation()
        self._time = time.time()
        return obs, {}
