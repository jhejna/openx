from typing import Dict

from openx.data.utils import RobotType, StateEncoding


def libero_dataset_transform(ep: Dict):
    observation = {
        "image": {"agent": ep["observation"]["agent_image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"]["ee_pos"],
            StateEncoding.EE_EULER: ep["observation"]["state"]["ee_euler"],
            StateEncoding.GRIPPER: ep["observation"]["state"]["gripper_qpos"][..., :1],
            StateEncoding.JOINT_POS: ep["observation"]["state"]["joint_pos"],
        },
        "language_instruction": ep["language_instruction"],
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., -1:],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]
    ep["demo_idx"] = ep["episode_metadata"]["demo_idx"]
    ep["task"] = ep["episode_metadata"]["task"]

    return ep
