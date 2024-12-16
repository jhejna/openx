import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class SquareMh(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "agent_image": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.FeaturesDict(
                                        {
                                            "ee_pos": tfds.features.Tensor(
                                                shape=(3,), dtype=np.float32, doc="Robot EEF Position"
                                            ),
                                            "ee_quat": tfds.features.Tensor(
                                                shape=(4,), dtype=np.float32, doc="Robot EEF Quat"
                                            ),
                                            "gripper_qpos": tfds.features.Tensor(
                                                shape=(2,), dtype=np.float32, doc="Robot EEF Quat"
                                            ),
                                            "joint_pos": tfds.features.Tensor(
                                                shape=(7,),
                                                dtype=np.float32,
                                                doc="Robot joint angles.",
                                            ),
                                            "joint_vel": tfds.features.Tensor(
                                                shape=(7,),
                                                dtype=np.float32,
                                                doc="Robot joint angles.",
                                            ),
                                            "object": tfds.features.Tensor(
                                                shape=(14,),
                                                dtype=np.float32,
                                                doc="Ground truth object position.",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot EEF action.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount if provided, default to 1."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32, doc="Reward if provided, 1 on final step for demos."
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        Define filepaths for data splits.
        Modify this at each call.
        """
        dataset_path = os.path.join(dl_manager.manual_dir, "image.hdf5")
        if "square" in dataset_path:
            language_instruction = "Put the square peg on the round hole."
        elif "can" in dataset_path:
            language_instruction = "Pick up the can and move it to the bin."
        elif "lift" in dataset_path:
            language_instruction = "Lift the block."
        else:
            raise ValueError("Unknown Task.")

        return {
            "train": self._generate_examples(
                path=dataset_path,
                language_instruction=language_instruction,
                train=True,
            ),
            "val": self._generate_examples(
                path=dataset_path,
                language_instruction=language_instruction,
                train=False,
            ),
        }

    def _generate_examples(self, path: str, language_instruction: str, train: bool = True) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        f = h5py.File(path, "r")
        if train:
            # Extract the training demonstrations
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/train"][:])]
        else:
            # Extract the validation
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/valid"][:])]

        for demo in demos:
            demo_length = f["data"][demo]["dones"].shape[0]
            data = dict(
                action=f["data"][demo]["actions"][:].astype(np.float32),
                observation=dict(
                    agent_image=f["data"][demo]["obs"]["agentview_image"][:],
                    wrist_image=f["data"][demo]["obs"]["robot0_eye_in_hand_image"][:],
                    state=dict(
                        ee_pos=f["data"][demo]["obs"]["robot0_eef_pos"][:].astype(np.float32),
                        ee_quat=f["data"][demo]["obs"]["robot0_eef_quat"][:].astype(np.float32),
                        gripper_qpos=f["data"][demo]["obs"]["robot0_gripper_qpos"][:].astype(np.float32),
                        joint_pos=f["data"][demo]["obs"]["robot0_joint_pos"][:].astype(np.float32),
                        joint_vel=f["data"][demo]["obs"]["robot0_joint_vel"][:].astype(np.float32),
                        object=f["data"][demo]["obs"]["object"][:].astype(np.float32),
                    ),
                ),
                is_first=np.zeros(demo_length, dtype=np.bool_),
                is_last=np.zeros(demo_length, dtype=np.bool_),
                is_terminal=np.zeros(demo_length, dtype=np.bool_),
                discount=np.ones(demo_length, dtype=np.float32),
                reward=f["data"][demo]["rewards"][:],
            )
            data["is_first"][0] = True

            episode = []
            for i in range(demo_length):
                step = tf.nest.map_structure(lambda x, i=i: x[i], data)
                step["language_instruction"] = language_instruction
                episode.append(step)

            # Finally add the terminal states.
            terminal_step = dict(
                action=np.zeros(7, dtype=np.float32),
                observation=dict(
                    agent_image=f["data"][demo]["next_obs"]["agentview_image"][demo_length - 1],
                    wrist_image=f["data"][demo]["next_obs"]["robot0_eye_in_hand_image"][demo_length - 1],
                    state=dict(
                        ee_pos=f["data"][demo]["next_obs"]["robot0_eef_pos"][demo_length - 1].astype(np.float32),
                        ee_quat=f["data"][demo]["next_obs"]["robot0_eef_quat"][demo_length - 1].astype(np.float32),
                        gripper_qpos=f["data"][demo]["next_obs"]["robot0_gripper_qpos"][demo_length - 1].astype(
                            np.float32
                        ),
                        joint_pos=f["data"][demo]["next_obs"]["robot0_joint_pos"][demo_length - 1].astype(np.float32),
                        joint_vel=f["data"][demo]["next_obs"]["robot0_joint_vel"][demo_length - 1].astype(np.float32),
                        object=f["data"][demo]["next_obs"]["object"][demo_length - 1].astype(np.float32),
                    ),
                ),
                is_first=False,
                is_last=True,
                is_terminal=True,
                discount=1.0,
                reward=0,
                language_instruction=language_instruction,
            )
            episode.append(terminal_step)
            metadata = dict(ep_idx=int(demo.split("_")[-1]), file_path=os.path.join(path, demo))

            # Try to add a quality score.
            if all(k in f["mask"] for k in ("better", "okay", "worse")):
                demo_key = demo.encode("utf-8")
                if demo_key in f["mask/better"]:
                    metadata["quality_score"] = 3
                elif demo_key in f["mask/okay"]:
                    metadata["quality_score"] = 2
                elif demo_key in f["mask/worse"]:
                    metadata["quality_score"] = 1
                else:
                    raise ValueError("Episode was not in an quality mask.")

            # Try to add the operator.
            operator_keys = [
                k for k in f["mask"] if "operator" in k and not k.endswith("train") and not k.endswith("valid")
            ]
            if len(operator_keys) > 0:
                demo_key = demo.encode("utf-8")
                for k in operator_keys:
                    if demo_key in f["mask"][k]:
                        metadata["operator"] = k
                assert "operator" in metadata, "Operator was not found."

            yield demo, dict(steps=episode, episode_metadata=metadata)

        # Finally close the file.
        f.close()
