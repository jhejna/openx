# Defines a config for training DiffusionPolicy on a single task from the libero benchmark.
import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.libero import libero_dataset_transform
from openx.data.filters import task_filter
from openx.data.utils import NormalizationType, StateEncoding
from openx.envs.libero_env import LiberoEnv
from openx.networks.action_heads.ddpm import DDPMActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.components.unet import ConditionalUnet1D
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "libero_10,KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,1"):
    benchmark, task, seed = config_str.split(",")
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_EULER: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
            "image": {
                "agent": (84, 84),
                "wrist": (84, 84),
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
    }

    dataloader = dict(
        datasets={
            task: dict(
                path="path/to/datasets/libero_rlds/{benchmark}/1.0.0".format(benchmark=benchmark),
                train_split="train",
                transform=ModuleSpec.create(libero_dataset_transform),
                filter=ModuleSpec.create(task_filter, task),
            )
        },
        n_obs=2,
        n_action=16,
        augment_kwargs=dict(scale_range=(0.85, 1.0), aspect_ratio_range=None),
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        cache=False,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [128], activate_final=False), flatten_time=True
            ),
        ),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                ConditionalUnet1D, down_features=(256, 512, 1024), mid_layers=2, time_features=128, kernel_size=5
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
            action_dim=7,
            action_horizon=16,
            num_noise_samples=1,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=1000,
        decay_steps=500000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)
    envs = {task: ModuleSpec.create(LiberoEnv, benchmark=benchmark, task=task, horizon=12)}
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=500000,
            log_freq=500,
            val_freq=2500,
            eval_freq=20000,
            save_freq=100000,
            val_steps=25,
            n_eval_proc=24,
            eval_ep=24,
            exec_horizon=8,
            seed=seed,
        )
    )
