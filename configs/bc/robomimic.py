# Define the config for robomimic
import optax
from ml_collections import ConfigDict

from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.envs.robomimic import RobomimicEnv
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import Concatenate
from openx.networks.resnet import ResNet18
from openx.networks.unet import ConditionalUnet1D
from openx.utils.spec import ModuleSpec


def get_config():
    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_QUAT: NormalizationType.NONE,
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
        datasets=dict(
            square_ph=dict(
                path="path/to/robomimic/dataset",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(robomimic_dataset_transform),
            ),
        ),
        n_obs=2,
        n_action=16,
        augment_kwargs=dict(scale_range=(0.85, 1.0), aspect_ratio_range=None),
        chunk_img=True,
        goal_conditioned=False,
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=True,
    )

    model = ModuleSpec.create(
        Model,
        encoders={
            "observation->image->agent": ModuleSpec.create(ResNet18),
            "observation->image->wrist": ModuleSpec.create(ResNet18),
            "observation->state": None,
        },
        trunk=ModuleSpec.create(Concatenate, features=128, flatten_time=True),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                ConditionalUnet1D, down_features=(256, 512, 1024), mid_layers=2, time_features=128, kernel_size=5
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
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

    envs = dict(square_ph=ModuleSpec.create(RobomimicEnv, path="path/to/robomimic/dataset", horizon=400))
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            model=model,
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
            seed=0,
        )
    )
