# Define the config for robomimic
import os

import optax
from ml_collections import ConfigDict

from openx.data.mixes import OXE_ALL, OXE_MAGIC_SOUP, RTX_MIX
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads import DDPMActionHead
from openx.networks.core import Model
from openx.networks.mlp import MLPResNet
from openx.networks.vit import SmallStem, ViT_S
from openx.utils.schedules import warmup_rsqrt_schedule
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "magic_soup,size"):
    data_mix, data_weight = config_str.split(",")
    assert data_mix in {"all", "magic_soup", "rtx"}
    assert data_weight in {"size", "uniform"}

    structure = {
        "observation": {
            "image": {
                "agent": (224, 224),  # Height x width
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

    # Get the dataset mix
    datasets = {"all": OXE_ALL, "magic_soup": OXE_MAGIC_SOUP, "rtx": RTX_MIX}[data_mix]

    if data_weight == "uniform":
        # Weight uniformly
        for dataset in datasets:
            datasets[dataset]["weight"] = 1.0
    else:
        assert all([dataset_config["weight"] != 1.0 for dataset_config in datasets.values()])

    total_weight = sum([dataset_config["weight"] for dataset_config in datasets.values()])

    # Add the path to all the datasets
    # Allocate the parallel threads
    for dataset in datasets:
        datasets[dataset]["path"] = os.path.join("path/to/oxe/datasets", datasets[dataset]["path"])
        datasets[dataset]["num_parallel_reads"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))
        datasets[dataset]["num_parallel_calls"] = max(1, int(32 * datasets[dataset]["weight"] / total_weight))

    dataloader = dict(
        datasets=datasets,
        n_obs=2,
        n_action=2,
        augment_kwargs=dict(
            scale_range=(0.85, 1.0),
            aspect_ratio_range=(0.85, 1.15),
            aligned=True,
            brightness=0.1,
            contrast_range=[0.9, 1.1],
            saturation_range=[0.9, 1.1],
            hue=0.03,
        ),
        chunk_img=True,
        goal_conditioned=True,
        shuffle_size=500000,
        batch_size=512,
        recompute_statistics=False,
        weight_by_size=False,
        num_parallel_calls=128,
        num_batch_parallel_calls=None,
        restrict_memory=True,
    )

    model = ModuleSpec.create(
        Model,
        encoders={
            "observation->image->agent,goal->image->agent": ModuleSpec.create(SmallStem, embed_dim=384, patch_size=16),
        },
        trunk=ModuleSpec.create(ViT_S),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                MLPResNet, hidden_dim=256, num_blocks=3, time_features=64, dropout_rate=None, use_layer_norm=True
            ),
            clip_sample=1.0,
            timesteps=50,
            variance_type="fixed_small",
        ),
    )

    lr_schedule = ModuleSpec.create(
        warmup_rsqrt_schedule,
        init_value=0,
        peak_value=3e-4,
        warmup_steps=2000,
        timescale=10000,
    )
    optimizer = ModuleSpec.create(optax.adamw, mu_dtype="bfloat16", weight_decay=0.05)

    envs = None
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            clip_gradient=1.0,
            # Add training parameters
            steps=400000,
            log_freq=500,
            val_freq=10000,
            eval_freq=20000,
            save_freq=50000,
            val_steps=20,
            seed=0,
        )
    )
