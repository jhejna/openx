# OpenX

This is a codebase primarily developed by [Joey Hejna](https://jhejna.github.io) for training robot models using Jax, Flax, and the OpenX Embodiment datasets. We build heavily upon ideas used in the [Octo repository](https://github.com/octo-models/octo).

**Principles**: this codebase is desined to be fucntional in nature. Feel free to define types and dataclasses and use objects from other libraries, but our implementations should be functions. This makes it easier to scale code across multiple platforms and for distributed training.

## Installation
First, create a conda environment with python 3.11, and then install requirements and this repo.
```
conda create -n openx python=3.11
pip install -r requirements.txt
pip install -e .
```
If you are on GPU, you will additionally need to install the corresponding jaxlib verison.
```
pip install --upgrade "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If you are on TPU, instead run:
```
pip install --upgrade "jax[tpu]==0.4.26" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Robomimic**
We benchmarked some of our implementations against Pytorch versions in robomimic. Installing the correct robomimic version corresponding to that used in the [original Robomimic paper](https://arxiv.org/abs/2108.03298) is pain. We provide more details commented out in the requirements.txt file, but the basics are as follows.

First, follow the instructions to install `mujoco210_linux` found [here](https://github.com/openai/mujoco-py)

Then, install robosuite, robomimic, and needed dependencies.
```
# Robosuite
git clone https://github.com/ARISE-Initiative/robosuite/
cd robosuite
git checkout offline_study
pip install -e . --no-deps # Ignore
cd ..
# Robomimic
git clone https://github.com/ARISE-Initiative/robomimic/
cd robosuite
git checkout v0.2.0
pip install -e . --no-deps # Ignore
cd ..
# Dependencies
pip install "mujoco-py<2.2,>=2.0"
pip install cython==0.29.37
```

Then repeatedly try to import mujoco_py, robosuite, and robomimic until it works. There are a few manual changes to the code in robosuite and robomimic you will need to make:
1. Comment out all references to EGL Probe if you are using TPU.
2. You will need to change some imports to `from collections.abc` from `from collections`. This is because some typing hints used in robosuite and robomimic were deprecated in Python 3.11.

## Usage

You can train a Behavior Cloning model with
```
python scripts/train_bc.py --config path/to/config --path save/path --name name/on/wandb --project project/on/wandb
```

Example config files can be found in `configs`.

## Datasets

Dataloading is designed to happen in a functional pipeline. Implementations in `openx/datasets/core.py` include core functionality. `openx/datasets/dataloader.py` combines the functions in core in a user-approachable and configurable way.

There are

1. `load_dataset`. This is when you load and RLDS dataset, and must be used everywhere. After this step is when you can apply dataset specific transformations.
2. `compute_dataset_statistics` computes and caches dataset statistics globally from a path. This ignores splits.
3. `standardize_dataset`. This standardizes all datasets to the same format according to a given structure and applies standard episode level transforms. Finally removes the last timestep.
4. `flatten_dataset`. This flattens the dataset into a dataset of steps from a dataset of trajectories.

The dataloader class does this for all datasets in a standard fashion and then shuffles, decodes images, and applies augmentations.


## TODOs:

The following features are planned:

* Incorporate language (choose where the instruction / encoding belongs)
* allow for changing the action keys for different datasets. ie on bridge we want to train on `achieved_delta` but on other datasets we want `desired_delta`.
* figure out if we can make the OXE shuffle buffer bigger.
* Add more architectures and masking schemes
