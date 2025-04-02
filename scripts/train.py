import functools
import json
import os
from collections import defaultdict

import flax
import gymnasium as gym
import jax
import numpy as np
import optax
import tensorflow as tf
import tqdm
from absl import app, flags
from jax.experimental import compilation_cache, multihost_utils
from ml_collections import config_flags
from orbax import checkpoint as ocp

import wandb
from openx.data.dataloader import make_dataloader
from openx.envs.wrappers import wrap_env
from openx.utils.evaluate import eval_policy
from openx.utils.logger import DummyLogger, Logger, Timer
from openx.utils.spec import ModuleSpec, recursively_instantiate

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "/tmp/test/", "Path to save logs and checkpoints.")
flags.DEFINE_string("name", "train", "Name of the experiment")
flags.DEFINE_string("project", "openx", "WandB project to save logs to.")
flags.DEFINE_bool("debug", False, "Whether or not to enable debug mode.")
# Always lock the config to avoid subtle bugs
config_flags.DEFINE_config_file(
    "config", None, "File path to the training hyperparameter configuration.", lock_config=True
)


def main(_):
    # Initialize experimental jax compilation cache
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))
    else:
        compilation_cache.compilation_cache.set_cache_dir(os.environ["JAX_COMPILATION_CACHE_DIR"])

    assert FLAGS.config.dataloader.batch_size % jax.device_count() == 0

    # Define Shardings
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_spec = jax.sharding.PartitionSpec()
    rep_sharding = jax.sharding.NamedSharding(mesh, rep_spec)

    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Determine the save path
    save_path = FLAGS.path if FLAGS.path.startswith("gs://") else os.path.abspath(FLAGS.path)
    save_path = tf.io.gfile.join(save_path, FLAGS.name)

    ### Init Checkpointing ###
    if not FLAGS.debug:
        state_checkpointer = ocp.CheckpointManager(
            tf.io.gfile.join(save_path, "state"),
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
        )
        weights_checkpointer = ocp.CheckpointManager(save_path)
        start_step = 0 if state_checkpointer.latest_step() is None else state_checkpointer.latest_step()
    else:
        start_step = 0

    ### Set seeds, different per process ###
    tf.random.set_seed(start_step + FLAGS.config.seed + jax.process_index())
    np.random.seed(start_step + FLAGS.config.seed + jax.process_index())
    rng = jax.random.key(start_step + FLAGS.config.seed)
    rng = jax.random.fold_in(rng, jax.process_index())

    ### Create the dataloader, limiting size for debug ###
    dataloader_config = FLAGS.config.dataloader.to_dict()
    if FLAGS.debug:
        # Limit the size of datasets for faster debugging with significantly less fileIO.
        for dataset in dataloader_config["datasets"].items():
            if "train_split" in dataset:
                dataloader_config["datasets"]["train_split"] = "all[:1]"
            if "val_split" in dataset:
                dataloader_config["datasets"]["val_split"] = "all[:1]"
        if dataloader_config.get("shuffle_size", 0) > 0:
            dataloader_config["shuffle_size"] = 10

    train_dataset, val_datasets, dataset_statistics, dataset_ids = make_dataloader(
        **dataloader_config, structure=FLAGS.config.structure.to_dict(), split_for_jax=True
    )

    # Create the data iterators, avoiding copy
    def shard(batch):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        return multihost_utils.host_local_array_to_global_array(batch, mesh, dp_spec)

    train_iterator = map(shard, train_dataset)
    val_iterators = {name: map(shard, ds) for name, ds in val_datasets.items()}

    # Deque the first batch to use as an example for instantiating the model
    example_batch = jax.tree.map(lambda x: x[:1], multihost_utils.process_allgather(next(train_iterator)))

    ### Construct the state ###
    # Instantiate the model
    alg = recursively_instantiate(FLAGS.config.alg.to_dict())

    # Instantiate the optimizer
    tx = ModuleSpec.instantiate(FLAGS.config.optimizer)
    lr_schedule = ModuleSpec.instantiate(FLAGS.config.lr_schedule)()
    kwargs = dict(learning_rate=lr_schedule)

    def _get_decay_mask(params):
        return jax.tree_util.tree_map_with_path(lambda path, _: "kernel" in jax.tree_util.keystr(path), params)

    if any(tx.func is func for func in (optax.adadelta, optax.adafactor, optax.lars)):
        kwargs["weight_decay_mask"] = _get_decay_mask
    elif any(tx.func is func for func in (optax.adamw, optax.adan, optax.adamaxw, optax.lion, optax.nadamw)):
        kwargs["mask"] = _get_decay_mask

    tx = tx(**kwargs)  # Finally create the optimizer
    if "clip_gradient" in FLAGS.config and FLAGS.config.clip_gradient is not None:
        tx = optax.chain(optax.clip_by_global_norm(FLAGS.config.clip_gradient), tx)

    rng, init_rng = jax.random.split(rng)
    state = alg.init(example_batch, tx, init_rng)

    # Restore if needed
    if start_step != 0:
        # restore if we need to.
        state = jax.tree.map(lambda x: jax.device_put(x, rep_sharding), state)
        state = state_checkpointer.restore(start_step, args=ocp.args.StandardRestore(state))

    # Create the train and val steps.
    jitted_train_step = jax.jit(
        alg.train_step,
        # Data parallel sharding, the RNG Key is not sharded and is kept separate across procs.
        in_shardings=(rep_sharding, dp_sharding, None),
        out_shardings=(rep_sharding, rep_sharding),
        # Allow jax to overwrite the buffer in place.
        donate_argnums=0,
    )

    jitted_val_step = jax.jit(alg.val_step, in_shardings=(rep_sharding, dp_sharding, None), out_shardings=rep_sharding)

    ### Setup Eval Envs ###
    envs = dict()
    if FLAGS.config.get("envs", None) is not None and len(FLAGS.config.envs) > 0:
        # Must dereference for pickle-ability
        structure = FLAGS.config.structure.to_dict()
        n_obs, n_action = FLAGS.config.dataloader.n_obs, FLAGS.config.dataloader.n_action
        augment_kwargs = FLAGS.config.dataloader.to_dict().get("augment_kwargs", dict())
        exec_horizon = FLAGS.config.exec_horizon

        def _make_env(fn, stats):
            env = fn()
            return wrap_env(
                env,
                structure=structure,
                dataset_statistics=stats,
                n_obs=n_obs,
                n_action=n_action,
                exec_horizon=exec_horizon,
                augment_kwargs=augment_kwargs,
            )

        for env_name, env_spec in FLAGS.config.envs.to_dict().items():
            env_fn = functools.partial(
                _make_env, fn=ModuleSpec.instantiate(env_spec), stats=dataset_statistics[env_name]
            )
            if FLAGS.config.n_eval_proc > 1:
                envs[env_name] = gym.vector.AsyncVectorEnv(
                    [env_fn for _ in range(FLAGS.config.n_eval_proc)], context="spawn", shared_memory=True
                )
            else:
                envs[env_name] = gym.vector.SyncVectorEnv([env_fn])

        # No sharding for jitted predict, instead we will broadcast results across processes.
        jitted_predict = jax.jit(alg.predict)

    ### Worker Saves Statistics, Configs, ExBatch ###
    if jax.process_index() == 0 and not FLAGS.debug:
        # Save the example batch
        example_batch_path = tf.io.gfile.join(save_path, "example_batch.msgpack")
        with tf.io.gfile.GFile(example_batch_path, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(example_batch))

        # Save the dataset statistics
        dataset_statistics_path = tf.io.gfile.join(save_path, "dataset_statistics.json")
        with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
            json.dump(
                jax.tree.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dataset_statistics), f, indent=4
            )

        # Save the config
        config_path = tf.io.gfile.join(save_path, "config.json")
        with tf.io.gfile.GFile(config_path, "w") as f:
            json.dump(FLAGS.config.to_dict(), f, indent=4)

        # Setup logging
        if os.environ.get("WANDB_API_KEY") is not None:
            # See if a wandb log file exists
            wandb_path = tf.io.gfile.join(save_path, "wandb_id.txt")
            if start_step != 0:
                with tf.io.gfile.GFile(wandb_path, "r") as f:
                    wandb_run = wandb.Api().run("{project}/{id}".format(project=FLAGS.project, id=f.read()))
                wandb.init(project=wandb_run.project, id=wandb_run.id, entity=wandb_run.entity, resume="must")
            else:
                wandb.init(config=FLAGS.config.to_dict(), project=FLAGS.project, name=FLAGS.name, mode="online")
                with tf.io.gfile.GFile(wandb_path, "w") as f:
                    f.write(wandb.run.id)
            writers = ("csv", "wandb")
        else:
            # Otherwise log with tensorboard
            writers = ("csv", "tb")
        if "eval_freq" in FLAGS.config:
            writers = (*writers, "eval")  # Add the eval writer.
    else:
        writers = ()

    # Init Logging, make sure this is done after wandb
    logger = Logger(save_path, writers) if jax.process_index() == 0 else DummyLogger()
    timer = Timer()

    # Training constants
    train_metrics = defaultdict(list)
    for i in tqdm.tqdm(
        range(start_step, FLAGS.config.steps), initial=start_step, total=FLAGS.config.steps, dynamic_ncols=True
    ):
        rng = jax.random.fold_in(rng, i)

        with timer("dataset"):
            batch = next(train_iterator)

        with timer("train"):
            state, info = jitted_train_step(state, batch, rng)
            info["lr"] = lr_schedule(state.step)
            for k, v in info.items():
                train_metrics[k].append(v)

        step = i + 1
        if step % FLAGS.config.log_freq == 0:
            # Log training loss and timing
            logger.update(train_metrics, prefix="train")
            logger.update(timer.times, prefix="time")

            logger.dump(step=step, prefix="train")
            train_metrics = defaultdict(list)
            timer.reset()

        if step % FLAGS.config.val_freq == 0:
            # Run evaluation
            val_metrics = defaultdict(list)
            with timer("val"):
                for ds_name, val_iterator in val_iterators.items():
                    prefix = ds_name.replace("/", "-")  # Remove the '/' for logger
                    val_rng = jax.random.fold_in(rng, dataset_ids[ds_name])
                    for j in tqdm.tqdm(range(FLAGS.config.val_steps), total=FLAGS.config.val_steps):
                        val_rng = jax.random.fold_in(val_rng, j)
                        batch = next(val_iterator)
                        info = jitted_val_step(state, batch, val_rng)
                        for k, v in info.items():
                            val_metrics[prefix + "/" + k].append(v)

            logger.update(val_metrics, prefix="val")
            logger.dump(step=step, prefix="val")

        if "eval_freq" in FLAGS.config and step % FLAGS.config.eval_freq == 0:
            for env_idx, (env_name, env) in enumerate(envs.items()):
                eval_rng = jax.random.fold_in(rng, env_idx)
                with timer("eval/" + env_name):
                    eval_metrics, _ = eval_policy(
                        env, functools.partial(jitted_predict, state), eval_rng, num_ep=FLAGS.config.eval_ep
                    )
                    # Join data from each host to one global array so we log all results.
                    eval_metrics = multihost_utils.host_local_array_to_global_array(eval_metrics, mesh, rep_spec)
                    eval_metrics["num_ep"] = next(iter(eval_metrics.values())).shape[0]
                    logger.update(eval_metrics, prefix="eval/" + env_name)
            # Dump the logger with eval metrics
            logger.dump(step=step, prefix="eval")

        if step % FLAGS.config.save_freq == 0 and not FLAGS.debug:
            # save the train state.
            with timer("save"):
                state_checkpointer.save(step, args=ocp.args.StandardSave(state))
                weights_checkpointer.save(step, args=ocp.args.StandardSave(state.params))

    if not FLAGS.debug:
        state_checkpointer.wait_until_finished()
        state_checkpointer.close()
        weights_checkpointer.wait_until_finished()
        weights_checkpointer.close()


if __name__ == "__main__":
    app.run(main)
