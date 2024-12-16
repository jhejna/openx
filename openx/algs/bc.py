import functools
from typing import Any, Dict, Tuple

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from openx.networks.action_heads.core import ActionHead

from .core import Algorithm


class BehaviorCloningModel(nn.Module):
    """
    This is a Dummy Module that allows for factoring of encoding observations
    and predicting actions.
    """

    observation_encoder: nn.Module
    action_head: ActionHead

    def __call__(self, batch, train: bool = True, **kwargs):
        obs = self.observation_encoder(batch, train=train)
        return self.action_head.predict(obs, train=train, **kwargs)

    def loss(self, batch, train: bool = True):
        obs = self.observation_encoder(batch, train=train)
        return self.action_head.loss(obs, batch["action"], train=train)

    def loss_and_prediction(self, batch, train: bool = True, **kwargs):
        obs = self.observation_encoder(batch)
        loss = self.action_head.loss(obs, batch["action"], train=train)
        pred = self.action_head.predict(obs, train=train, **kwargs)
        return loss, pred


class BehaviorCloning(Algorithm):
    def __init__(self, observation_encoder: nn.Module, action_head: nn.Module):
        self.model = BehaviorCloningModel(observation_encoder, action_head)

    def init(self, batch, tx: optax.GradientTransformation, rng: jax.Array) -> train_state.TrainState:
        """
        Returns a state object and initializes the model.
        """
        params = jax.jit(functools.partial(self.model.init, train=False, method="loss"))(rng, batch)
        return train_state.TrainState.create(
            apply_fn=functools.partial(self.model.apply, method="loss"), params=params, tx=tx
        )

    def train_step(
        self, state: train_state.TrainState, batch: Dict, rng: jax.Array
    ) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        """
        The main train step function. This should return a dictionary of metrics.
        """
        params_rng, dropout_rng = jax.random.split(rng)

        def _loss(params):
            loss = state.apply_fn(params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=True)
            loss = loss * batch["mask"]
            return jnp.mean(loss) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)

        loss, grads = jax.value_and_grad(_loss)(state.params)
        new_state = state.apply_gradients(grads=grads)
        info = dict(loss=loss)
        return new_state, info

    def val_step(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Dict[str, jax.Array]:
        """
        Take a validation step and return a dictionary of metrics.
        """
        params_rng, dropout_rng = jax.random.split(rng)
        loss, pred = self.model.apply(
            state.params,
            batch,
            rngs=dict(params=params_rng, dropout=dropout_rng),
            train=False,
            method="loss_and_prediction",
        )
        loss = loss * batch["mask"]
        loss = jnp.mean(loss) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)
        mse = batch["mask"] * jnp.square(pred - batch["action"]).sum(axis=-1)
        mse = jnp.mean(mse) / jnp.clip(jnp.mean(batch["mask"]), a_min=1e-5, a_max=None)
        return dict(loss=loss, mse=mse)

    def predict(self, state: train_state.TrainState, batch: Dict, rng: jax.Array, **kwargs) -> Any:
        params_rng, dropout_rng = jax.random.split(rng)
        return self.model.apply(
            state.params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=False, **kwargs
        )
