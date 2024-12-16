from typing import Optional

import distrax
import jax
from flax import linen as nn
from jax import numpy as jnp

from . import core


class DiscreteActionHead(core.ActionHead):
    n_action_bins: int
    bin_type: str = "uniform"
    temperature: Optional[bool] = None

    def setup(self):
        assert self.n_action_bins <= 256, "Maximum action bins supported is 256 due to uint8."
        if self.bin_type == "uniform":
            self.bins = jnp.linspace(-1, 1, self.n_action_bins + 1)
        elif self.bin_type == "gaussian":
            # Values chosen to approximate -5 to 5
            self.bins = jax.scipy.stats.norm.ppf(jnp.linspace(5e-3, 1 - 5e-3, self.n_action_bins + 1), scale=2)
        else:
            raise ValueError("Invalid bin type provided")
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train) if self.model is not None else obs
        if self.action_horizon is None:
            pred_dim = self.action_dim * self.n_action_bins
            shape = x.shape[:-1] + (self.action_dim, self.action_bins)
        else:
            pred_dim = self.n_action_bins * self.action_dim * self.n_action_bins
            shape = (x.shape[0], self.action_horizon, self.action_dim, self.n_action_bins)

        pred_dim = self.action_dim if len(x.shape) == 3 else self.action_dim * self.action_horizon
        x = nn.Dense(pred_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.reshape(x, shape)

    def predict(self, obs: jax.Array, train: bool = True):
        logits = self(obs, train=train)
        # by default we are not going to sample
        if self.temperature is None:
            action = jnp.argmax(logits, axis=-1)
        else:
            rng, key = jax.random.split(self.make_rng("dropout"))
            dist = distrax.Categorical(logits=logits / self.temperature)
            action = dist.sample(seed=key).astype(jnp.int32)
        return self.bin_centers[action]

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        logits = self(obs, train=train)  # (B, T, D, N)

        # Clip the actions to be in range
        action = jnp.clip(action, -1, 1) if self.bin_type == "uniform" else jnp.clip(action, -5, 5)

        # Compute the binned actions
        action = action[..., None]  # (B, T, D, 1)
        action_one_hot = (action < self.bins[1:]) & (action >= self.bins[:-1])
        action_one_hot = action_one_hot.astype(logits.dtype)

        logprobs = jax.nn.log_softmax(logits, axis=-1)  # (B, T, D, N)
        return -jnp.sum(logprobs * action_one_hot, axis=(-1, -2))  # Sum over dist and action dims
