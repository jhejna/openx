import jax
from flax import linen as nn
from jax import numpy as jnp

from . import core


class L2ActionHead(core.ActionHead):
    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train) if self.model is not None else obs
        if self.action_horizon is None:
            return nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dense(self.action_dim * self.action_horizon, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.reshape(x, (x.shape[0], self.action_horizon, self.action_dim))

    def predict(self, obs: jax.Array, train: bool = True):
        return self(obs, train=train)

    def loss(self, obs: jax.Array, action: jax.Array, mask: jax.Array, train: bool = True):
        pred = self(obs, train=train)
        loss = jnp.square(pred - action).sum(axis=-1)
        return jnp.mean(loss * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


class L1ActionHead(core.ActionHead):
    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train) if self.model is not None else obs
        if self.action_horizon is None:
            return nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dense(self.action_dim * self.action_horizon, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.reshape(x, (x.shape[0], self.action_horizon, self.action_dim))

    def predict(self, obs: jax.Array, train: bool = True):
        return self(obs, train=train)

    def loss(self, obs: jax.Array, action: jax.Array, mask: jax.Array, train: bool = True):
        pred = self(obs, train=train)
        loss = jnp.abs(pred - action).sum(axis=-1)  # (B, T, D) --> (B, T)
        return jnp.mean(loss * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)
