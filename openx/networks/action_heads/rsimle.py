import jax
from flax import linen as nn
from jax import numpy as jnp

from . import core


class RSIMLEActionHead(core.ActionHead):
    """
    Diffusion action head. Based on the DDPM implementation from Octo and Bridge.
    """

    num_noise_samples: int = 16
    z_dim: int = 8
    z_per_timestep: bool = True
    projection_predicts_sequence: bool = False
    metric: str = "l2"
    epsilon: float | tuple[float, float] = 0.03
    num_backprop: int = 1

    def setup(self):
        self.action_proj = nn.Dense(
            self.action_dim * self.action_horizon if self.projection_predicts_sequence else self.action_dim
        )

    def __call__(self, obs: jax.Array, z: jax.Array, train: bool = True):
        pred = self.action_proj(self.model(obs, z=z, train=train))
        return jnp.reshape(pred, (obs.shape[0], self.action_horizon, self.action_dim))

    def loss(self, obs: jax.Array, action: jax.Array, mask: jax.Array, train: bool = True):
        # handle rng creation, piggy back off of the dropout one.
        noise_key, eps_key = jax.random.split(self.make_rng("dropout"))
        b = action.shape[0]
        z_shape = (
            (b * self.num_noise_samples, self.action_horizon, self.z_dim)
            if self.z_per_timestep
            else (b * self.num_noise_samples, self.z_dim)
        )
        z = jax.random.normal(noise_key, z_shape, dtype=jnp.float32)
        obs = jnp.tile(obs[:, None], (1, self.num_noise_samples, *(1 for _ in range(len(obs.shape) - 1))))
        obs = jnp.reshape(obs, (b * self.num_noise_samples, *obs.shape[2:]))

        # Run the network
        pred = self(obs=obs, z=z, train=train)
        pred = jnp.reshape(pred, (b, self.num_noise_samples, self.action_horizon, self.action_dim))

        # Distance Computation, applying the mask to the difference before computation.
        diff = pred - action[:, None]  # (B, N, T, D)
        diff = diff * mask[:, None, :, None]
        if self.metric == "mse":
            dist = jnp.sum(jnp.square(diff), axis=-1)  # (B, N, T)
            dist = jnp.sum(dist, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1)[:, None], 1.0)
        elif self.metric == "l2":
            dist = jnp.sum(jnp.square(diff), axis=-1)  # (B, N, T)
            dist = jnp.sqrt(jnp.where(dist == 0, 1e-14, dist))  # No Sqrt on Zero to avoid NaN gradients
        else:
            raise ValueError("Invalid metric specified")
        dist = jnp.sum(dist, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1)[:, None], 1.0)

        # Rejection Sampling
        if isinstance(self.epsilon, tuple):
            low, high = self.epsilon
            epsilon = jax.random.uniform(eps_key, shape=(b, 1), dtype=jnp.float32, minval=low, maxval=high)
        else:
            epsilon = self.epsilon
        dist = jnp.where(dist < epsilon, jnp.inf, dist)  # (B, N)

        # Loss Computation
        if self.num_backprop == 1:
            # Avoid the sort by using argmin.
            loss = dist[jnp.arange(b), jnp.argmin(dist, axis=-1)]  # (B,)
        else:
            loss = jnp.sort(dist, axis=-1)[:, : self.num_backprop]  # (B, N)

        # Mask to avoid inf from computations.
        inf_mask = ~jnp.isinf(loss)
        return jnp.sum(loss * inf_mask) / jnp.maximum(jnp.sum(inf_mask), 1.0)

    def predict(self, obs: jax.Array, train: bool = True):
        noise_key = self.make_rng("dropout")
        b = obs.shape[0]
        z_shape = (b, self.action_horizon, self.z_dim) if self.z_per_timestep else (b, self.z_dim)
        z = jax.random.normal(noise_key, z_shape, dtype=jnp.float32)
        return self(obs=obs, z=z, train=train)
