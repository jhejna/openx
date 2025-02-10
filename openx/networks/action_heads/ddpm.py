from typing import Optional

import jax
from flax import linen as nn
from jax import numpy as jnp

from . import core


def _squaredcos_cap_v2(timesteps, s=0.008):
    t = jnp.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class DDPMActionHead(core.ActionHead):
    """
    Diffusion action head. Based on the DDPM implementation from Octo and Bridge.
    """

    timesteps: int = 100
    clip_sample: Optional[float] = None
    variance_type: str = "fixed_large"
    num_noise_samples: int = 1
    projection_predicts_sequence: bool = False

    def setup(self):
        assert self.action_horizon is not None, "Must have action horizon set for DDPM Action Head."
        assert self.model is not None, "Must have a model for DDPM Action Head."
        betas = _squaredcos_cap_v2(self.timesteps).astype(jnp.float32)
        self.alphas = 1.0 - betas  # So betas = 1 - alphas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.action_proj = nn.Dense(
            self.action_dim * self.action_horizon if self.projection_predicts_sequence else self.action_dim
        )

    def __call__(self, obs: jax.Array, action: jax.Array, time: jax.Array, train: bool = True):
        pred = self.action_proj(self.model(obs, action=action, time=time, train=train))
        return jnp.reshape(pred, action.shape)

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        # handle rng creation
        time_key, noise_key = jax.random.split(self.make_rng("dropout"))
        b = action.shape[0]
        time = jax.random.randint(
            time_key, shape=(self.num_noise_samples, b, 1), minval=0, maxval=self.timesteps
        )  # (N, B, 1, 1)
        noise = jax.random.normal(noise_key, (self.num_noise_samples, *action.shape))  # (B, T, D)
        # Add noise to the action according to the schedule
        sqrt_alpha_prod = jnp.sqrt(self.alphas_cumprod[time[..., None]])  # (N, B, 1, 1)
        sqrt_one_minus_alpha_prod = jnp.sqrt(1 - self.alphas_cumprod[time[..., None]])  # (N, B, 1, 1)
        if self.clip_sample is not None:
            # If we are clipping at inference time, better assume the same range for train time!
            action = jnp.clip(action, -self.clip_sample, self.clip_sample)
        noisy_action = sqrt_alpha_prod * action[None] + sqrt_one_minus_alpha_prod * noise

        # Tile the obs for num_noise_samples
        obs = jnp.tile(obs[None], (self.num_noise_samples,) + len(obs.shape) * (1,))

        # Reshape to a single batch dimension so model logic can remain the same.
        # For some reason changing this seems to affect things, even though I think it shouldnt
        # my hypothesis is that __call__ cannot be used with differing dimensions or it maybe does something? idk.
        obs = jnp.reshape(obs, (self.num_noise_samples * b, *obs.shape[2:]))
        noisy_action = jnp.reshape(noisy_action, (self.num_noise_samples * b, self.action_horizon, self.action_dim))
        time = jnp.reshape(time, (self.num_noise_samples * b, 1))

        # Run the network
        pred = self(obs=obs, action=noisy_action, time=time, train=train)  # (N, B, T, D)
        pred = jnp.reshape(pred, (self.num_noise_samples, b, self.action_horizon, self.action_dim))
        return jnp.square(pred - noise).sum(axis=-1).mean(axis=0)  # (N, B, T, D) --> (B, T)

    def predict(self, obs: jax.Array, train: bool = True):
        """
        Code inspired by diffusers:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_flax.py
        """
        module, variables = self.unbind()

        def loop_body(i, args):
            sample, rng = args
            time = self.timesteps - 1 - i
            # Note that here time is (B, 1, 1) where as in loss in is (B, 1)
            time = jnp.broadcast_to(time, (sample.shape[0], 1, 1))
            alpha = self.alphas[time]
            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = jnp.where(time > 0, self.alphas_cumprod[time - 1], jnp.array(1.0, dtype=jnp.float32))

            # Run the model. Reduce time to (B, 1) for the model.
            eps = module.apply(variables, obs, action=sample, time=time[:, 0], train=train)

            # Predict x_0, clip if desired.
            orig = (sample - jnp.sqrt(1 - alpha_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
            if self.clip_sample is not None:
                orig = jnp.clip(orig, -self.clip_sample, self.clip_sample)

            # Compute x_{t-1} using x_0
            orig_coeff = jnp.sqrt(alpha_prod_t_prev) * (1 - alpha) / (1 - alpha_prod_t)
            current_coeff = jnp.sqrt(alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

            prev = orig_coeff * orig + current_coeff * sample

            # Add noise according to the schedule
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha)
            if self.variance_type == "fixed_large":
                variance = 1 - alpha
            elif self.variance_type == "fixed_small":
                variance = jnp.clip(variance, a_min=1e-20)
            else:
                raise ValueError("Invalid schedule provided")

            rng, key = jax.random.split(rng)
            variance = jnp.where(time > 0, variance, jnp.zeros(eps.shape, dtype=jnp.float32))
            prev = prev + jnp.sqrt(variance) * jax.random.normal(key, shape=sample.shape, dtype=jnp.float32)
            return (prev, rng)

        rng, key = jax.random.split(self.make_rng("dropout"))
        noisy_action = jax.random.normal(key, (obs.shape[0], self.action_horizon, self.action_dim), dtype=jnp.float32)
        noisy_action, _ = jax.lax.fori_loop(0, self.timesteps, loop_body, (noisy_action, rng))

        return noisy_action
