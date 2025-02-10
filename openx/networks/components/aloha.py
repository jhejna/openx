from typing import Any, Optional

from flax import linen as nn
from jax import numpy as jnp

from .mlp import SinusoidalPosEmb
from .transformer import PositionalEmbedding, TransformerDecoder


class ACTDecoder(nn.Module):
    """
    ACT Decoder from the original paper, except we use a standard decoder instead of DETR
    """

    action_horizon: int
    num_layers: int
    num_heads: int
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, train: bool = True):
        # We assume the observation is already a transformer decoder sequence
        assert len(obs.shape) == 3
        embed_dim = obs.shape[-1]

        # Construct the action embedding
        tgt = jnp.zeros((obs.shape[0], self.action_horizon, embed_dim))
        tgt = PositionalEmbedding(self.dtype)(tgt)

        return TransformerDecoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(tgt, obs, train=train)


class ACTDiffusionDecoder(nn.Module):
    """
    ACT Diffusion Decoder from ALOHA Unleashed.
    """

    num_layers: int
    num_heads: int
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, action, time, train: bool = True):
        # We assume the observation is already a transformer decoder sequence
        embed_dim = obs.shape[-1]
        time = SinusoidalPosEmb(embed_dim, scale=200.0)(time)
        time = nn.Dense(embed_dim)(time)[..., None, :]  # project and add token dim.
        memory = jnp.concatenate((obs, time), axis=-2)  # Concate on time dim

        # Construct the action embedding
        tgt = nn.Dense(embed_dim)(action)
        tgt = PositionalEmbedding(self.dtype)(tgt)

        return TransformerDecoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(tgt, memory, train=train)
