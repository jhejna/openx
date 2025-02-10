"""
Transformer implementation.
Borrows from Google Big Vision
https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py

and the Octo repository
"""

import math
from typing import Any, Optional, TypeVar

import flax.linen as nn
import jax.numpy as jnp

T = TypeVar("T")


class PositionalEmbedding(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        t, d = x.shape[-2:]  # Get (T, D)
        emb = self.param("positional_embedding", nn.initializers.normal(stddev=1 / math.sqrt(d)), (1, t, d), self.dtype)
        return x + emb


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Applies Transformer MlpBlock module."""
        inits = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return nn.Dense(d, dtype=self.dtype, **inits)(x)


class EncoderBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Attention Block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(y, y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = x + y

        # MLP Block
        y = nn.LayerNorm()(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(y, train=train)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    num_layers: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i in range(self.num_layers):
            x = EncoderBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoder_block_{i}",
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, train=train)
        return nn.LayerNorm(name="encoder_norm")(x)


class DecoderBlock(nn.Module):
    """Single transformer decoder block (MHSA + CrossAttn + MLP)."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, tgt, memory, train: bool = True):
        # First perform self attention on the tgt
        x = nn.LayerNorm(dtype=self.dtype)(tgt)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        tgt = tgt + x

        # Now perform cross attention to the encoder.
        x = nn.LayerNorm(dtype=self.dtype)(tgt)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(x, memory)  # Here is cross attention input as kv
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        tgt = tgt + x

        # Now finally add the MLP block
        x = nn.LayerNorm(dtype=self.dtype)(tgt)
        x = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(x, train=train)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return tgt + x


class TransformerDecoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation."""

    num_layers: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, tgt, memory, train: bool = True):
        for i in range(self.num_layers):
            tgt = DecoderBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"decoder_block_{i}",
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(tgt, memory, train=train)
        return nn.LayerNorm(name="decoder_norm")(tgt)


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x, train: bool = True):
        # TODO
        b, _, d = x.shape  # pylint: disable=unused-variable
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [b, 1, 1])

        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform())(
            probe, x
        )
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y, train=train)
        return x[:, 0]
