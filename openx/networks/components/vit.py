from typing import Any, Callable, Optional, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp

from .transformer import MAPHead, PositionalEmbedding, TransformerEncoder

T = TypeVar("T")


class PatchEncoder(nn.Module):
    embed_dim: int
    patch_size: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array], train: bool = True):
        # Determine whether or not we concatenate
        x = obs if goal is None else jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)

        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        return nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
            dtype=self.dtype,
        )(x)


def weight_standardize(w, axis, eps: float = 1e-5):
    """Subtracts mean and divides by standard deviation."""
    w = w - jnp.mean(w, axis=axis)
    return w / (jnp.std(w, axis=axis) + eps)


class StdConv(nn.Conv):
    """Convolution with weight standardization."""

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == "kernel":
            param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
        return param


class SmallStem(nn.Module):
    """
    Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    embed_dim: int
    patch_size: int = 16
    kernel_sizes: tuple = (3, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2)
    features: tuple = (32, 96, 192, 384)  # modified from 48 -> 32 first layer for GroupNorm
    padding: tuple = (1, 1, 1, 1)
    use_std_conv: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array], train: bool = True):
        # Determine whether or not we concatenate
        x = obs if goal is None else jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)

        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        conv_cls = StdConv if self.use_std_conv else nn.Conv
        for kernel_size, stride, features, padding in zip(
            self.kernel_sizes,
            self.strides,
            self.features,
            self.padding,
            strict=False,
        ):
            x = conv_cls(
                features=features,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
            )(x)
            x = nn.GroupNorm(epsilon=1e-5, dtype=self.dtype)(x)
            x = nn.relu(x)

        return nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size // 16, self.patch_size // 16),
            strides=(self.patch_size // 16, self.patch_size // 16),
            padding="VALID",
            name="embedding",
        )(x)


class ViT(nn.Module):
    num_layers: int
    num_heads: int
    num_registers: Optional[int] = None
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    pool_type: Optional[str] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        assert self.pool_type is None or self.pool_type in ("cls", "avg", "map"), "Invalid pool_type."
        # Add positional embedding
        x = PositionalEmbedding(dtype=self.dtype)(x)
        # Add registers
        b, t, d = x.shape
        if self.num_registers is not None:
            registers = self.param(
                "registers",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_registers, d),
                self.dtype,
            )
            x = jnp.concatenate((x, jnp.tile(registers, [b, 1, 1])), axis=1)  # Registers at the end

        # Input dropout on everything, but never dropout the CLS token.
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        if self.pool_type == "cls":
            cls_token = self.param("cls", nn.initializers.zeros, (1, 1, d), x.dtype)
            x = jnp.concatenate((jnp.tile(cls_token, [b, 1, 1]), x), axis=1)  # CLS at the beginning

        # Run the transformer
        x = TransformerEncoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
        )(x, train=train)

        # Collect outputs via specified pooling
        if self.pool_type == "cls":
            return x[:, 0]  # (B, D)
        if self.pool_type == "avg":
            return jnp.mean(x[:, :t], axis=1)  # Ignore registers
        if self.pool_type == "map":
            return MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x[:, :t])  # Ignore registers
        if self.pool_type is not None:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")
        return x


class ViTT(ViT):
    embed_dim: int = 192
    num_layers: int = 12
    mlp_dim: int = 768
    num_heads: int = 3
    num_registers: int = 4
    dropout_rate: float = 0.0


class ViTS(ViT):
    embed_dim: int = 384
    num_layers: int = 12
    mlp_dim: int = 1536
    num_heads: int = 6
    num_registers: int = 4
    dropout_rate: float = 0.0


class ViTB(ViT):
    embed_dim: int = 768
    num_layers: int = 12
    mlp_dim: int = 3072
    num_heads: int = 12
    num_registers: int = 8
    dropout_rate: float = 0.0
