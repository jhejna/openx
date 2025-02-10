"""
Flax implementation of ResNet V1.5.

Much of this implementation was borrowed from
https://github.com/google/flax/commits/main/examples/imagenet/models.py
under the APACHE 2.0 license. See the Flax repo for details.
"""

import math
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    transpose: bool = False
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides if not self.transpose else (1, 1), padding="SAME")(x)
        y = self.norm()(y)
        y = self.act(y)
        if self.transpose and (self.strides[0] > 1 or self.strides[1] > 1):
            h, w, c = y.shape[-3:]
            y = jax.image.resize(
                x, shape=y.shape[:-3] + (self.strides[0] * h, self.strides[1] * w, c), method="nearest"
            )
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            if self.transpose:
                h, w, c = residual.shape[-3:]
                residual = jax.image.resize(
                    residual,
                    shape=residual.shape[:-3] + (self.strides[0] * h, self.strides[1] * w, c),
                    method="nearest",
                )
                kernel_size, strides = (3, 3), (1, 1)
            else:
                kernel_size, strides = (1, 1), self.strides
            residual = self.conv(
                self.filters,
                kernel_size,
                strides,
                name="conv_proj",
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class SpatialSoftmax(nn.Module):
    num_kp: Optional[int] = 64
    temperature: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Input is shape (Batch dims...., H, W, C)
        if self.num_kp is not None:
            x = nn.Conv(self.num_kp, kernel_size=(1, 1), strides=1, name="keypoints")(x)

        h, w, c = x.shape[-3:]
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, w, dtype=self.dtype), jnp.linspace(-1.0, 1.0, h, dtype=self.dtype)
        )
        pos_x, pos_y = pos_x.reshape((h * w, 1)), pos_y.reshape((h * w, 1))  # (H*W, 1)
        x = x.reshape(x.shape[:-3] + (h * w, c))  # (..., H, W, C)

        attention = jax.nn.softmax(x / self.temperature, axis=-2)  # (B..., H*W, K)
        expected_x = (pos_x * attention).sum(axis=-2, keepdims=True)  # (B..., 1, K)
        expected_y = (pos_y * attention).sum(axis=-2, keepdims=True)
        expected_xy = jnp.concatenate((expected_x, expected_y), axis=-2)  # (B..., 2, K)

        return expected_xy.reshape(x.shape[:-2] + (2 * c,))


class SpatialCoordinates(nn.Module):
    """
    Inspired by https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/vision/resnet_v1.py
    but simplified to be a bit more readable and stay in jnp
    """

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        h, w = x.shape[-3:-1]
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, w, dtype=self.dtype), jnp.linspace(-1.0, 1.0, h, dtype=self.dtype)
        )
        coords = jnp.stack((pos_x, pos_y), axis=-1)  # (H, W, 2)
        coords = jnp.broadcast_to(coords, x.shape[:-3] + coords.shape)
        return jnp.concatenate((x, coords), axis=-1)


class AttentionPool2d(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        h, w, c = x.shape[-3:]
        x = jnp.reshape(x, x.shape[:-3] + (h * w, c))
        # the query token is the avg plus pos emb
        x = jnp.concatenate((jnp.mean(x, axis=-2, keepdims=True), x), axis=-2)
        pos_emb = self.param(
            "positional_embedding", nn.initializers.normal(stddev=1 / math.sqrt(c)), x.shape[-2:], self.dtype
        )
        x = x + pos_emb

        # NOTE: Clip adds a linear projection after this, but we usually add an output laye afterwards.
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=0.0,
            dtype=self.dtype,
        )(x[..., :1, :], x)
        # Remove the channel dim
        return x[..., 0, :]


class ResNet(nn.Module):
    """ResNetV1.5, except with group norm instead of BatchNorm"""

    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = ResNetBlock
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    spatial_coordinates: bool = False
    num_kp: Optional[int] = None
    attention_pool: bool = False
    average_pool: bool = False
    use_clip_stem: bool = False

    @nn.compact
    def __call__(self, obs, goal: Optional[jax.Array] = None, train: bool = True):
        assert (
            sum((self.attention_pool, self.average_pool, self.num_kp is not None)) <= 1
        ), "Multiple types of pooling provided. Can only use one."

        # Initialize layers
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, 32, epsilon=1e-5, dtype=self.dtype)
        act = getattr(jax.nn, self.act)

        # Obs is shape (B, T, H, W, C), Goal is shape (B, 1, H, W, C)
        x = obs if goal is None else jnp.concatenate((obs, jnp.broadcast_to(goal, obs.shape)), axis=-1)
        # Shift inputs to -1 to 1 from 0 to 1
        x = 2 * x - 1

        if self.spatial_coordinates:
            # Add spatial coordinates.
            x = SpatialCoordinates(dtype=self.dtype)(x)

        if self.use_clip_stem:
            # Use the CLIP resnet stem
            # see https://github.com/openai/CLIP/blob/main/clip/model.py
            x = conv(self.num_filters // 2, (3, 3), (2, 2), padding="SAME", name="conv_stem_1")(x)
            x = norm(name="norm_stem_1")(x)
            x = nn.relu(x)
            x = conv(self.num_filters // 2, (3, 3), padding="SAME", name="conv_stem_2")(x)
            x = norm(name="norm_stem_2")(x)
            x = nn.relu(x)
            x = conv(self.num_filters, (3, 3), padding="SAME", name="conv_stem_3")(x)
            x = norm(name="norm_stem_3")(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        else:
            x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(x)
            x = norm(name="gn_init")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # The main body of the resnet
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i, strides=strides, conv=conv, norm=norm, act=act)(x)

        if self.num_kp is not None:
            return SpatialSoftmax(num_kp=self.num_kp)(x)
        if self.attention_pool:
            return AttentionPool2d(num_heads=x.shape[-1] // 16)(x)
        if self.average_pool:
            return jnp.mean(x, axis=(-3, -2))  # (..., H, W, C) -> (B, T, C).
        return x  # (B, T, H, W, C)


class ResNetDecoder(nn.Module):
    """ResNet Decoder Network"""

    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = ResNetBlock
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    spatial_coordinates: bool = False

    @nn.compact
    def __call__(self, z, obs, goal: Optional[jax.Array] = None, train: bool = True):
        assert self.block_cls is ResNetBlock, "BottleNeckBlock not yet implemented."
        assert goal is None, "goal not yet supported."

        # Initialize layers
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, 32, epsilon=1e-5, dtype=self.dtype)
        act = getattr(jax.nn, self.act)

        # Run initial projection
        x = nn.Dense(512)(z)
        x = jnp.reshape(x, x.shape[:-1] + (1, 1, 512))

        # These parameters are known to work for 84x84, 128x128, 224x224
        h, w, _ = obs.shape[-3:]
        scale_h, scale_w = round(h / 32) + 1, round(w / 32) + 1
        num_with_padding = round(h / 64) + 1

        x = jax.image.resize(x, shape=x.shape[:-3] + (scale_h, scale_w, 512), method="nearest")
        # Go through the same computations as before, but reverse the stage sizes.
        for i, block_size in reversed(list(enumerate(self.stage_sizes))):
            for j in reversed(list(range(block_size))):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                padding = 2 if i < num_with_padding else "SAME"
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=partial(conv, padding=padding),
                    norm=norm,
                    act=act,
                    transpose=True,
                )(x)

        x = jax.image.resize(x, shape=x.shape[:-3] + (2 * x.shape[-3], 2 * x.shape[-2], x.shape[-1]), method="nearest")
        if obs.shape[-3] == x.shape[-3]:
            output_pad = "PAD"
        else:
            assert obs.shape[-3] > x.shape[-3]
            output_pad = obs.shape[-3] - x.shape[-3]

        output_pad = "SAME" if obs.shape[-3] == x.shape[-3] else obs.shape[-3] - x.shape[-3] - 1
        x = conv(3, (3, 3), padding=output_pad)(x)

        # TODO: maybe add a final activation and/or project values back to -1 to 1?
        # For now image labels are in 0, 1 range from tfds
        return jnp.reshape(x, obs.shape)


class ResNet18(ResNet):
    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    block_cls: ModuleDef = ResNetBlock


class ResNet34(ResNet):
    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = ResNetBlock


class ResNet50(ResNet):
    stage_sizes: Sequence[int] = (3, 4, 6, 3)
    block_cls: ModuleDef = BottleneckResNetBlock


class ResNet18Decoder(ResNetDecoder):
    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    block_cls: ModuleDef = ResNetBlock
