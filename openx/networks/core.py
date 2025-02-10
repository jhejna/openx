from typing import Dict, Optional

import flax
import jax
from flax import linen as nn
from jax import numpy as jnp

"""
Defines the core model components.
"""


class MultiEncoder(nn.Module):
    """
    Takes multiple inputs and returns them as a single representation.
    """

    encoders: Dict[str, nn.Module]  # Encodes things separately
    trunk: nn.Module  # Merges encoder outputs

    def _encode(self, batch: Dict, train: bool = True):
        # Support passing multiple modalities to a single encoder. Specify via tuple-key with `->`
        modalities = dict()
        for encoder_keys, encoder in self.encoders.items():
            # Assemble the args for the different modules
            args = []
            for encoder_key in encoder_keys.split(","):
                v = batch
                for k in encoder_key.split("->"):
                    v = v[k]
                args.append(v)
            args = tuple(args)
            if encoder is None:
                modalities[encoder_keys] = args[0] if len(args) == 1 else args
            else:
                modalities[encoder_keys] = encoder(*args, train=train)
        # For later: consider re-organizing the outputs using flax traversals
        return modalities

    def __call__(self, batch: Dict, train: bool = True):
        x = self._encode(batch, train=train)
        return self.trunk(x, train=train)


class MultiDecoder(nn.Module):
    """
    Takes a single representation and returns them as multiple inputs.
    """

    trunk: nn.Module
    decoders: Dict[str, nn.Module]

    def _decode(self, z, batch: Dict, train: bool = True):
        output = dict()
        for decoder_keys, decoder in self.decoders.items():
            # Assemble the args for the different modules
            args = []
            for decoder_key in decoder_keys.split(","):
                v = batch
                for k in decoder_key.split("->"):
                    v = v[k]
                args.append(v)
            args = tuple(args)
            first_decoder_key = decoder_keys.split(",")[0]
            assert first_decoder_key not in output
            output[first_decoder_key] = z if decoder is None else decoder(z, *args, train=train)

        return flax.traverse_util.unflatten_dict(output, sep="->")

    def __call__(self, z, batch, train: bool = True):
        z = self.trunk(z)
        return self._decode(z, batch, train=train)


class Concatenate(nn.Module):
    model: Optional[nn.Module] = None
    flatten_time: bool = True

    @nn.compact
    def __call__(self, modalities: Dict[str, jax.Array], train: bool = False):
        # TODO(jhejna): consider re-organizing using flax traversals.
        if self.flatten_time:
            x = jnp.concatenate(
                [modalities[k].reshape(modalities[k].shape[0], -1) for k in sorted(modalities.keys())], axis=-1
            )  # (B, D)
        else:
            x = jnp.concatenate(
                [jnp.reshape(modalities[k], modalities[k].shape[:2] + (-1,)) for k in sorted(modalities.keys())],
                axis=-1,
            )  # (B, T, D)
        if self.model is not None:
            x = self.model(x, train=train)
        return x


class Tokenize(nn.Module):
    """
    Tokenizers modalities into (B, T, S, D) or (B, T, D) and feeds it to `model`
    TODO: in a future release see if we can merge this class and the Concatenate class
    """

    embed_dim: int
    flatten_time: bool = True
    project_all: bool = False
    model: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, modalities: Dict[str, jax.Array], train: bool = False):
        # Assume all modalities are shape (B, T, ..., D) or reshape to match
        tokens = []
        for k in sorted(modalities.keys()):
            shape = modalities[k].shape
            if len(shape) == 2:
                new_shape = (shape[0], 1, 1, shape[-1])  # Unsqueeze to add a time and token dim.
            elif len(shape) == 3:
                new_shape = (shape[0], shape[1], 1, shape[-1])  # Unsqueeze to add a token dim.
            elif len(shape) > 3:
                new_shape = (shape[0], shape[1], -1, shape[-1])  # Flatten intermediate dims.
            else:
                new_shape = shape
            modality = jnp.reshape(modalities[k], new_shape)  # Reshape to (B, T, S, D)

            # If we are not at the embed dimension, project.
            if modality.shape[-1] != self.embed_dim or self.project_all:
                modality = nn.Dense(self.embed_dim)(modality)
            tokens.append(modality)

        # Final tokens are all of shape (B, T, S, D)
        b, t = tokens[0].shape[:2]
        if not all(x.shape[1] == t for x in tokens) and not self.flatten_time:
            raise ValueError(
                "flatten_time was not set to True in Tokenize, but not all modalities had the same time dimension."
            )

        if self.flatten_time:
            tokens = [jnp.reshape(x, (b, -1, self.embed_dim)) for x in tokens]

        x = jnp.concatenate(tokens, axis=-2)  # Concat on token dim

        if self.model is not None:
            x = self.model(x, train=train)
        return x
