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
        x = jnp.concatenate([modalities[k] for k in sorted(modalities.keys())], axis=-1)  # (B, T, D)
        if self.flatten_time:
            x = x.reshape((x.shape[0], -1))
        if self.model is not None:
            x = self.model(x, train=train)
        return x
