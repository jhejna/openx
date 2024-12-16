from typing import Dict

from flax import linen as nn

"""
Defines the core model
"""


class MultiEncoder(nn.Module):
    encoders: Dict[str, nn.Module]
    trunk: nn.Module

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
        return modalities

    def __call__(self, batch: Dict, train: bool = True):
        # Exists so we get full tracing with module.init
        # Should not be used for training.
        x = self._encode(batch, train=train)
        return self.trunk(x, train=train)
