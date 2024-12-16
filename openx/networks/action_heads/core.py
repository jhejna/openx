import abc
from typing import Optional

import jax
from flax import linen as nn


class ActionHead(nn.Module, abc.ABC):
    action_dim: int
    action_horizon: Optional[int] = None
    model: Optional[nn.Module] = None

    @abc.abstractmethod
    def predict(self, obs: jax.Array, train: bool = True):
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        raise NotImplementedError
