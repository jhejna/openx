import abc
from typing import Any, Dict, Tuple

import jax
import optax
from flax.training import train_state


class Algorithm(abc.ABC):
    """
    A base implementation of an algorithm. Contains all attributes to be used by train.py
    """

    @abc.abstractmethod
    def init(self, batch, tx: optax.GradientTransformation, rng: jax.Array) -> train_state.TrainState:
        """
        Returns a state object and initializes the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(
        self, state: train_state.TrainState, batch: Dict, rng: jax.Array
    ) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        """
        The main train step function. This should return a dictionary of metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def val_step(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Dict[str, jax.Array]:
        """
        Take a validation step and return a dictionary of metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Any:
        raise NotImplementedError
