import csv
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from functools import partial
from typing import Dict, Optional

import numpy as np
import tensorboard
import tensorflow as tf

"""
Generally taken from

https://github.com/jhejna/research-lightning/blob/main/research/utils/logger.py
"""

try:
    import wandb

    WANDB_IMPORTED = True
except ModuleNotFoundError:
    WANDB_IMPORTED = False


class Writer(ABC):
    def __init__(self, path, on_prefix: str | None = None):
        self.path = path
        self.on_prefix = on_prefix
        self.values = {}

    def update(self, d: Dict) -> None:
        self.values.update(d)

    def dump(self, step: int, prefix: str | None = None) -> None:
        if self.on_prefix is None or prefix == str(self.on_prefix):
            self._dump(step)

    @abstractmethod
    def _dump(self, step: int) -> None:
        return NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class TensorBoardWriter(Writer):
    def __init__(self, path, on_prefix=None):
        super().__init__(path, on_prefix=on_prefix)
        self.writer = tensorboard.summary.Writer(self.path)

    def _dump(self, step):
        for k, v in self.values.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()
        self.values.clear()

    def close(self):
        self.writer.close()


class CSVWriter(Writer):
    def __init__(self, path, on_prefix="val"):
        assert on_prefix is not None, "on_prefix must be set for CSVWriter."
        super().__init__(path, on_prefix=on_prefix)
        self._csv_path = tf.io.gfile.join(self.path, on_prefix + ".csv")
        self._csv_file_handler = None
        self.csv_logger = None
        self.num_keys = 0

        # If we are continuing to train, make sure that we know how many keys to expect.
        if tf.io.gfile.exists(self._csv_path):
            with tf.io.gfile.GFile(self._csv_path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames.copy()
                num_keys = len(fieldnames)
            if num_keys > self.num_keys:
                self.num_keys = num_keys
                # Create a new CSV handler with the fieldnames set.
                self.csv_file_handler = tf.io.gfile.GFile(self._csv_path, "a")
                self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(fieldnames))

    def _reset_csv_handler(self):
        if self._csv_file_handler is not None:
            self._csv_file_handler.close()  # Close our fds
        self.csv_file_handler = tf.io.gfile.GFile(self._csv_path, "w")  # Write a new one
        self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.values.keys()))
        self.csv_logger.writeheader()

    def update(self, d: Dict) -> None:
        # Override this method only for csv to ignore timing metrics
        self.values.update({k: v for k, v in d.items() if not k.startswith("time")})

    def _dump(self, step):
        # Record the step
        self.values["step"] = step
        if len(self.values) < self.num_keys:
            # We haven't gotten all keys yet, return without doing anything.
            return
        if len(self.values) > self.num_keys:
            # Got a new key, so re-create the writer
            self.num_keys = len(self.values)
            # We encountered a new key. We need to recreate the file handler and overwrite old data
            self._reset_csv_handler()

        # We should now have all the keys
        self.csv_logger.writerow(self.values)
        self.csv_file_handler.flush()
        # Note: Don't reset the CSV because the file handler doesn't support it.

    def close(self):
        self.csv_file_handler.close()


class WandBWriter(Writer):
    def __init__(self, path: str, on_prefix: str | None = None):
        super().__init__(path, on_prefix=on_prefix)

    def _dump(self, step: int) -> None:
        wandb.log(self.values, step=step)
        self.values.clear()  # reset the values

    def close(self) -> None:
        wandb.finish()


class Logger(object):
    def __init__(self, path: str, writers: Iterable[str] = ("csv",)):
        writers = set(writers)  # Avoid duplication
        if WANDB_IMPORTED and wandb.run is not None:
            writers.add("wandb")  # If wandb is initialized, make sure we have it.

        self.writers = []
        for writer in writers:
            self.writers.append(
                {
                    "tb": TensorBoardWriter,
                    "csv": partial(CSVWriter, on_prefix="val"),
                    "wandb": WandBWriter,
                    "eval": partial(CSVWriter, on_prefix="eval"),
                }[writer](path)
            )

    def update(self, d: Dict, prefix: Optional[str] = None) -> None:
        d = {k: np.mean(v) for k, v in d.items()}
        if prefix is not None:
            d = {prefix + "/" + k: v for k, v in d.items()}
        for writer in self.writers:
            writer.update(d)

    def dump(self, step: int, prefix: str | None = None) -> None:
        for writer in self.writers:
            writer.dump(step, prefix=prefix)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()


class DummyLogger(object):
    """
    A Dummy Logger util that just doesn't do anything
    (for use in other workers when you don't want to think about code logic)
    """

    def __init__(self, *args, **kwargs):
        return

    def update(self, *args, **kwargs):
        return

    def dump(self, *args, **kwargs):
        return

    def close(self, *args, **kwargs):
        return


class Timer:
    """
    Inspired by https://github.com/rail-berkeley/orca/blob/main/octo/utils/train_utils.py#L78
    """

    def __init__(self):
        self._times = defaultdict(float)
        self._counts = defaultdict(int)

    @property
    def times(self):
        return {k: self._times[k] / self._counts[k] for k in self._times}

    def reset(self):
        self._times = defaultdict(float)
        self._counts = defaultdict(int)

    @contextmanager
    def __call__(self, key: str):
        start_time = time.time()
        try:
            yield None
        finally:
            self._times[key] += time.time() - start_time
            self._counts[key] += 1
