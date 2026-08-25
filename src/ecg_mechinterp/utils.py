"""Small shared helpers: seeding, timing, and a disk cache keyed by filename."""
from __future__ import annotations

import os
import pickle
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[{name}] {time.time() - t0:.1f}s")


class DiskCache:
    """Pickle cache under `cache_dir`, one file per name. Used to avoid recomputing
    activation extraction or SAE training across notebook/script re-runs."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def exists(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def save(self, obj, name: str) -> None:
        with open(self.path(name), "wb") as f:
            pickle.dump(obj, f)

    def load(self, name: str):
        with open(self.path(name), "rb") as f:
            return pickle.load(f)
