"""Seeding of DataLoader worker processes, shared by the three split datasets."""

import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and `random` in a DataLoader worker from the seed PyTorch gives it.

    Defined at module level so that it can be pickled when workers are started with
    the spawn method (the default on macOS and Windows).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
