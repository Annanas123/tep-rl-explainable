"""
reproducibility.py - Deterministic seeding for fully reproducible experiments.

Every experiment run calls ``seed_everything(seed)`` once before any
randomness is introduced.  The function seeds:

  * Python's built-in ``random`` module
  * NumPy (global generator and a returned ``np.random.Generator``)
  * PyTorch CPU and CUDA kernels
  * The PYTHONHASHSEED environment variable (effective for the current process)

Usage
-----
    from tep_rl.reproducibility import seed_everything, get_run_id

    rng = seed_everything(seed=7)
    run_id = get_run_id(experiment_name="proxy_moppo", seed=7)
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> np.random.Generator:
    """
    Seed all random-number generators that could affect training results.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.  Use one of ``THESIS_SEEDS`` for
        results reported in the thesis.

    Returns
    -------
    np.random.Generator
        A freshly seeded NumPy generator that can be passed to environments
        or other components that accept one.
    """
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)                        # Used by libraries that rely on global state.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA ops may reduce GPU throughput slightly.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("Global seed set to %d", seed)
    return np.random.default_rng(seed)


def get_run_id(experiment_name: str, seed: int, timestamp: Optional[str] = None) -> str:
    """
    Build a short, human-readable run identifier that encodes the experiment
    name, seed, and wall-clock time.  Suitable as a sub-directory name.

    Example
    -------
    ``"proxy_moppo_seed007_20240315T143022"``
    """
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{experiment_name}_seed{seed:03d}_{ts}"


def config_fingerprint(config_dict: dict) -> str:
    """
    Return a short hex digest of a configuration dictionary so that two runs
    with identical hyperparameters produce the same fingerprint.  Useful for
    de-duplicating cached results.
    """
    canonical = str(sorted(config_dict.items())).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]
