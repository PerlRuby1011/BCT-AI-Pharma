"""Shared helpers used across the BCT-AI-Pharma codebase: config loading,
deterministic seeding, and lightweight logging setup."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the project YAML configuration file into a nested dictionary.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy (and, if installed, TensorFlow/PyTorch) RNGs.

    Args:
        seed: Random seed to apply everywhere for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured console logger.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        level: Logging level.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Args:
        relative_path: Path relative to the repo root (e.g. ``"results/foo.csv"``).

    Returns:
        Absolute ``Path`` under the repository root.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
