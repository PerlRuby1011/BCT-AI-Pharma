"""Shared helpers used across the BCT-AI-Pharma codebase: config loading,
deterministic seeding, and lightweight logging setup."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _load_dotenv(env_path: Path = REPO_ROOT / ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables are never overwritten (the process
    environment always wins over the file). No-op if ``env_path`` doesn't
    exist -- ``.env`` is optional; see ``.env.example``.

    Args:
        env_path: Path to the ``.env`` file.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the project YAML configuration file into a nested dictionary.

    Honors the ``BCT_AI_CONFIG_PATH``, ``BCT_AI_RANDOM_SEED``, and
    ``BCT_AI_RESULTS_DIR`` environment variables (see ``.env.example``) as
    overrides applied after the YAML is parsed.

    Args:
        config_path: Path to the config YAML file. Defaults to
            ``BCT_AI_CONFIG_PATH`` if set, else ``config/config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = os.environ.get("BCT_AI_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed_override = os.environ.get("BCT_AI_RANDOM_SEED")
    if seed_override:
        config["random_seed"] = int(seed_override)

    results_dir_override = os.environ.get("BCT_AI_RESULTS_DIR")
    if results_dir_override:
        config["paths"]["results_dir"] = results_dir_override
        config["paths"]["simulation_results_csv"] = str(
            Path(results_dir_override) / "simulation_results.csv"
        )
        config["paths"]["performance_metrics_json"] = str(
            Path(results_dir_override) / "performance_metrics.json"
        )

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


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Create (or fetch) a configured console logger.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        level: Logging level. Defaults to the ``BCT_AI_LOG_LEVEL``
            environment variable (``DEBUG``/``INFO``/``WARNING``/``ERROR``)
            if set, else ``INFO``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    if level is None:
        level_name = os.environ.get("BCT_AI_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

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
