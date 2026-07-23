"""Unit tests for utils.py, in particular the .env.example environment overrides."""
from __future__ import annotations

import os

import pytest

from utils import load_config


@pytest.fixture(autouse=True)
def _clean_env():
    keys = ["BCT_AI_RANDOM_SEED", "BCT_AI_RESULTS_DIR", "BCT_AI_CONFIG_PATH"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_load_config_without_env_overrides_uses_yaml_defaults() -> None:
    config = load_config()
    assert config["random_seed"] == 42
    assert config["paths"]["results_dir"] == "results"


def test_bct_ai_random_seed_env_var_overrides_config() -> None:
    os.environ["BCT_AI_RANDOM_SEED"] = "999"
    config = load_config()
    assert config["random_seed"] == 999


def test_bct_ai_results_dir_env_var_overrides_all_result_paths() -> None:
    os.environ["BCT_AI_RESULTS_DIR"] = "/tmp/bct_test_results"
    config = load_config()
    assert config["paths"]["results_dir"] == "/tmp/bct_test_results"
    assert config["paths"]["simulation_results_csv"] == "/tmp/bct_test_results/simulation_results.csv"
    assert config["paths"]["performance_metrics_json"] == "/tmp/bct_test_results/performance_metrics.json"
