"""Unit tests for evaluation.run_simulation and evaluation.statistical_validation."""
from __future__ import annotations

import copy

from evaluation.run_simulation import run_all_simulation_runs, run_single_simulation_run
from evaluation.statistical_validation import compute_statistical_validation_from_runs
from utils import load_config


def _small_config():
    """A config clone with a tiny transaction count and 3 runs, for fast tests."""
    config = load_config()
    config = copy.deepcopy(config)
    config["simulation"]["transactions_per_run"] = 5000
    config["statistical_validation"]["n_runs"] = 3
    return config


def test_run_single_simulation_run_returns_expected_keys() -> None:
    config = _small_config()
    result = run_single_simulation_run(config, seed=42)

    for key in (
        "seed",
        "n_transactions",
        "n_counterfeit",
        "n_anomalies",
        "bct_ai_counterfeit_detection",
        "ai_only_counterfeit_detection",
        "bct_ai_recall_efficiency",
        "ai_only_recall_efficiency",
    ):
        assert key in result

    assert result["n_transactions"] == 5000
    assert 0.0 <= result["bct_ai_counterfeit_detection"] <= 1.0
    assert 0.0 <= result["ai_only_counterfeit_detection"] <= 1.0


def test_different_seeds_produce_different_runs() -> None:
    config = _small_config()
    result_a = run_single_simulation_run(config, seed=42)
    result_b = run_single_simulation_run(config, seed=43)

    # Genuinely independent runs should not be bitwise identical.
    assert result_a["bct_ai_counterfeit_detection"] != result_b["bct_ai_counterfeit_detection"]


def test_bct_ai_outperforms_ai_only_baseline() -> None:
    """Sanity check the qualitative claim the paper makes: the integrated
    blockchain+AI system should detect counterfeits at a higher rate than
    an AI-only system lacking the blockchain provenance signal."""
    config = _small_config()
    result = run_single_simulation_run(config, seed=42)
    assert result["bct_ai_counterfeit_detection"] > result["ai_only_counterfeit_detection"]
    assert result["bct_ai_recall_efficiency"] > result["ai_only_recall_efficiency"]


def test_run_all_simulation_runs_uses_sequential_seeds() -> None:
    config = _small_config()
    runs = run_all_simulation_runs(config)
    assert len(runs) == 3
    seeds = [r["seed"] for r in runs]
    assert seeds == [42, 43, 44]


def test_compute_statistical_validation_from_runs_produces_real_ttest() -> None:
    config = _small_config()
    runs = run_all_simulation_runs(config)
    result = compute_statistical_validation_from_runs(runs)

    assert result["n_runs"] == 3
    assert "counterfeit_detection_ttest" in result
    assert "recall_efficiency_ttest" in result
    assert result["counterfeit_detection_ttest"]["degrees_of_freedom"] == 2
    # The genuine t-statistic need not match the paper's reported 6.42 --
    # only that it is a real, finite number derived from actual per-run data.
    assert not __import__("math").isnan(result["counterfeit_detection_ttest"]["t_statistic"])
