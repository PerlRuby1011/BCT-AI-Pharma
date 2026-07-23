"""Unit tests for evaluation.baseline_comparison."""
from __future__ import annotations

from evaluation.baseline_comparison import compute_improvement_ratios, compute_recall_localization
from utils import load_config


def test_recall_localization_returns_expected_keys() -> None:
    config = load_config()
    result = compute_recall_localization(config)

    assert result["n_locations"] == 847
    assert result["bct_ai_accuracy_pct"] == 100.0
    assert result["bct_ai_time_minutes"] > 0
    assert "paper_reference" in result
    assert result["paper_reference"]["time_minutes"] == 2.8


def test_recall_localization_time_is_much_faster_than_traditional() -> None:
    """Whatever the exact minutes figure, BCT-AI's location-by-location
    recall must be dramatically faster than the Traditional baseline's
    72.3-hour localization time -- the qualitative claim the paper makes,
    even though this repository's from-first-principles formula does not
    reproduce the paper's exact "2.8 minutes" (see docstring in
    evaluation/baseline_comparison.py: the paper does not specify enough
    of its recall-localization methodology to reproduce that figure
    exactly from the network's documented TPS/overhead parameters alone)."""
    config = load_config()
    result = compute_recall_localization(config)
    assert result["bct_ai_time_minutes"] / 60.0 < result["traditional_time_hours"] / 100.0


def test_improvement_ratios_match_table_vi_arithmetic() -> None:
    config = load_config()
    ratios = compute_improvement_ratios(config)

    assert 4.0 <= ratios["vs_traditional_ratio"] <= 4.3
    assert 2.2 <= ratios["vs_blockchain_only_ratio"] <= 2.5
    assert ratios["paper_stated_ratio"] == 4.1
    # The paper's stated 4.1x matches vs-Traditional, not vs-Blockchain-Only.
    assert abs(ratios["vs_traditional_ratio"] - ratios["paper_stated_ratio"]) < 0.1
    assert abs(ratios["vs_blockchain_only_ratio"] - ratios["paper_stated_ratio"]) > 1.0
