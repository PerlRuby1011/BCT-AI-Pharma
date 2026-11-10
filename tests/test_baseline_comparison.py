"""Unit tests for evaluation.baseline_comparison (recall-localization estimate)."""
from __future__ import annotations

from evaluation.baseline_comparison import compute_recall_localization
from historical.bcca2026_reference import reference_recall_localization
from utils import load_config


def test_recall_localization_returns_expected_keys() -> None:
    config = load_config()
    result = compute_recall_localization(config)

    assert result["n_locations"] == 847
    assert result["bct_ai_time_minutes"] > 0
    assert "not an observed time" in result["basis"]
    # Conference figures are not part of the computed output any more.
    assert "paper_reference" not in result
    assert "bct_ai_accuracy_pct" not in result
    assert "traditional_time_hours" not in result


def test_recall_localization_formula() -> None:
    config = load_config()
    result = compute_recall_localization(config)
    tps = config["recall_localization_model"]["tps"]
    overhead = config["blockchain"]["ai_inference_overhead_ms"]
    expected = 847 * (1 / tps + overhead / 1000) / 60
    assert abs(result["bct_ai_time_minutes"] - round(expected, 1)) < 1e-9


def test_calculated_time_differs_from_conference_reported_time() -> None:
    """The calculated estimate does not reproduce the conference's 2.8 min;
    that figure is available only as a labelled reference value."""
    config = load_config()
    calculated = compute_recall_localization(config)["bct_ai_time_minutes"]
    reference = reference_recall_localization(config)
    assert reference["time_minutes"] == 2.8
    assert calculated != reference["time_minutes"]
