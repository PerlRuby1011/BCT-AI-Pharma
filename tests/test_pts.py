"""Unit tests for pts.product_trust_score and pts.pts_sensitivity_analysis."""
from __future__ import annotations

import pytest

from pts.product_trust_score import (
    ProductState,
    compute_pts,
    compute_pts_for_drug_class,
    score_age_shelf_life,
    score_ai_confidence,
    score_anomaly_history,
    score_handling_quality,
    score_provenance_integrity,
    score_regulatory_status,
    score_temperature_compliance,
    score_verification_frequency,
    weights_for_drug_class,
)
from pts.pts_sensitivity_analysis import run_sensitivity_analysis, summarize_sensitivity

PTS_CONFIG = {
    "drug_classes": {
        "A": {
            "w1_provenance_integrity": 0.25,
            "w2_temperature_compliance": 0.35,
            "w8_ai_confidence": 0.10,
            "remaining_total": 0.30,
        },
        "B": {
            "w1_provenance_integrity": 0.25,
            "w2_temperature_compliance": 0.30,
            "w8_ai_confidence": 0.10,
            "remaining_total": 0.35,
        },
        "C": {
            "w1_provenance_integrity": 0.35,
            "w2_temperature_compliance": 0.15,
            "w8_ai_confidence": 0.10,
            "remaining_total": 0.40,
        },
    },
    "alert_threshold": 0.75,
    "quarantine_threshold": 0.50,
    "sensitivity_analysis": {
        "drug_class": "A",
        "varied_weights": ["w1_provenance_integrity", "w2_temperature_compliance"],
        "variation_pct": 0.15,
        "n_steps": 5,
    },
}


def test_score_provenance_integrity_is_product_of_trust() -> None:
    state = ProductState(custody_chain_trust_scores=[1.0, 0.5, 0.8])
    assert score_provenance_integrity(state) == pytest.approx(0.4)


def test_score_temperature_compliance_perfect_when_within_tolerance() -> None:
    state = ProductState(
        temperature_readings_c=[5.0, 5.5, 4.5],
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
    )
    assert score_temperature_compliance(state) == pytest.approx(1.0)


def test_score_temperature_compliance_penalizes_excursion() -> None:
    state = ProductState(
        temperature_readings_c=[15.0, 15.0, 15.0],
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=10.0,
    )
    score = score_temperature_compliance(state)
    assert 0.0 <= score < 1.0


def test_score_handling_quality_range() -> None:
    state = ProductState(shock_events_severity=[0.2, 0.3], max_allowed_shocks=5.0)
    assert score_handling_quality(state) == pytest.approx(1 - 0.5 / 5.0)


def test_score_verification_frequency_capped_at_one() -> None:
    state = ProductState(verification_count=10, expected_verifications=3)
    assert score_verification_frequency(state) == 1.0


def test_score_age_shelf_life_decays_over_time() -> None:
    fresh = ProductState(days_since_manufacture=0, shelf_life_days=365)
    old = ProductState(days_since_manufacture=365, shelf_life_days=365)
    assert score_age_shelf_life(fresh) == pytest.approx(1.0)
    assert score_age_shelf_life(old) == pytest.approx(0.5, rel=1e-3)


def test_score_anomaly_history_range() -> None:
    state = ProductState(anomaly_count=2, max_anomalies=5)
    assert score_anomaly_history(state) == pytest.approx(0.6)


def test_score_regulatory_status_values() -> None:
    assert score_regulatory_status(ProductState(regulatory_status="good_standing")) == 1.0
    assert score_regulatory_status(ProductState(regulatory_status="under_investigation")) == 0.5
    assert score_regulatory_status(ProductState(regulatory_status="sanctioned")) == 0.0

    with pytest.raises(ValueError):
        score_regulatory_status(ProductState(regulatory_status="unknown"))


def test_score_ai_confidence_averages_cnn_and_inverse_isolation() -> None:
    state = ProductState(cnn_authenticity_score=0.9, isolation_forest_anomaly_score=0.1)
    assert score_ai_confidence(state) == pytest.approx((0.9 + 0.9) / 2)


def test_weights_for_drug_class_sum_matches_expected_total() -> None:
    weights = weights_for_drug_class(PTS_CONFIG["drug_classes"]["A"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["provenance_integrity"] == 0.25
    assert weights["temperature_compliance"] == 0.35
    assert weights["ai_confidence"] == 0.10


def test_compute_pts_perfect_product_scores_near_one() -> None:
    state = ProductState(
        custody_chain_trust_scores=[1.0, 1.0],
        temperature_readings_c=[5.0, 5.0, 5.0],
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        shock_events_severity=[],
        verification_count=3,
        expected_verifications=3,
        days_since_manufacture=0,
        shelf_life_days=730,
        anomaly_count=0,
        max_anomalies=5,
        regulatory_status="good_standing",
        cnn_authenticity_score=1.0,
        isolation_forest_anomaly_score=0.0,
    )
    weights = weights_for_drug_class(PTS_CONFIG["drug_classes"]["A"])
    result = compute_pts(state, weights)
    assert result["pts"] == pytest.approx(1.0, abs=1e-6)
    assert result["status"] == "active"


def test_compute_pts_for_drug_class_triggers_quarantine_on_severe_excursion() -> None:
    state = ProductState(
        custody_chain_trust_scores=[0.9],
        temperature_readings_c=[25.0] * 10,
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=5.0,
        shock_events_severity=[0.5, 0.5],
        verification_count=0,
        expected_verifications=3,
        days_since_manufacture=600,
        shelf_life_days=730,
        anomaly_count=4,
        max_anomalies=5,
        regulatory_status="under_investigation",
        cnn_authenticity_score=0.5,
        isolation_forest_anomaly_score=0.9,
    )
    result = compute_pts_for_drug_class(state, "A", PTS_CONFIG)
    assert result["pts"] < 0.50
    assert result["status"] == "quarantine"


def test_sensitivity_analysis_produces_expected_component_curves() -> None:
    results = run_sensitivity_analysis(PTS_CONFIG)
    assert set(results.keys()) == {"provenance_integrity", "temperature_compliance"}
    for rows in results.values():
        assert len(rows) == 5
        deltas = [row["delta_pct"] for row in rows]
        assert deltas == sorted(deltas)

    summary = summarize_sensitivity(results)
    assert "temperature_compliance" in summary
    assert "pts_delta_per_10pct" in summary["temperature_compliance"]
