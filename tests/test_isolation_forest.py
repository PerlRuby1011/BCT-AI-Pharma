"""Unit tests for ai_modules.isolation_forest_detector."""
from __future__ import annotations

import numpy as np

from ai_modules.isolation_forest_detector import (
    ANOMALY_TYPES,
    IsolationForestConfig,
    calibrate_score_normalization,
    evaluate_all_anomaly_types,
    evaluate_anomaly_type,
    normalized_anomaly_scores,
    train_isolation_forest,
)


def test_train_isolation_forest_returns_fitted_model() -> None:
    config = IsolationForestConfig(n_estimators=10, max_samples=32)
    model = train_isolation_forest(config, n_readings=500, seed=1)
    assert hasattr(model, "predict")
    assert model.n_estimators == 10


def test_calibrate_score_normalization_bounds_are_ordered() -> None:
    config = IsolationForestConfig(n_estimators=10, max_samples=32)
    model = train_isolation_forest(config, n_readings=500, seed=1)
    raw_low, raw_high = calibrate_score_normalization(
        model, config, n_normal_reference=100, n_anomalous_reference_per_type=20, seed=2
    )
    assert raw_high > raw_low


def test_normalized_anomaly_scores_bounded() -> None:
    config = IsolationForestConfig(n_estimators=10, max_samples=32)
    model = train_isolation_forest(config, n_readings=500, seed=1)
    calibration = calibrate_score_normalization(
        model, config, n_normal_reference=100, n_anomalous_reference_per_type=20, seed=2
    )
    X = np.random.default_rng(0).normal(0, 1, size=(20, config.n_features)).astype(np.float32)
    scores = normalized_anomaly_scores(model, X, calibration)
    assert scores.shape == (20,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_normalized_anomaly_scores_discriminate_normal_from_anomalous() -> None:
    """A fixed calibration should score a clearly nominal batch low and a
    strongly anomalous batch high -- not both near 1.0, which was the
    behavior under the old per-batch min/max normalization bug."""
    config = IsolationForestConfig(n_estimators=50, max_samples=128)
    model = train_isolation_forest(config, n_readings=3000, seed=1)
    calibration = calibrate_score_normalization(model, config, seed=2)

    rng = np.random.default_rng(5)
    from ai_modules.isolation_forest_detector import _apply_anomaly, _generate_normal_readings

    normal_batch = _generate_normal_readings(50, rng)
    spike_batch = _apply_anomaly(_generate_normal_readings(50, rng), "Sudden Spike", rng)

    normal_scores = normalized_anomaly_scores(model, normal_batch, calibration)
    spike_scores = normalized_anomaly_scores(model, spike_batch, calibration)

    assert normal_scores.mean() < spike_scores.mean()


def test_evaluate_anomaly_type_returns_expected_keys() -> None:
    config = IsolationForestConfig(n_estimators=10, max_samples=32)
    model = train_isolation_forest(config, n_readings=500, seed=1)
    calibration = calibrate_score_normalization(model, config, seed=2)
    result = evaluate_anomaly_type(
        model, config, "Sudden Spike", calibration, n_episodes=5, episode_length=20, seed=2
    )
    for key in ("detection", "fpr", "latency_min", "latency_std_min"):
        assert key in result
    assert 0.0 <= result["detection"] <= 1.0
    assert 0.0 <= result["fpr"] <= 1.0


def test_evaluate_all_anomaly_types_includes_overall() -> None:
    config = IsolationForestConfig(n_estimators=10, max_samples=32)
    model = train_isolation_forest(config, n_readings=500, seed=1)
    results = evaluate_all_anomaly_types(model, config, seed=3)

    for anomaly_type in ANOMALY_TYPES:
        assert anomaly_type in results
    assert "overall" in results
    assert 0.0 <= results["overall"]["detection"] <= 1.0
