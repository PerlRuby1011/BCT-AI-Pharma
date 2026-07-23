"""Unit tests for the Monte Carlo degradation simulations in evaluation.run_simulation."""
from __future__ import annotations

import pytest

from ai_modules import cnn_verification as cnn_mod
from ai_modules import isolation_forest_detector as if_mod
from evaluation.run_simulation import (
    simulate_adversarial_gan_monte_carlo,
    simulate_mobile_image_variability_monte_carlo,
    simulate_sensor_drift_monte_carlo,
)
from utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def if_results(config):
    if_cfg = config["isolation_forest"]
    model_config = if_mod.IsolationForestConfig(
        n_estimators=20, max_samples=64, contamination=if_cfg["contamination"]
    )
    model = if_mod.train_isolation_forest(model_config, n_readings=3000, seed=1)
    calibration = if_mod.calibrate_score_normalization(model, model_config, seed=2)
    return {"_artifacts": {"model": model, "model_config": model_config, "calibration": calibration}}


@pytest.fixture(scope="module")
def cnn_results():
    model_config = cnn_mod.CNNConfig(
        image_size=32, dense_units_1=32, dense_units_2=16, max_epochs=3, batch_size=8, pretrained=False
    )
    model, _ = cnn_mod.train_cnn(model_config, n_authentic=20, n_tampered_per_class=5, seed=1)
    return {"_artifacts": {"model": model, "model_config": model_config}}


def test_sensor_drift_monte_carlo_returns_expected_keys(if_results, config) -> None:
    result = simulate_sensor_drift_monte_carlo(if_results, config, n_samples=30)
    for key in (
        "n_samples",
        "baseline_detection_mean",
        "baseline_detection_std",
        "degraded_detection_mean",
        "degraded_detection_std",
        "paper_reference",
    ):
        assert key in result
    assert 0.0 <= result["baseline_detection_mean"] <= 1.0
    assert 0.0 <= result["degraded_detection_mean"] <= 1.0


def test_mobile_image_variability_monte_carlo_returns_expected_keys(cnn_results, config) -> None:
    result = simulate_mobile_image_variability_monte_carlo(cnn_results, config, n_samples=25)
    assert "baseline_detection_mean" in result
    assert "degraded_detection_mean" in result
    assert result["n_samples"] > 0
    assert 0.0 <= result["baseline_detection_mean"] <= 1.0
    assert 0.0 <= result["degraded_detection_mean"] <= 1.0


def test_adversarial_gan_monte_carlo_returns_expected_keys(cnn_results, config) -> None:
    result = simulate_adversarial_gan_monte_carlo(cnn_results, config, n_samples=25)
    assert "baseline_detection_mean" in result
    assert "adversarial_detection_mean" in result
    assert result["epsilon"] > 0
    assert 0.0 <= result["baseline_detection_mean"] <= 1.0
    assert 0.0 <= result["adversarial_detection_mean"] <= 1.0
