"""Unit tests for ai_modules.cnn_verification."""
from __future__ import annotations

import numpy as np
import pytest

from ai_modules.cnn_verification import (
    CNNConfig,
    build_cnn_model,
    evaluate_cnn,
    generate_synthetic_packaging_images,
    quantize_and_benchmark,
    train_cnn,
)


@pytest.fixture(scope="module")
def small_config() -> CNNConfig:
    return CNNConfig(image_size=32, dense_units_1=32, dense_units_2=16, max_epochs=2, batch_size=8)


def test_generate_synthetic_packaging_images_shapes(small_config: CNNConfig) -> None:
    images, auth_labels, tamper_labels = generate_synthetic_packaging_images(
        n_authentic=8, n_tampered_per_class=4, config=small_config, seed=1
    )
    n_expected = 8 + 4 * 4
    assert images.shape == (n_expected, 3, 32, 32)
    assert auth_labels.shape == (n_expected,)
    assert tamper_labels.shape == (n_expected,)
    assert set(np.unique(tamper_labels)).issubset(set(range(5)))
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_build_cnn_model_forward_pass(small_config: CNNConfig) -> None:
    import torch

    model = build_cnn_model(small_config)
    sample = torch.randn(2, 3, small_config.image_size, small_config.image_size)
    auth_logits, tamper_logits = model(sample)
    assert auth_logits.shape == (2,)
    assert tamper_logits.shape == (2, 5)


def test_train_evaluate_and_quantize_cnn(small_config: CNNConfig) -> None:
    model, history = train_cnn(small_config, n_authentic=8, n_tampered_per_class=4, seed=2)
    assert len(history["train_loss"]) >= 1

    metrics = evaluate_cnn(model, small_config, n_test_per_class=3, seed=3)
    assert "weighted_avg" in metrics
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= metrics["weighted_avg"][key] <= 1.0

    quant_metrics = quantize_and_benchmark(model, small_config)
    assert quant_metrics["model_size_mb"] > 0
    assert quant_metrics["inference_latency_ms"] > 0
