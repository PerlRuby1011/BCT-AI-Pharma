"""Unit tests for ai_modules.lstm_forecasting."""
from __future__ import annotations

import numpy as np

from ai_modules.lstm_forecasting import (
    LSTMConfig,
    build_lstm_model,
    evaluate_lstm,
    generate_synthetic_sequences,
    train_lstm,
)


def test_generate_synthetic_sequences_shapes() -> None:
    config = LSTMConfig(input_timesteps=20, n_features=15)
    X, y_demand, y_delay = generate_synthetic_sequences(50, config, seed=1)

    assert X.shape == (50, 20, 15)
    assert y_demand.shape == (50,)
    assert y_delay.shape == (50,)
    assert set(np.unique(y_delay)).issubset({0.0, 1.0})


def test_build_lstm_model_output_shapes() -> None:
    config = LSTMConfig(input_timesteps=20, n_features=15, units_layer1=8, units_layer2=4, dense_units=4)
    model = build_lstm_model(config)
    X, _, _ = generate_synthetic_sequences(4, config, seed=2)

    demand_pred, delay_pred = model.predict(X, verbose=0)
    assert demand_pred.shape == (4, 1)
    assert delay_pred.shape == (4, 1)
    assert np.all((delay_pred >= 0) & (delay_pred <= 1))


def test_train_and_evaluate_lstm_runs_end_to_end() -> None:
    config = LSTMConfig(
        input_timesteps=10,
        n_features=15,
        units_layer1=8,
        units_layer2=4,
        dense_units=4,
        early_stopping_patience=2,
        max_epochs=2,
        batch_size=16,
    )
    model, history = train_lstm(config, n_train=64, n_val=16, seed=3)

    assert "loss" in history
    assert len(history["loss"]) >= 1

    metrics = evaluate_lstm(model, config, n_test=16, seed=4)
    for key in ("mape", "r2", "auc_roc", "f1"):
        assert key in metrics
    assert metrics["mape"] >= 0
