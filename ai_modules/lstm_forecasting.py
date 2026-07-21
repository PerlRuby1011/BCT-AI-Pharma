"""Stacked LSTM network for demand forecasting and shipment delay prediction.

Implements the architecture from Section III-B of the paper: a 128-unit LSTM
feeding a 64-unit LSTM, followed by a 32-unit dense layer, with two output
heads (demand forecast regression and >24h delay-probability classification).
Trained with the Adam optimizer, MSE loss for the regression head, and
binary cross-entropy for the classification head, with dropout regularization
and early stopping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, r2_score, roc_auc_score

FEATURE_NAMES = [
    "daily_sales",
    "inventory_level",
    "shipping_duration",
    "seasonal_month",
    "seasonal_quarter",
    "holiday_proximity",
    "weather_severity",
    "port_congestion",
    "day_of_week",
    "promo_flag",
    "backorder_rate",
    "lead_time_variance",
    "supplier_reliability",
    "regional_demand_index",
    "prior_delay_flag",
]


@dataclass
class LSTMConfig:
    """Hyperparameters for the demand-forecasting / delay-prediction LSTM.

    Attributes:
        input_timesteps: Number of historical time steps in each input sequence.
        n_features: Number of engineered features per time step.
        units_layer1: Units in the first (return-sequence) LSTM layer.
        units_layer2: Units in the second LSTM layer.
        dense_units: Units in the shared dense layer before the output heads.
        dropout: Dropout rate applied after each LSTM layer.
        learning_rate: Adam optimizer learning rate.
        early_stopping_patience: Epochs with no validation improvement before stopping.
        batch_size: Training batch size.
        max_epochs: Maximum number of training epochs.
    """

    input_timesteps: int = 20
    n_features: int = 15
    units_layer1: int = 128
    units_layer2: int = 64
    dense_units: int = 32
    dropout: float = 0.3
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    batch_size: int = 64
    max_epochs: int = 50


def generate_synthetic_sequences(
    n_samples: int, config: LSTMConfig, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic blockchain-derived time-series training data.

    Builds ``n_samples`` sequences of shape ``(timesteps, n_features)`` with
    an embedded latent demand trend and delay-risk signal so the LSTM has a
    learnable target, standing in for real blockchain transaction history.

    Args:
        n_samples: Number of sequences to generate.
        config: LSTM configuration (controls sequence shape).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(X, y_demand, y_delay)`` where ``X`` has shape
        ``(n_samples, timesteps, n_features)``, ``y_demand`` has shape
        ``(n_samples,)`` (continuous demand target), and ``y_delay`` has
        shape ``(n_samples,)`` (binary delay-over-24h label).
    """
    rng = np.random.default_rng(seed)
    t = config.input_timesteps
    f = config.n_features

    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, t, f)).astype(np.float32)
    # Inject a smooth latent trend per sample so sequences are learnable.
    trend = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
    seasonality = np.sin(np.linspace(0, 3 * np.pi, t))[None, :]
    X[:, :, 0] += trend + seasonality  # daily_sales carries the trend
    X[:, :, 2] += rng.normal(loc=0.0, scale=0.5, size=(n_samples, 1))  # shipping_duration

    latent = X[:, -1, 0] * 1.5 + X.mean(axis=1)[:, 2] * 0.8
    y_demand = 100 + 15 * latent + rng.normal(0, 3, size=n_samples)

    # Dedicated delay-risk latent factor, embedded as a strong, low-noise
    # constant-per-sample offset across several features (shipping duration,
    # weather severity, port congestion) so it is cleanly recoverable by
    # averaging over the 20 time steps -- giving the classifier head a
    # genuinely learnable signal rather than one dominated by per-timestep noise.
    delay_risk = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
    X[:, :, 2] += delay_risk * 1.5  # shipping_duration
    X[:, :, 6] += delay_risk * 1.2  # weather_severity
    X[:, :, 7] += delay_risk * 1.0  # port_congestion

    delay_logit = 2.5 * delay_risk.ravel()
    delay_prob = 1 / (1 + np.exp(-delay_logit))
    y_delay = (rng.random(n_samples) < delay_prob).astype(np.float32)

    return X, y_demand.astype(np.float32), y_delay


def build_lstm_model(config: LSTMConfig):
    """Build the two-headed stacked LSTM Keras model.

    Args:
        config: LSTM hyperparameters.

    Returns:
        A compiled ``tf.keras.Model`` with inputs ``(timesteps, n_features)``
        and outputs ``demand_output`` (linear regression) and
        ``delay_output`` (sigmoid classification).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(config.input_timesteps, config.n_features), name="sequence_input")
    x = layers.LSTM(config.units_layer1, return_sequences=True)(inputs)
    x = layers.Dropout(config.dropout)(x)
    x = layers.LSTM(config.units_layer2, return_sequences=False)(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)

    demand_output = layers.Dense(1, activation="linear", name="demand_output")(x)
    delay_output = layers.Dense(1, activation="sigmoid", name="delay_output")(x)

    model = models.Model(inputs=inputs, outputs=[demand_output, delay_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={"demand_output": "mse", "delay_output": "binary_crossentropy"},
        metrics={"demand_output": ["mae"], "delay_output": ["accuracy"]},
    )
    return model


def train_lstm(
    config: LSTMConfig,
    n_train: int,
    n_val: int,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Train the LSTM model on synthetic sequence data.

    Args:
        config: LSTM hyperparameters.
        n_train: Number of training sequences to generate.
        n_val: Number of validation sequences to generate.
        seed: Random seed for data generation and model initialization.

    Returns:
        Tuple of ``(model, history_dict)`` where ``history_dict`` is the
        Keras training history's ``.history`` dictionary.
    """
    import tensorflow as tf

    tf.random.set_seed(seed)
    model = build_lstm_model(config)

    X_train, y_demand_train, y_delay_train = generate_synthetic_sequences(n_train, config, seed=seed)
    X_val, y_demand_val, y_delay_val = generate_synthetic_sequences(n_val, config, seed=seed + 1)

    # Standardize the demand regression target so its MSE loss magnitude is
    # comparable to the delay head's binary cross-entropy loss; otherwise the
    # much larger-magnitude demand loss dominates gradients through the
    # shared LSTM trunk and starves the delay head of useful signal. The
    # scaler is attached to the model so evaluate_lstm can invert it.
    demand_mean = float(y_demand_train.mean())
    demand_std = float(y_demand_train.std()) or 1.0
    y_demand_train_scaled = (y_demand_train - demand_mean) / demand_std
    y_demand_val_scaled = (y_demand_val - demand_mean) / demand_std

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        {"demand_output": y_demand_train_scaled, "delay_output": y_delay_train},
        validation_data=(
            X_val,
            {"demand_output": y_demand_val_scaled, "delay_output": y_delay_val},
        ),
        batch_size=config.batch_size,
        epochs=config.max_epochs,
        callbacks=[early_stopping],
        verbose=0,
    )
    model.demand_mean = demand_mean
    model.demand_std = demand_std
    return model, history.history


def evaluate_lstm(model: Any, config: LSTMConfig, n_test: int, seed: int = 123) -> Dict[str, float]:
    """Evaluate a trained LSTM model on freshly generated synthetic test data.

    Args:
        model: Trained Keras model returned by :func:`train_lstm`.
        config: LSTM hyperparameters (controls test sequence shape).
        n_test: Number of test sequences to generate.
        seed: Random seed for test-data generation.

    Returns:
        Dictionary with keys ``mape``, ``r2``, ``auc_roc``, ``f1``.
    """
    X_test, y_demand_test, y_delay_test = generate_synthetic_sequences(n_test, config, seed=seed)
    demand_pred, delay_pred = model.predict(X_test, verbose=0)
    demand_pred = demand_pred.ravel()
    delay_pred = delay_pred.ravel()

    demand_mean = getattr(model, "demand_mean", 0.0)
    demand_std = getattr(model, "demand_std", 1.0)
    demand_pred = demand_pred * demand_std + demand_mean

    mape = float(
        np.mean(np.abs((y_demand_test - demand_pred) / np.clip(np.abs(y_demand_test), 1e-6, None))) * 100
    )
    r2 = float(r2_score(y_demand_test, demand_pred))

    try:
        auc = float(roc_auc_score(y_delay_test, delay_pred))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(y_delay_test, (delay_pred > 0.5).astype(int)))

    return {"mape": mape, "r2": r2, "auc_roc": auc, "f1": f1}
