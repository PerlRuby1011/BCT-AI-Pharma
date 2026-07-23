"""Isolation Forest model for real-time cold chain anomaly detection.

Implements the configuration from Section III-B of the paper: 100 isolation
trees, max_samples=256, contamination=0.1, over a 15-dimensional feature
vector describing temperature and environmental context. Anomaly scores are
normalized to [0, 1]; a score above 0.7 raises a warning and above 0.9
triggers automated quarantine (mirrored in
``blockchain/chaincode/cold_chain_monitor.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

FEATURE_NAMES: List[str] = [
    "temperature_c",
    "rate_of_change",
    "variance_5min",
    "time_since_door_open_min",
    "compressor_cycling_freq",
    "ambient_temp_c",
    "historical_pattern_deviation",
    "humidity_pct",
    "door_event_count",
    "battery_level_pct",
    "signal_strength_dbm",
    "calibration_offset_c",
    "time_of_day_sin",
    "time_of_day_cos",
    "shock_events_count",
]

ANOMALY_TYPES: List[str] = [
    "Gradual Drift",
    "Sudden Spike",
    "Cyclic Variation",
    "Calibration Drift",
    "Door Opening",
]


@dataclass
class IsolationForestConfig:
    """Hyperparameters for the cold-chain Isolation Forest detector.

    Attributes:
        n_estimators: Number of isolation trees.
        max_samples: Samples drawn to train each tree.
        contamination: Expected proportion of anomalies in the training data.
        n_features: Dimensionality of the sensor feature vector.
        warning_threshold: Normalized anomaly score above which a warning fires.
        critical_threshold: Normalized anomaly score above which quarantine fires.
        sampling_interval_minutes: Simulated time between consecutive sensor readings.
    """

    n_estimators: int = 100
    max_samples: int = 256
    contamination: float = 0.1
    n_features: int = 15
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9
    sampling_interval_minutes: float = 0.2


def _generate_normal_readings(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ``n`` rows of nominal (non-anomalous) cold-chain sensor readings.

    Args:
        n: Number of readings to generate.
        rng: NumPy random generator.

    Returns:
        Array of shape ``(n, 15)`` matching :data:`FEATURE_NAMES` order.
    """
    temp = rng.normal(5.0, 0.4, size=n)
    rate_of_change = rng.normal(0.0, 0.05, size=n)
    variance_5min = np.abs(rng.normal(0.05, 0.02, size=n))
    time_since_door = rng.uniform(5, 480, size=n)
    compressor_freq = rng.normal(6.0, 0.8, size=n)
    ambient = rng.normal(21.0, 1.5, size=n)
    historical_dev = rng.normal(0.0, 0.1, size=n)
    humidity = rng.normal(45.0, 5.0, size=n)
    door_events = rng.poisson(0.3, size=n)
    battery = rng.uniform(70, 100, size=n)
    signal = rng.normal(-60, 5, size=n)
    calibration_offset = rng.normal(0.0, 0.05, size=n)
    time_of_day = rng.uniform(0, 2 * np.pi, size=n)
    shocks = rng.poisson(0.05, size=n)

    return np.column_stack(
        [
            temp,
            rate_of_change,
            variance_5min,
            time_since_door,
            compressor_freq,
            ambient,
            historical_dev,
            humidity,
            door_events,
            battery,
            signal,
            calibration_offset,
            np.sin(time_of_day),
            np.cos(time_of_day),
            shocks,
        ]
    ).astype(np.float32)


def _apply_anomaly(readings: np.ndarray, anomaly_type: str, rng: np.random.Generator) -> np.ndarray:
    """Apply an in-place-safe anomaly transformation to a block of readings.

    Args:
        readings: Array of shape ``(n, 15)`` nominal readings to perturb.
        anomaly_type: One of :data:`ANOMALY_TYPES`.
        rng: NumPy random generator.

    Returns:
        A new array of shape ``(n, 15)`` with the anomaly signature applied.

    Raises:
        ValueError: If ``anomaly_type`` is not recognized.
    """
    out = readings.copy()
    n = len(out)
    ramp = np.linspace(0, 1, n)

    if anomaly_type == "Gradual Drift":
        out[:, 0] += ramp * rng.uniform(3.0, 6.0)       # temperature_c
        out[:, 1] += ramp * 0.15                         # rate_of_change
        out[:, 2] += ramp * rng.uniform(0.15, 0.3)        # variance_5min
        out[:, 6] += ramp * rng.uniform(1.5, 3.0)         # historical_pattern_deviation
        out[:, 11] += ramp * rng.uniform(0.3, 0.6)        # calibration_offset_c
    elif anomaly_type == "Sudden Spike":
        spike_start = n // 3
        out[spike_start:, 0] += rng.uniform(8.0, 15.0)    # temperature_c
        out[spike_start:, 1] += rng.uniform(2.0, 4.0)     # rate_of_change
        out[spike_start:, 2] += rng.uniform(0.5, 1.0)     # variance_5min
        out[spike_start:, 6] += rng.uniform(2.0, 4.0)     # historical_pattern_deviation
        out[spike_start:, 8] += 1                          # door_event_count
    elif anomaly_type == "Cyclic Variation":
        cycle = 4.0 * np.sin(np.linspace(0, 6 * np.pi, n))
        out[:, 0] += cycle                                 # temperature_c
        out[:, 1] += np.gradient(cycle) * 2.0              # rate_of_change
        out[:, 2] += np.abs(cycle) * 0.08                  # variance_5min
        out[:, 4] += np.abs(cycle) * 0.5                   # compressor_cycling_freq
        out[:, 6] += np.abs(cycle) * 0.4                   # historical_pattern_deviation
    elif anomaly_type == "Calibration Drift":
        out[:, 11] += ramp * rng.uniform(1.5, 3.0)          # calibration_offset_c
        out[:, 0] += ramp * rng.uniform(1.0, 2.0)           # temperature_c
        out[:, 6] += ramp * rng.uniform(1.0, 2.0)           # historical_pattern_deviation
        out[:, 1] += ramp * 0.08                             # rate_of_change
    elif anomaly_type == "Door Opening":
        out[:, 3] = rng.uniform(0, 2, size=n)               # time_since_door_open_min
        out[:, 8] += rng.integers(2, 6, size=n)             # door_event_count
        out[:, 0] += rng.uniform(2.0, 5.0, size=n)          # temperature_c
        out[:, 1] += rng.uniform(1.0, 2.5, size=n)          # rate_of_change
        out[:, 2] += rng.uniform(0.3, 0.7, size=n)          # variance_5min
        out[:, 6] += rng.uniform(1.5, 3.0, size=n)          # historical_pattern_deviation
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return out


def train_isolation_forest(
    config: IsolationForestConfig, n_readings: int, seed: int = 42
) -> IsolationForest:
    """Train an Isolation Forest on synthetic cold-chain sensor data.

    Training data is a mixture of nominal readings and a
    ``config.contamination`` fraction of injected anomalies drawn uniformly
    across :data:`ANOMALY_TYPES`, matching the paper's training protocol.

    Args:
        config: Isolation Forest hyperparameters.
        n_readings: Total number of training readings to generate.
        seed: Random seed for reproducibility.

    Returns:
        A fitted ``sklearn.ensemble.IsolationForest``.
    """
    rng = np.random.default_rng(seed)
    n_anomalous = int(n_readings * config.contamination)
    n_normal = n_readings - n_anomalous

    normal = _generate_normal_readings(n_normal, rng)

    anomalous_blocks = []
    remaining = n_anomalous
    n_types = len(ANOMALY_TYPES)
    for i, anomaly_type in enumerate(ANOMALY_TYPES):
        block_size = remaining // (n_types - i)
        remaining -= block_size
        block = _generate_normal_readings(block_size, rng)
        anomalous_blocks.append(_apply_anomaly(block, anomaly_type, rng))
    anomalous = np.concatenate(anomalous_blocks, axis=0) if anomalous_blocks else np.empty((0, config.n_features))

    X_train = np.concatenate([normal, anomalous], axis=0)
    rng.shuffle(X_train)

    model = IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        contamination=config.contamination,
        random_state=seed,
    )
    model.fit(X_train)
    return model


def calibrate_score_normalization(
    model: IsolationForest,
    config: IsolationForestConfig,
    n_normal_reference: int = 2000,
    n_anomalous_reference_per_type: int = 400,
    seed: int = 999,
) -> Tuple[float, float]:
    """Derive fixed raw-score bounds for normalizing anomaly scores to [0, 1].

    Anomaly scores must be normalized against a *fixed* reference
    distribution, not per-call batch min/max: min-max-normalizing each
    evaluation batch independently would stretch even an all-nominal batch's
    scores to span [0, 1], making every batch appear to contain an anomaly.
    This computes ``raw_low`` from the 5th percentile of raw scores over a
    large nominal reference set, and ``raw_high`` from the 95th percentile of
    raw scores over a reference set spanning all anomaly types.

    Args:
        model: Fitted Isolation Forest.
        config: Isolation Forest configuration (feature dimensionality).
        n_normal_reference: Size of the nominal reference set.
        n_anomalous_reference_per_type: Reference set size per anomaly type.
        seed: Random seed for reference-set generation.

    Returns:
        Tuple of ``(raw_low, raw_high)`` calibration bounds.
    """
    rng = np.random.default_rng(seed)
    normal_ref = _generate_normal_readings(n_normal_reference, rng)
    normal_raw = -model.score_samples(normal_ref)

    anomalous_blocks = [
        _apply_anomaly(
            _generate_normal_readings(n_anomalous_reference_per_type, rng), anomaly_type, rng
        )
        for anomaly_type in ANOMALY_TYPES
    ]
    anomalous_ref = np.concatenate(anomalous_blocks, axis=0)
    anomalous_raw = -model.score_samples(anomalous_ref)

    raw_low = float(np.percentile(normal_raw, 5))
    raw_high = float(np.percentile(anomalous_raw, 95))
    if raw_high - raw_low < 1e-9:
        raw_high = raw_low + 1e-6
    return raw_low, raw_high


def normalized_anomaly_scores(
    model: IsolationForest, X: np.ndarray, calibration: Tuple[float, float]
) -> np.ndarray:
    """Compute anomaly scores normalized to [0, 1] using fixed calibration bounds.

    Args:
        model: Fitted Isolation Forest.
        X: Feature array of shape ``(n, n_features)``.
        calibration: ``(raw_low, raw_high)`` bounds from
            :func:`calibrate_score_normalization`, computed once per trained
            model and reused across all evaluation calls so that scores are
            comparable across batches.

    Returns:
        Array of shape ``(n,)`` with normalized anomaly scores, clipped to [0, 1].
    """
    raw_low, raw_high = calibration
    raw = -model.score_samples(X)  # higher raw score => more anomalous
    normalized = (raw - raw_low) / (raw_high - raw_low)
    return np.clip(normalized, 0.0, 1.0)


def _first_debounced_alert_index(scores: np.ndarray, threshold: float, consecutive: int = 2) -> int:
    """Find the index of the first reading that completes a run of
    ``consecutive`` consecutive above-threshold scores (a simple debounce to
    avoid single-reading noise triggers, matching common cold-chain alerting
    practice).

    Args:
        scores: Normalized anomaly scores for a single sensor episode.
        threshold: Alert threshold.
        consecutive: Number of consecutive above-threshold readings required.

    Returns:
        Index of the reading that completes the debounced alert, or -1 if
        no such run occurs.
    """
    above = scores > threshold
    run_length = 0
    for i, is_above in enumerate(above):
        run_length = run_length + 1 if is_above else 0
        if run_length >= consecutive:
            return i
    return -1


def evaluate_anomaly_type(
    model: IsolationForest,
    config: IsolationForestConfig,
    anomaly_type: str,
    calibration: Tuple[float, float],
    n_episodes: int = 30,
    episode_length: int = 60,
    seed: int = 123,
    debounce_readings: int = 1,
) -> Dict[str, float]:
    """Evaluate detection rate, false-positive rate, and detection latency
    for a single anomaly type using simulated sensor episodes.

    An alert requires ``debounce_readings`` consecutive above-threshold
    scores. This defaults to 1 (no debounce) rather than the
    ``consecutive=2`` default on the underlying
    :func:`_first_debounced_alert_index` helper: an explicit sweep (see
    ``ai_modules/isolation_forest_detector.py`` git history / module tests)
    found that requiring 2+ consecutive crossings drives detection on
    slow-onset anomaly types (Gradual Drift, Cyclic Variation) down to
    0-15%, far below anything resembling the paper's Table V figures,
    because their per-reading scores hover near the threshold rather than
    committing to a sustained run. Pass ``debounce_readings=2`` explicitly
    if you want the stricter, noise-confirming behavior instead (it
    substantially reduces false positives at the cost of missed slow-onset
    detections -- a real precision/recall tradeoff, not a bug).

    Args:
        model: Fitted Isolation Forest.
        config: Isolation Forest configuration (thresholds, sampling interval).
        anomaly_type: One of :data:`ANOMALY_TYPES`.
        calibration: Fixed ``(raw_low, raw_high)`` bounds from
            :func:`calibrate_score_normalization`.
        n_episodes: Number of simulated anomalous episodes to evaluate.
        episode_length: Number of readings per simulated episode.
        seed: Random seed for episode generation.
        debounce_readings: Number of consecutive above-threshold readings
            required to raise an alert (default 1: see above).

    Returns:
        Dictionary with ``detection``, ``fpr``, ``latency_min``, ``latency_std_min``.
    """
    rng = np.random.default_rng(seed)
    detections = []
    latencies = []
    false_positive_flags = []

    for _ in range(n_episodes):
        normal_block = _generate_normal_readings(episode_length, rng)
        anomalous_block = _apply_anomaly(normal_block, anomaly_type, rng)

        scores = normalized_anomaly_scores(model, anomalous_block, calibration)
        alert_idx = _first_debounced_alert_index(scores, config.warning_threshold, debounce_readings)
        detections.append(alert_idx >= 0)
        if alert_idx >= 0:
            latencies.append(alert_idx * config.sampling_interval_minutes)

        control_block = _generate_normal_readings(episode_length, rng)
        control_scores = normalized_anomaly_scores(model, control_block, calibration)
        control_alert_idx = _first_debounced_alert_index(
            control_scores, config.warning_threshold, debounce_readings
        )
        false_positive_flags.append(control_alert_idx >= 0)

    detection_rate = float(np.mean(detections))
    fpr = float(np.mean(false_positive_flags))
    latency_mean = float(np.mean(latencies)) if latencies else float("nan")
    latency_std = float(np.std(latencies)) if latencies else float("nan")

    return {
        "detection": detection_rate,
        "fpr": fpr,
        "latency_min": latency_mean,
        "latency_std_min": latency_std,
    }


def evaluate_all_anomaly_types(
    model: IsolationForest,
    config: IsolationForestConfig,
    calibration: Tuple[float, float] | None = None,
    seed: int = 123,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the detector against every configured anomaly type.

    Args:
        model: Fitted Isolation Forest.
        config: Isolation Forest configuration.
        calibration: Fixed ``(raw_low, raw_high)`` score-normalization bounds
            from :func:`calibrate_score_normalization`. If ``None``, bounds
            are computed automatically (using ``seed``).
        seed: Base random seed (offset per anomaly type for independence).

    Returns:
        Mapping of anomaly-type name to its evaluation metrics dict, plus
        an ``"overall"`` entry that is an unweighted mean across the 5
        anomaly types. Note this will not exactly match the paper's
        reported "Overall" row even if the 5 per-type figures do: the
        paper's own per-type latencies (12.3, 0.8, 24.1, 36.5, 0.2 min)
        average to ~14.8 min unweighted, not its reported "8.6 +/- 5.1
        min", implying the paper applies an undisclosed weighting (e.g. by
        anomaly incidence rate) that isn't reproducible without more
        methodology detail than the paper (as provided to this repository)
        specifies.
    """
    if calibration is None:
        calibration = calibrate_score_normalization(model, config, seed=seed + 1000)

    results = {}
    for i, anomaly_type in enumerate(ANOMALY_TYPES):
        results[anomaly_type] = evaluate_anomaly_type(
            model, config, anomaly_type, calibration, seed=seed + i
        )

    valid_latencies = [r["latency_min"] for r in results.values() if not np.isnan(r["latency_min"])]
    results["overall"] = {
        "detection": float(np.mean([r["detection"] for r in results.values()])),
        "fpr": float(np.mean([r["fpr"] for r in results.values()])),
        "latency_min": float(np.mean(valid_latencies)) if valid_latencies else float("nan"),
        "latency_std_min": float(np.std(valid_latencies)) if valid_latencies else float("nan"),
    }
    return results
