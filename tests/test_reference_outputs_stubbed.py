"""Output-level checks with stub models (no model training).

Each test runs a real pipeline function with its training/inference
primitives replaced by stubs that return distinctive computed values, then
checks the OUTPUT (not the source code) for conference-reported reference
material in two ways:

* marker check: ``find_reference_values`` finds no provenance marker or
  reference-section key;
* value check: no numeric leaf equal to a non-trivial value in
  ``config.yaml:bcca2026_reference_values`` appears, and no key named
  ``target_metrics``, ``paper_reference``, ``real_world_range`` or
  ``ai_only_baseline_pct`` is present.

The value check is what catches a bare reference number copied without its
marker, which the runtime marker guard does not detect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from ai_modules import cnn_verification as cnn_mod
from ai_modules import isolation_forest_detector as if_mod
from ai_modules import lstm_forecasting as lstm_mod
from evaluation import run_simulation as rs
from evaluation.result_guards import REFERENCE_CONFIG_KEY, find_reference_values
from utils import load_config

FORBIDDEN_KEYS = {"target_metrics", "paper_reference", "real_world_range", "ai_only_baseline_pct"}
# Values too common to be evidence of leakage (counts, bounds, df, alpha).
TRIVIAL = {0.0, 1.0, 100.0, 9.0, 0.01}


# --------------------------------------------------------------------------- helpers
def _numeric_leaves(obj: Any, path: str = "$") -> Iterator[Tuple[str, float]]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float, np.integer, np.floating)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numeric_leaves(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _numeric_leaves(v, f"{path}[{i}]")
    elif isinstance(obj, pd.DataFrame):
        yield from _numeric_leaves(obj.to_dict(orient="list"), path)
    elif isinstance(obj, np.ndarray):
        yield from _numeric_leaves(obj.tolist(), path)


def _keys(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _keys(v)
    elif isinstance(obj, pd.DataFrame):
        yield from (str(c) for c in obj.columns)


def reference_values(config) -> Set[float]:
    vals = {round(v, 6) for _, v in _numeric_leaves(config[REFERENCE_CONFIG_KEY])}
    return {v for v in vals if v not in TRIVIAL}


def assert_clean(obj: Any, config, allowed_paths: Set[str] = frozenset()) -> None:
    assert not find_reference_values(obj), find_reference_values(obj)
    bad_keys = FORBIDDEN_KEYS & set(_keys(obj))
    assert not bad_keys, f"reference keys in output: {bad_keys}"
    ref = reference_values(config)
    hits = [(p, v) for p, v in _numeric_leaves(obj) if round(v, 6) in ref and p not in allowed_paths]
    assert not hits, f"reference values in output: {hits[:10]}"


def _monte_carlo_allowed_paths(out: dict, epsilon: float = None) -> Set[str]:
    """Paths whose value cannot be evidence of a copied reference number.

    With a stub model and a small trial count, a detection mean is a fraction
    k/n of the ``n_samples`` trials (e.g. 4/20 = 0.2, 1/20 = 0.05), which can
    coincide with a coarse reference value by chance. Such a leaf is allowed
    only if it is exactly k/n for an integer k; the FGSM ``epsilon`` echoed
    from the function argument is allowed only if it equals that argument.
    """
    n = out["n_samples"]
    allowed = {p for p, v in _numeric_leaves(out)
               if p.endswith("_mean") and abs(v * n - round(v * n)) < 1e-9}
    if epsilon is not None:
        assert out["epsilon"] == epsilon
        allowed.add("$.epsilon")
    return allowed


# --------------------------------------------------------------------------- stubs
class StubIF:
    """Deterministic stand-in for a fitted IsolationForest (no training)."""

    def score_samples(self, X):
        return -np.abs(X[:, 0] - 5.0) * 0.137


LSTM_METRICS = {"mape": 8.3141, "r2": 0.9613, "auc_roc": 0.8123, "f1": 0.7713}
CNN_METRICS = {"weighted_avg": {"precision": 0.9711, "recall": 0.9713, "f1": 0.9717},
               "Authentic": {"precision": 0.9811, "recall": 0.9813, "f1": 0.9817}}
IF_METRICS = {"overall": {"detection": 0.8931, "fpr": 0.0213, "latency_min": 3.17},
              "Gradual Drift": {"detection": 0.8711, "fpr": 0.0311, "latency_min": 4.13}}


@pytest.fixture()
def config():
    return load_config()


@pytest.fixture()
def stub_training(monkeypatch):
    monkeypatch.setattr(lstm_mod, "train_lstm", lambda *a, **k: (object(), None))
    monkeypatch.setattr(lstm_mod, "evaluate_lstm", lambda *a, **k: dict(LSTM_METRICS))
    monkeypatch.setattr(cnn_mod, "train_cnn", lambda *a, **k: (object(), None))
    monkeypatch.setattr(cnn_mod, "evaluate_cnn", lambda *a, **k: json.loads(json.dumps(CNN_METRICS)))
    monkeypatch.setattr(cnn_mod, "quantize_and_benchmark",
                        lambda *a, **k: {"model_size_mb": 3.217, "inference_latency_ms": 1.913})
    monkeypatch.setattr(if_mod, "train_isolation_forest", lambda *a, **k: StubIF())
    monkeypatch.setattr(if_mod, "calibrate_score_normalization", lambda *a, **k: (0.0123, 0.4567))
    monkeypatch.setattr(if_mod, "evaluate_all_anomaly_types",
                        lambda *a, **k: json.loads(json.dumps(IF_METRICS)))


def _stub_runs(n=10):
    rng = np.random.default_rng(3)
    return [{"seed": 42 + i, "n_transactions": 1234, "n_counterfeit": 57, "n_anomalies": 211,
             "bct_ai_counterfeit_detection": 0.7141 + 0.0071 * rng.standard_normal(),
             "ai_only_counterfeit_detection": 0.6833 + 0.0063 * rng.standard_normal(),
             "bct_ai_recall_efficiency": 0.5521 + 0.0057 * rng.standard_normal(),
             "ai_only_recall_efficiency": 0.4419 + 0.0051 * rng.standard_normal()} for i in range(n)]


def _stub_cnn_mc(cnn_results, config, n_samples=1000, **kw):
    return {"n_samples": 37, "baseline_detection_mean": 0.8713, "degraded_detection_mean": 0.8117}


def _stub_adv_mc(cnn_results, config, n_samples=1000, **kw):
    return {"n_samples": 37, "epsilon": 0.0313, "baseline_detection_mean": 0.8713,
            "adversarial_detection_mean": 0.6119}


def _fast_sensor_drift(monkeypatch):
    real_drift = rs.simulate_sensor_drift_monte_carlo
    monkeypatch.setattr(rs, "simulate_sensor_drift_monte_carlo",
                        lambda r, c, n_samples=1000, **k: real_drift(r, c, n_samples=20))


# --------------------------------------------------------------------------- component pipelines
@pytest.mark.parametrize("runner,metric_key", [
    (rs.run_lstm_pipeline, "mape"),
    (rs.run_cnn_pipeline, "weighted_avg"),
    (rs.run_isolation_forest_pipeline, "overall"),
], ids=["lstm", "cnn", "isolation_forest"])
def test_component_pipeline_outputs_are_computed_only(runner, metric_key, config, stub_training) -> None:
    out = {k: v for k, v in runner(config).items() if k != "_artifacts"}
    assert metric_key in out["metrics"]
    assert set(out) <= {"metrics", "quantization"}
    assert_clean(out, config)


# --------------------------------------------------------------------------- Monte Carlo
def test_sensor_drift_monte_carlo_output_with_stub_model(config) -> None:
    if_results = {"_artifacts": {"model": StubIF(), "model_config": if_mod.IsolationForestConfig()}}
    out = rs.simulate_sensor_drift_monte_carlo(if_results, config, n_samples=20)
    assert {"baseline_detection_mean", "degraded_detection_mean"} <= set(out)
    assert_clean(out, config)


def test_monte_carlo_pipeline_output(config, monkeypatch) -> None:
    monkeypatch.setattr(rs, "simulate_mobile_image_variability_monte_carlo", _stub_cnn_mc)
    monkeypatch.setattr(rs, "simulate_adversarial_gan_monte_carlo", _stub_adv_mc)
    _fast_sensor_drift(monkeypatch)
    if_results = {"_artifacts": {"model": StubIF(), "model_config": if_mod.IsolationForestConfig()}}
    out = rs.run_monte_carlo_pipeline(config, {"_artifacts": {}}, if_results)
    assert set(out) == {"sensor_drift", "mobile_image_variability", "adversarial_gan_counterfeits"}
    assert_clean(out, config)


def _stub_cnn_results():
    torch = pytest.importorskip("torch")

    class StubCNN(torch.nn.Module):
        """Untrained fixed-weight stand-in returning (auth_logit, tamper_logits)."""

        def __init__(self, n_classes):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.w = torch.nn.Parameter(torch.randn(3, n_classes, generator=g))

        def forward(self, x):
            logits = x.mean(dim=(2, 3)) @ self.w
            return logits[:, :1], logits

    cfg = cnn_mod.CNNConfig(image_size=16, pretrained=False)
    return {"_artifacts": {"model": StubCNN(len(cfg.class_names)), "model_config": cfg}}


def test_mobile_image_monte_carlo_output_with_stub_model(config) -> None:
    out = rs.simulate_mobile_image_variability_monte_carlo(_stub_cnn_results(), config, n_samples=20)
    assert_clean(out, config, _monte_carlo_allowed_paths(out))


def test_adversarial_monte_carlo_output_with_stub_model(config) -> None:
    out = rs.simulate_adversarial_gan_monte_carlo(_stub_cnn_results(), config, n_samples=20, epsilon=0.05)
    assert_clean(out, config, _monte_carlo_allowed_paths(out, epsilon=0.05))


# --------------------------------------------------------------------------- statistical validation
def test_statistical_validation_output(config) -> None:
    out = rs.compute_statistical_validation_from_runs(_stub_runs())
    assert_clean(out, config)
    assert "6.42" not in out["note"] and "7.11" not in out["note"]


# --------------------------------------------------------------------------- full pipeline + writer
def test_full_pipeline_written_output(config, stub_training, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rs, "run_transaction_simulation",
                        lambda c: {"validation_stats": {"n_transactions": 1234, "n_anomalies": 211}})
    monkeypatch.setattr(rs, "run_all_simulation_runs", lambda c: _stub_runs())
    monkeypatch.setattr(rs, "run_blockchain_benchmark",
                        lambda c: {"table_iii": pd.DataFrame({"tps_target": [513], "avg_latency_ms": [191.3]}),
                                   "ai_inference_latency_impact": {"overhead_ms": 83.1}})
    monkeypatch.setattr(rs, "run_pts_pipeline", lambda c: {"sample_scores": {"A": 0.6131}})
    monkeypatch.setattr(rs, "simulate_mobile_image_variability_monte_carlo", _stub_cnn_mc)
    monkeypatch.setattr(rs, "simulate_adversarial_gan_monte_carlo", _stub_adv_mc)
    _fast_sensor_drift(monkeypatch)

    results = rs.run_full_pipeline(config)
    assert not {"baseline_comparison", "improvement_ratios"} & set(results)
    assert_clean(results, config)

    config["paths"]["results_dir"] = str(tmp_path)
    config["paths"]["simulation_results_csv"] = str(tmp_path / "sim.csv")
    config["paths"]["performance_metrics_json"] = str(tmp_path / "metrics.json")
    rs.write_results(results, config)
    assert_clean(json.loads((tmp_path / "metrics.json").read_text()), config)
    assert_clean(pd.read_csv(tmp_path / "sim.csv"), config)


# --------------------------------------------------------------------------- calibration diagnostic
def test_calibration_diagnostic_payload_and_no_overwrite(config, tmp_path: Path) -> None:
    from pts import calibration_diagnostic as cd

    results = {"baseline": {"detection_mean": 0.7141, "detection_std": 0.0141, "recall_mean": 0.5521,
                            "recall_std": 0.0057, "ai_only_detection_mean": 0.6833,
                            "per_seed_detection": [0.7131, 0.7151], "per_seed_recall": [0.5511, 0.5531]}}
    payload = cd.build_payload(results)
    assert "ai_only_baseline_pct" not in payload
    assert not hasattr(cd, "AI_ONLY_BASELINE_PCT")
    assert_clean(payload, config)

    archived = tmp_path / "pts_calibration_diagnostic.json"
    archived.write_text('{"archived": true}')
    new_path = cd.write_payload(payload, archived)
    assert new_path != archived
    assert json.loads(archived.read_text()) == {"archived": True}
    assert json.loads(new_path.read_text())["results"]["baseline"]["detection_mean"] == 0.7141


# --------------------------------------------------------------------------- self-test of the value check
def test_value_check_catches_an_untagged_reference_number(config) -> None:
    leaked = {"metrics": {"detection": config[REFERENCE_CONFIG_KEY]["BCT-AI"]["detection_rate_pct"]}}
    assert not find_reference_values(leaked)  # the marker guard alone does not see it
    with pytest.raises(AssertionError):
        assert_clean(leaked, config)


def test_value_check_catches_a_restored_target_metrics_key(config) -> None:
    leaked = {"metrics": {"mape": 8.3141}, "target_metrics": {"mape": 7.3}}
    with pytest.raises(AssertionError):
        assert_clean(leaked, config)
