"""BCCA 2026 conference-reported reference values (historical; not computed).

This module preserves, for reproduction only, the code that up to release
1.0.x built the "Table VI baseline comparison" from configuration. Every
value it returns is a BCCA 2026 conference-reported figure read from
``config.yaml:bcca2026_reference_values``. None is computed from
transaction-level predictions, and none is reproduced by this repository.

Separated from the current evaluation in release 1.1.0:

* ``build_reference_table`` (formerly
  ``evaluation.baseline_comparison.build_baseline_comparison_table``)
  returns the configured table, now with a ``provenance`` column.
* ``simulate_per_run_detection_rates`` (formerly in the same module) adds
  Gaussian jitter around the configured values. Its output is NOT a set of
  measurements; it is kept unchanged only so earlier plots can be
  regenerated and audited.
* ``compute_improvement_ratios`` / ``compute_relative_improvement`` are
  arithmetic on the configured values.
* ``reference_component_metrics``, ``reference_statistical_tests``,
  ``reference_monte_carlo`` and ``reference_ai_only_detection_pct`` return
  the other conference-reported figures that releases up to 1.0.x stored as
  ``target_metrics``, ``statistical_validation`` t-values, Monte Carlo
  ``paper_reference`` values / ``real_world_range``, and the calibration
  diagnostic's 78.2% comparison line.

All outputs carry :data:`evaluation.result_guards.REFERENCE_MARKER`, so the
computed-results writers reject them (see ``evaluation/result_guards.py``).

Usage (prints the reference table; writes nothing)::

    python -m historical.bcca2026_reference
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from evaluation.result_guards import REFERENCE_CONFIG_KEY, REFERENCE_MARKER

METHOD_ORDER = ["Traditional", "Blockchain-Only", "AI-Only", "BCT-AI"]


def load_reference_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the conference-reported reference section of the configuration."""
    return config[REFERENCE_CONFIG_KEY]


def build_reference_table(config: Dict[str, Any]) -> pd.DataFrame:
    """Conference-reported Table VI as a DataFrame (one row per method).

    Columns: ``method``, ``detection_rate_pct``, ``localization_time_hours``,
    ``missed_detection_pct``, ``provenance``.
    """
    ref = load_reference_values(config)
    rows = []
    for method in METHOD_ORDER:
        entry = ref[method]
        rows.append(
            {
                "method": method,
                "detection_rate_pct": entry["detection_rate_pct"],
                "localization_time_hours": entry["localization_time_hours"],
                "missed_detection_pct": entry["missed_detection_pct"],
                "provenance": REFERENCE_MARKER,
            }
        )
    return pd.DataFrame(rows)


def simulate_per_run_detection_rates(
    config: Dict[str, Any], n_runs: int, seed: int = 42
) -> pd.DataFrame:
    """HISTORICAL. Jitter-generated samples around the reference values.

    Unchanged algorithm from release 1.0.x: adds Gaussian noise
    (sd = max(0.5, 1% of the value)) around each configured detection rate,
    so the across-run mean stays anchored to the conference-reported value.
    These samples are not measurements and must not be used in statistical
    tests or evidentiary plots.

    Returns:
        Long-format DataFrame: ``method``, ``run``, ``detection_rate_pct``,
        ``provenance``.
    """
    ref = load_reference_values(config)
    rng = np.random.default_rng(seed)
    rows = []
    for method in METHOD_ORDER:
        mean_rate = ref[method]["detection_rate_pct"]
        std = max(0.5, mean_rate * 0.01)
        samples = np.clip(rng.normal(mean_rate, std, size=n_runs), 0, 100)
        for run_idx, sample in enumerate(samples):
            rows.append({"method": method, "run": run_idx, "detection_rate_pct": float(sample),
                         "provenance": REFERENCE_MARKER})
    return pd.DataFrame(rows)


def compute_improvement_ratios(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ratios between conference-reported detection rates (arithmetic only).

    The conference text states "4.1x improvement over blockchain-only", but
    96.8/41.7 = 2.32x; 96.8/23.4 (Traditional) = 4.14x is the arithmetic match.
    """
    ref = load_reference_values(config)
    bct_ai = ref["BCT-AI"]["detection_rate_pct"]
    traditional = ref["Traditional"]["detection_rate_pct"]
    blockchain_only = ref["Blockchain-Only"]["detection_rate_pct"]
    ai_only = ref["AI-Only"]["detection_rate_pct"]
    return {
        "provenance": REFERENCE_MARKER,
        "vs_traditional_ratio": round(bct_ai / traditional, 2),
        "vs_blockchain_only_ratio": round(bct_ai / blockchain_only, 2),
        "vs_ai_only_ratio": round(bct_ai / ai_only, 2),
        "paper_stated_ratio": ref["paper_stated_improvement_ratio"],
        "paper_ratio_note": (
            "The conference's 4.1x figure matches BCT-AI vs Traditional "
            f"({bct_ai}/{traditional}={round(bct_ai / traditional, 2)}x), not vs "
            f"Blockchain-Only ({bct_ai}/{blockchain_only}={round(bct_ai / blockchain_only, 2)}x)."
        ),
    }


def compute_relative_improvement(config: Dict[str, Any]) -> Dict[str, Any]:
    """Differences and time ratios between conference-reported values."""
    ref = load_reference_values(config)
    bct_ai = ref["BCT-AI"]
    results: Dict[str, Any] = {"provenance": REFERENCE_MARKER}
    for method in ["Traditional", "Blockchain-Only", "AI-Only"]:
        entry = ref[method]
        results[f"{method}_detection_improvement_pct"] = (
            bct_ai["detection_rate_pct"] - entry["detection_rate_pct"]
        )
        if entry["localization_time_hours"] > 0:
            results[f"{method}_time_reduction_factor"] = (
                entry["localization_time_hours"] / bct_ai["localization_time_hours"]
            )
    return results


def reference_recall_localization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Conference-reported recall-localization figures (2.8 min, 100%)."""
    ref = load_reference_values(config)
    return {
        "provenance": REFERENCE_MARKER,
        "time_minutes": ref["recall_localization"]["paper_stated_time_minutes"],
        "accuracy_pct": ref["recall_localization"]["paper_stated_accuracy_pct"],
        "traditional_time_hours": ref["Traditional"]["localization_time_hours"],
    }


def _tagged(values: Any) -> Dict[str, Any]:
    return {"provenance": REFERENCE_MARKER, "values": values}


def reference_component_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Conference-reported LSTM, CNN and Isolation Forest metrics.

    Formerly ``lstm/cnn/isolation_forest.target_metrics`` in the config and
    returned as ``target_metrics`` by the component pipelines (release 1.0.x).
    """
    ref = load_reference_values(config)
    return _tagged({
        "lstm": ref["lstm_metrics"],
        "cnn": ref["cnn_metrics"],
        "isolation_forest": ref["isolation_forest_metrics"],
    })


def reference_statistical_tests(config: Dict[str, Any]) -> Dict[str, Any]:
    """Conference-reported paired t-tests (t(9) = 6.42 and 7.11)."""
    return _tagged(load_reference_values(config)["statistical_tests"])


def reference_monte_carlo(config: Dict[str, Any]) -> Dict[str, Any]:
    """Conference-reported Monte Carlo degradation figures and "real-world range".

    Formerly returned as ``paper_reference`` sub-fields and
    ``real_world_range`` by ``run_monte_carlo_pipeline`` (release 1.0.x).
    """
    return _tagged(load_reference_values(config)["monte_carlo"])


def reference_ai_only_detection_pct(config: Dict[str, Any]) -> Dict[str, Any]:
    """Conference-reported AI-Only detection (78.2%).

    The original PTS calibration diagnostic (release 1.0.x) compared each
    variant with this value; its archived outputs keep that comparison.
    """
    return _tagged(load_reference_values(config)["AI-Only"]["detection_rate_pct"])


if __name__ == "__main__":  # pragma: no cover - display helper
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import load_config

    print("BCCA 2026 conference-reported reference values (NOT computed by this repository):")
    print(build_reference_table(load_config()).drop(columns="provenance").to_string(index=False))
