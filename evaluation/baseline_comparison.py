"""Baseline comparison across traceability approaches (Table VI).

Compares the integrated BCT-AI framework against a Traditional (centralized
database + barcode) system, a Blockchain-Only system, and an AI-Only system
on counterfeit detection rate, mean recall/localization time, and missed
detection rate.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

METHOD_ORDER = ["Traditional", "Blockchain-Only", "AI-Only", "BCT-AI"]


def build_baseline_comparison_table(config: Dict[str, Any]) -> pd.DataFrame:
    """Build the Table VI baseline comparison DataFrame from configuration.

    Args:
        config: Full project configuration (uses the ``baseline_comparison`` section).

    Returns:
        DataFrame with one row per method: ``method``, ``detection_rate_pct``,
        ``localization_time_hours``, ``missed_detection_pct``.
    """
    bc_cfg = config["baseline_comparison"]
    rows = []
    for method in METHOD_ORDER:
        entry = bc_cfg[method]
        rows.append(
            {
                "method": method,
                "detection_rate_pct": entry["detection_rate_pct"],
                "localization_time_hours": entry["localization_time_hours"],
                "missed_detection_pct": entry["missed_detection_pct"],
            }
        )
    return pd.DataFrame(rows)


def simulate_per_run_detection_rates(
    config: Dict[str, Any], n_runs: int, seed: int = 42
) -> pd.DataFrame:
    """Simulate per-run detection-rate samples for each baseline method.

    Adds small, method-appropriate Gaussian jitter around each method's
    Table VI point estimate so downstream plots/statistics have a per-run
    distribution to draw from, while keeping the across-run mean anchored
    to the paper's reported values.

    Args:
        config: Full project configuration.
        n_runs: Number of simulated runs per method.
        seed: Random seed for reproducibility.

    Returns:
        Long-format DataFrame with columns ``method``, ``run``, ``detection_rate_pct``.
    """
    bc_cfg = config["baseline_comparison"]
    rng = np.random.default_rng(seed)
    rows = []
    for method in METHOD_ORDER:
        mean_rate = bc_cfg[method]["detection_rate_pct"]
        std = max(0.5, mean_rate * 0.01)
        samples = np.clip(rng.normal(mean_rate, std, size=n_runs), 0, 100)
        for run_idx, sample in enumerate(samples):
            rows.append({"method": method, "run": run_idx, "detection_rate_pct": float(sample)})
    return pd.DataFrame(rows)


def compute_relative_improvement(config: Dict[str, Any]) -> Dict[str, float]:
    """Compute BCT-AI's relative improvement over each baseline method.

    Args:
        config: Full project configuration.

    Returns:
        Mapping of ``"{method}_detection_improvement_pct"`` /
        ``"{method}_time_reduction_factor"`` to their computed values.
    """
    bc_cfg = config["baseline_comparison"]
    bct_ai = bc_cfg["BCT-AI"]
    results: Dict[str, float] = {}
    for method in ["Traditional", "Blockchain-Only", "AI-Only"]:
        entry = bc_cfg[method]
        results[f"{method}_detection_improvement_pct"] = (
            bct_ai["detection_rate_pct"] - entry["detection_rate_pct"]
        )
        if entry["localization_time_hours"] > 0:
            results[f"{method}_time_reduction_factor"] = (
                entry["localization_time_hours"] / bct_ai["localization_time_hours"]
            )
    return results
