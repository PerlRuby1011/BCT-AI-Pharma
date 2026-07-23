"""Statistical validation across independent simulation runs (Section IV-C).

Runs paired t-tests between the BCT-AI framework and an AI-Only baseline
across ``n_runs`` genuinely independent simulation runs (see
:func:`evaluation.run_simulation.run_single_simulation_run`, which
regenerates a full labeled transaction dataset per seed and drives each
metric through the real PTS formula and smart-contract quarantine logic),
and computes 95% confidence intervals for headline metrics.

The resulting t-statistics are computed from real per-run variance, not
reverse-engineered to match a target value -- they will differ from the
paper's reported ``t(9) = 6.42`` (counterfeit detection) / ``t(9) = 7.11``
(recall efficiency), which were measured on the paper's full-scale
(2.3M-transaction) simulation. That is expected: this repository runs a
laptop-scale equivalent (10 x 230,000 transactions) and reports whatever
statistic genuinely emerges from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


@dataclass
class TTestResult:
    """Result of a paired t-test between two per-run metric series.

    Attributes:
        t_statistic: Computed t-statistic.
        p_value: Two-sided p-value.
        degrees_of_freedom: Degrees of freedom (``n_runs - 1``).
        mean_difference: Mean of the paired differences.
        ci_95: 95% confidence interval for the mean difference.
    """

    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    mean_difference: float
    ci_95: Tuple[float, float]


def run_paired_ttest(baseline: np.ndarray, treatment: np.ndarray) -> TTestResult:
    """Run a paired (dependent-samples) t-test between baseline and treatment runs.

    Args:
        baseline: Per-run metric values for the baseline system.
        treatment: Per-run metric values for the BCT-AI framework (same run order).

    Returns:
        A :class:`TTestResult`.
    """
    t_stat, p_value = stats.ttest_rel(treatment, baseline)
    diffs = treatment - baseline
    n = len(diffs)
    df = n - 1
    sem = diffs.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df, loc=diffs.mean(), scale=sem)
    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        degrees_of_freedom=df,
        mean_difference=float(diffs.mean()),
        ci_95=(float(ci[0]), float(ci[1])),
    )


def confidence_interval_95(values: np.ndarray) -> Tuple[float, float]:
    """Compute a 95% confidence interval for the mean of a metric series.

    Args:
        values: Per-run metric values.

    Returns:
        Tuple of ``(lower, upper)`` bounds.
    """
    n = len(values)
    if n < 2:
        v = float(values[0]) if n == 1 else float("nan")
        return (v, v)
    sem = values.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=values.mean(), scale=sem)
    return (float(ci[0]), float(ci[1]))


def compute_statistical_validation_from_runs(
    per_run_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute paired t-tests and confidence intervals from genuine per-run results.

    Args:
        per_run_results: List of dictionaries, one per independent
            simulation run, each produced by
            :func:`evaluation.run_simulation.run_single_simulation_run` and
            containing ``bct_ai_counterfeit_detection``,
            ``ai_only_counterfeit_detection``, ``bct_ai_recall_efficiency``,
            and ``ai_only_recall_efficiency``.

    Returns:
        Dictionary with per-run metric arrays, paired t-test results for
        counterfeit detection and recall efficiency, and 95% confidence
        intervals for the BCT-AI headline metrics.
    """
    n_runs = len(per_run_results)
    bct_ai_counterfeit = np.array([r["bct_ai_counterfeit_detection"] for r in per_run_results])
    ai_only_counterfeit = np.array([r["ai_only_counterfeit_detection"] for r in per_run_results])
    bct_ai_recall = np.array([r["bct_ai_recall_efficiency"] for r in per_run_results])
    ai_only_recall = np.array([r["ai_only_recall_efficiency"] for r in per_run_results])

    counterfeit_ttest = run_paired_ttest(ai_only_counterfeit, bct_ai_counterfeit)
    recall_ttest = run_paired_ttest(ai_only_recall, bct_ai_recall)

    return {
        "n_runs": n_runs,
        "per_run": {
            "bct_ai_counterfeit_detection": bct_ai_counterfeit.tolist(),
            "baseline_counterfeit_detection": ai_only_counterfeit.tolist(),
            "bct_ai_recall_efficiency": bct_ai_recall.tolist(),
            "baseline_recall_efficiency": ai_only_recall.tolist(),
        },
        "counterfeit_detection_ttest": vars(counterfeit_ttest),
        "recall_efficiency_ttest": vars(recall_ttest),
        "confidence_intervals_95": {
            "bct_ai_counterfeit_detection": confidence_interval_95(bct_ai_counterfeit),
            "bct_ai_recall_efficiency": confidence_interval_95(bct_ai_recall),
        },
        "note": (
            "t-statistics are computed from 10 genuinely independent "
            "230,000-transaction runs (laptop scale), not reverse-engineered "
            "to match the paper's full-scale (2.3M-transaction) reported "
            "values of t(9)=6.42 (counterfeit detection) / t(9)=7.11 "
            "(recall efficiency); differing from those exact figures is "
            "expected."
        ),
    }
