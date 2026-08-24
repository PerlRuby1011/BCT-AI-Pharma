"""OPTION5 pre-registered experiment: empirical-Bayes shrinkage fix for the
provenance-driven false-positive rate.

Protocol: preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt
(approved 2026-08-24). Read that file before reading this one — it defines
H7.1 (primary), H7.2 (positive control), the falsification rules, and the
binding reporting obligations this script's output must satisfy.

WHAT THIS MEASURES
-------------------
Reuses the exact ARM1/ARM4 configuration and the exact ten registered
seeds from ``run_fpr_diagnostic.py``. For each seed, computes:

  - baseline: compute_org_reputation(..., shrinkage=False)  [reproduces the
    already-registered ARM1/ARM4 numbers, included here only as an
    in-run sanity check that nothing else drifted]
  - shrunk:   compute_org_reputation(..., shrinkage=True)   [the OPTION5 fix]

for both ARM1 (uniform injection, positive control / H7.2) and ARM4
(clustered injection, primary test / H7.1), then reports paired
differences.

No-leakage note: the shrinkage strength k is derived automatically from
each seed's own burn-in data inside compute_org_reputation() itself (see
that function's docstring) — this script never touches k, never looks at
evaluation-half FPR before choosing anything. It only calls the function
twice per seed, once per shrinkage setting, and reports both.

REQUIREMENTS
------------
Needs torch + tensorflow (the trained CNN and Isolation Forest), so it
must run in the full project environment, not a lightweight sandbox:

    pip install -r requirements.txt
    python run_shrinkage_experiment.py

OUTPUT
------
results/shrinkage_experiment.json, plus a summary table on stdout.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.live_inference import (
    compute_cnn_authenticity,
    compute_isolation_scores,
    compute_org_reputation,
    compute_provenance_scores,
    split_timeline,
)
from run_fpr_diagnostic import DEFAULT_SEEDS, build_transactions, pts_for, train_models
from utils import load_config

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None

OUT_PATH = Path(__file__).resolve().parent / "results" / "shrinkage_experiment.json"

# clustered flag mirrors ARM1 / ARM4 in run_fpr_diagnostic.py; both use the
# published CNN and Isolation Forest, so only injection clustering varies.
CONDITIONS = {
    "ARM1": {"clustered": False, "label": "uniform injection (H7.2, positive control)"},
    "ARM4": {"clustered": True, "label": "clustered injection (H7.1, primary)"},
}


def run_condition_seed(cond_id: str, config: Dict[str, Any], models: Dict[str, Any],
                        seed: int, shrinkage: bool) -> Dict[str, Any]:
    """Measure TPR and FPR for one condition/seed/shrinkage setting."""
    cond = CONDITIONS[cond_id]
    pts_cfg = config["pts"]
    alert = pts_cfg["alert_threshold"]
    image_size = config["cnn"]["runtime_image_size"]

    tx = build_transactions(config, seed, cond["clustered"])
    burn_in, evaluation = split_timeline(tx)
    trust, _ = compute_org_reputation(burn_in, shrinkage=shrinkage)
    s1_all = compute_provenance_scores(evaluation, trust)

    cnn_model = models["cnn_published"]
    if_model = models["if_published"]
    if_cal = models["if_published_calibration"]

    counterfeit = (evaluation["anomaly_type"] == "counterfeit_product").to_numpy()
    clean = (~evaluation["is_anomaly"].to_numpy())

    out: Dict[str, Any] = {"seed": seed, "shrinkage": shrinkage}
    for label, mask in (("counterfeit", counterfeit), ("clean", clean)):
        rows = evaluation.loc[mask]
        if len(rows) == 0:
            out[label] = {"n": 0, "rate_below_alert": float("nan")}
            continue
        sev = rows["anomaly_severity"].fillna(0.4).to_numpy()
        sev_norm = np.clip((sev - 0.4) / 0.6, 0.0, 1.0)
        cnn_scores = compute_cnn_authenticity(cnn_model, rows, sev_norm, image_size)
        iso_scores = compute_isolation_scores(if_model, rows, if_cal)
        pts_vals = pts_for(rows, s1_all[mask], cnn_scores, iso_scores, pts_cfg)
        out[label] = {"n": int(len(rows)), "rate_below_alert": float(np.mean(pts_vals < alert))}

    out["tpr"] = out["counterfeit"]["rate_below_alert"]
    out["fpr"] = out["clean"]["rate_below_alert"]
    return out


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Paired t-test of a - b, with the OPTION5 section-4 power guard."""
    diff = a - b
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    if _stats is not None and len(diff) > 1 and sd > 0:
        t, p = _stats.ttest_rel(a, b)
        se = sd / np.sqrt(len(diff))
        ci = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
        ci_excludes_zero = not (ci[0] <= 0 <= ci[1])
    else:
        t, p, ci, ci_excludes_zero = float("nan"), float("nan"), (float("nan"), float("nan")), False
    near_zero_variance = sd < 1e-6
    return {
        "mean_diff": mean_diff, "sd": sd, "t": float(t), "p": float(p),
        "ci95": [ci[0], ci[1]],
        "guard_inconclusive": near_zero_variance is False and not ci_excludes_zero,
        "near_zero_variance": near_zero_variance,
    }


def main() -> None:
    t0 = time.time()
    print("#" * 78)
    print("OPTION5 EXPERIMENT — empirical-Bayes provenance shrinkage")
    print("Pre-registered 2026-08-24. Approved. Executed once.")
    print("#" * 78)

    config = load_config()
    print("Training models once (protocol section 4 reuse rule)...")
    models = train_models(config)

    results: Dict[str, Any] = {}
    for cond_id, cond in CONDITIONS.items():
        print("\n%s — %s" % (cond_id, cond["label"]))
        baseline_rows: List[Dict[str, Any]] = []
        shrunk_rows: List[Dict[str, Any]] = []
        for seed in DEFAULT_SEEDS:
            b = run_condition_seed(cond_id, config, models, seed, shrinkage=False)
            s = run_condition_seed(cond_id, config, models, seed, shrinkage=True)
            baseline_rows.append(b)
            shrunk_rows.append(s)
            print("  seed %d: baseline TPR=%.4f FPR=%.4f | shrunk TPR=%.4f FPR=%.4f" % (
                seed, b["tpr"], b["fpr"], s["tpr"], s["fpr"]))

        tpr_b = np.array([r["tpr"] for r in baseline_rows])
        fpr_b = np.array([r["fpr"] for r in baseline_rows])
        tpr_s = np.array([r["tpr"] for r in shrunk_rows])
        fpr_s = np.array([r["fpr"] for r in shrunk_rows])

        results[cond_id] = {
            "label": cond["label"],
            "baseline": {"per_seed": baseline_rows,
                         "tpr_mean": float(tpr_b.mean()), "fpr_mean": float(fpr_b.mean())},
            "shrunk": {"per_seed": shrunk_rows,
                       "tpr_mean": float(tpr_s.mean()), "fpr_mean": float(fpr_s.mean())},
            "d_tpr_baseline_minus_shrunk": paired_test(tpr_b, tpr_s),
            "d_fpr_baseline_minus_shrunk": paired_test(fpr_b, fpr_s),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt",
        "status": "post-hoc, pre-registered before execution, executed once",
        "seeds": DEFAULT_SEEDS,
        "conditions": results,
        "duration_seconds": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("-" * 78)
    for cond_id, r in results.items():
        b, s = r["baseline"], r["shrunk"]
        d_fpr = b["fpr_mean"] - s["fpr_mean"]
        d_tpr = b["tpr_mean"] - s["tpr_mean"]
        rel_fpr_reduction = (d_fpr / b["fpr_mean"] * 100) if b["fpr_mean"] > 0 else float("nan")
        print("%s (%s)" % (cond_id, r["label"]))
        print("  baseline: TPR=%.2f%% FPR=%.2f%%" % (b["tpr_mean"] * 100, b["fpr_mean"] * 100))
        print("  shrunk:   TPR=%.2f%% FPR=%.2f%%" % (s["tpr_mean"] * 100, s["fpr_mean"] * 100))
        print("  d_FPR=%.2fpp (%.1f%% relative reduction)  d_TPR=%.2fpp" % (
            d_fpr * 100, rel_fpr_reduction, d_tpr * 100))
    print("=" * 78)
    print("""
HOW TO READ THIS
  ARM4 is the primary test (H7.1). Prediction: FPR drops by >=25%% relative
  (12.34%% -> <=9.26%%) with TPR cost <=5.0pp. If not met, that is an
  informative null/partial result per protocol section 6, not a failure to
  hide -- it means the FPR is substantially driven by genuinely elevated
  (not just noisy) organization rates under clustering, and a different
  class of fix (product-level features) is needed, not this one.

  ARM1 is the positive control (H7.2). Prediction: both TPR and FPR change
  by <=1.0pp between baseline and shrunk. If not, the shrinkage correction
  has unintended side effects and needs reworking before it can be trusted
  anywhere.

  Report both outcomes regardless of which way they land.
""")
    print("Written to %s (%.1f min)" % (OUT_PATH, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
