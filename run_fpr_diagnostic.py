"""False-positive diagnostic for the live-inference arms (ARM1, ARM4, ARM5).

WHY THIS EXISTS
---------------
The counterfeit-detection metric used throughout this project is

    detection = P(PTS < alert_threshold | row is counterfeit)

which is a true-positive rate with no specificity term. A detector that
pushes *everything* below the threshold scores 100% on it.

The provenance-isolation experiment (OPTION4, 2026-08-23) found exactly
that failure mode in a provenance-only arm: 98.70% TPR against a 40.78%
FPR, versus the proxy design's 72.49% TPR at 0.00% FPR. On a balanced
measure the proxy discriminates BETTER, despite the far lower headline
number.

That arm used proxy classifiers. The published live arms (ARM1/ARM4/ARM5)
use trained models and have never had their FPR measured. If they show
similar inflation, the redesign claim in the manuscript needs qualifying
before submission. This script measures it.

STATUS
------
This is a post-hoc VALIDITY DIAGNOSTIC, not a registered hypothesis test.
It adds no new hypothesis. It measures a quantity that should have been
reported alongside the existing detection rates and was not. Report the
numbers it produces regardless of what they show.

REQUIREMENTS
------------
Needs torch + tensorflow (the trained CNN and Isolation Forest), so it
must run in the full project environment, not a lightweight sandbox:

    pip install -r requirements.txt
    python run_fpr_diagnostic.py

Runtime is comparable to the original live-inference run (~40-80 min for
all three arms over ten seeds). Use --arms ARM4 --seeds 42,43,44 for a
faster first look; the FPR is very stable across seeds, so three seeds is
enough to see whether there is a problem.

OUTPUT
------
results/fpr_diagnostic.json, plus a summary table on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_modules import cnn_verification as cnn_mod
from ai_modules import isolation_forest_detector as if_mod
from evaluation.live_inference import (
    compute_cnn_authenticity,
    compute_isolation_scores,
    compute_org_reputation,
    compute_provenance_scores,
    split_timeline,
)
from pts.product_trust_score import ProductState, compute_pts
from simulation.anomaly_injector import inject_anomalies
from simulation.transaction_generator import generate_transactions
from utils import load_config

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
OUT_PATH = Path(__file__).resolve().parent / "results" / "fpr_diagnostic.json"

# Mirrors ARMS in run_live_inference_experiment.py for the live arms only.
ARMS = {
    "ARM1": {"clustered": False, "cnn": "published", "if": "published",
             "label": "Live inference, published CNN, uniform injection"},
    "ARM4": {"clustered": True, "cnn": "published", "if": "published",
             "label": "Live inference, published CNN, CLUSTERED injection"},
    "ARM5": {"clustered": True, "cnn": "degraded", "if": "published",
             "label": "Live inference, DEGRADED CNN, CLUSTERED injection"},
}


def build_transactions(config: Dict[str, Any], seed: int, clustered: bool):
    sim = config["simulation"]
    tx = generate_transactions(sim["transactions_per_run"], sim["organizations"], seed=seed)
    return inject_anomalies(
        tx, sim["anomaly_counts"], seed=seed,
        cluster_counterfeit_by_manufacturer=clustered,
        counterfeit_cluster_fraction=config.get("data_quality", {}).get(
            "counterfeit_cluster_fraction", 0.3),
    )


def train_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train each distinct model once, exactly as the registered run does."""
    cnn_cfg, if_cfg = config["cnn"], config["isolation_forest"]
    models: Dict[str, Any] = {}

    for name, overlap in (("published", 0.10), ("degraded", 0.30)):
        mc = cnn_mod.CNNConfig(
            image_size=cnn_cfg["runtime_image_size"],
            dense_units_1=cnn_cfg["dense_units_1"], dense_units_2=cnn_cfg["dense_units_2"],
            dropout=cnn_cfg["dropout"], batch_size=cnn_cfg["batch_size"],
            max_epochs=cnn_cfg["max_epochs"],
            early_stopping_patience=cnn_cfg["early_stopping_patience"],
            pretrained=cnn_cfg.get("pretrained", True), overlap_factor=overlap,
        )
        print("  training CNN (%s, overlap=%.2f)..." % (name, overlap))
        model, _ = cnn_mod.train_cnn(mc, cnn_cfg["n_authentic"],
                                     cnn_cfg["n_tampered_per_class"], seed=42)
        models[f"cnn_{name}"] = model
        models[f"cnn_{name}_cfg"] = mc

    for name, contamination in (("published", 0.05), ("degraded", 0.20)):
        ic = if_mod.IsolationForestConfig(contamination=contamination)
        print("  training Isolation Forest (%s, contamination=%.2f)..." % (name, contamination))
        model = if_mod.train_isolation_forest(ic, if_cfg["n_train_readings"], seed=42)
        calibration = if_mod.calibrate_score_normalization(model, ic, seed=42 + 1000)
        models[f"if_{name}"] = model
        models[f"if_{name}_calibration"] = calibration

    return models


def pts_for(rows, s1, cnn_scores, iso_scores, pts_cfg) -> np.ndarray:
    """Two-component PTS over supplied per-row signals."""
    n = len(rows)
    dcs = rows["drug_class"].to_numpy()
    out = np.empty(n)
    for i in range(n):
        st = ProductState(
            custody_chain_trust_scores=[float(s1[i])],
            temperature_readings_c=[],
            cnn_authenticity_score=float(cnn_scores[i]),
            isolation_forest_anomaly_score=float(iso_scores[i]),
        )
        dc = dcs[i] if dcs[i] in ("A", "B", "C") else "C"
        cc = pts_cfg["drug_classes"][dc]
        out[i] = compute_pts(st, {
            "provenance_integrity": cc["w1_provenance_integrity"],
            "ai_confidence": cc["w8_ai_confidence"],
        })["pts"]
    return out


def run_arm_seed(arm_id: str, config: Dict[str, Any], models: Dict[str, Any],
                 seed: int) -> Dict[str, Any]:
    """Measure TPR and FPR for one live arm on one seed."""
    arm = ARMS[arm_id]
    pts_cfg = config["pts"]
    alert = pts_cfg["alert_threshold"]
    image_size = config["cnn"]["runtime_image_size"]

    tx = build_transactions(config, seed, arm["clustered"])
    burn_in, evaluation = split_timeline(tx)
    trust, _ = compute_org_reputation(burn_in)
    s1_all = compute_provenance_scores(evaluation, trust)

    cnn_model = models[f"cnn_{arm['cnn']}"]
    if_model = models[f"if_{arm['if']}"]
    if_cal = models[f"if_{arm['if']}_calibration"]

    counterfeit = (evaluation["anomaly_type"] == "counterfeit_product").to_numpy()
    clean = (~evaluation["is_anomaly"].to_numpy())

    out: Dict[str, Any] = {"seed": seed}
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
        out[label] = {
            "n": int(len(rows)),
            "rate_below_alert": float(np.mean(pts_vals < alert)),
            "pts_mean": float(np.mean(pts_vals)),
            "s1_mean": float(np.mean(s1_all[mask])),
        }

    tpr = out["counterfeit"]["rate_below_alert"]
    fpr = out["clean"]["rate_below_alert"]
    out["tpr"] = tpr
    out["fpr"] = fpr
    out["tpr_minus_fpr"] = tpr - fpr
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=str, default="ARM1,ARM4,ARM5")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: all ten)")
    args = parser.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS)
    arm_ids = [a.strip().upper() for a in args.arms.split(",")]

    t0 = time.time()
    print("#" * 74)
    print("FALSE-POSITIVE DIAGNOSTIC — live-inference arms")
    print("Post-hoc validity check, not a registered hypothesis test.")
    print("Arms: %s | Seeds: %s" % (arm_ids, seeds))
    print("#" * 74)

    config = load_config()
    print("Training models once (protocol section 4 reuse rule)...")
    models = train_models(config)

    results: Dict[str, Any] = {}
    for arm_id in arm_ids:
        print("\n%s — %s" % (arm_id, ARMS[arm_id]["label"]))
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            r = run_arm_seed(arm_id, config, models, seed)
            per_seed.append(r)
            print("  seed %d: TPR=%.4f  FPR=%.4f  TPR-FPR=%+.4f  (n_cf=%d, n_clean=%d)" % (
                seed, r["tpr"], r["fpr"], r["tpr_minus_fpr"],
                r["counterfeit"]["n"], r["clean"]["n"]))
        tprs = np.array([r["tpr"] for r in per_seed])
        fprs = np.array([r["fpr"] for r in per_seed])
        results[arm_id] = {
            "label": ARMS[arm_id]["label"],
            "per_seed": per_seed,
            "tpr_mean": float(tprs.mean()), "tpr_sd": float(tprs.std(ddof=1)) if len(tprs) > 1 else 0.0,
            "fpr_mean": float(fprs.mean()), "fpr_sd": float(fprs.std(ddof=1)) if len(fprs) > 1 else 0.0,
            "tpr_minus_fpr_mean": float((tprs - fprs).mean()),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Post-hoc FPR validity diagnostic for live-inference arms",
        "prompted_by": "OPTION4 provenance-isolation finding (98.70% TPR / 40.78% FPR)",
        "alert_threshold": config["pts"]["alert_threshold"],
        "seeds": seeds,
        "arms": results,
        "duration_seconds": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 74)
    print("%-6s %10s %10s %12s   %s" % ("ARM", "TPR", "FPR", "TPR-FPR", "reading"))
    print("-" * 74)
    print("%-6s %9.2f%% %9.2f%% %+11.4f   %s" % (
        "ARM6a", 72.49, 0.00, 0.7249, "proxy baseline (measured 2026-08-23)"))
    print("%-6s %9.2f%% %9.2f%% %+11.4f   %s" % (
        "ARM6b", 98.70, 40.78, 0.5791, "provenance-only (measured 2026-08-23)"))
    for arm_id in arm_ids:
        r = results[arm_id]
        flag = "*** FPR INFLATED ***" if r["fpr_mean"] > 0.10 else "ok"
        print("%-6s %9.2f%% %9.2f%% %+11.4f   %s" % (
            arm_id, r["tpr_mean"] * 100, r["fpr_mean"] * 100,
            r["tpr_minus_fpr_mean"], flag))
    print("=" * 74)
    print("""
HOW TO READ THIS
  If ARM4's FPR is near zero, the published 92.29% detection figure is
  sound and the OPTION4 finding is confined to the provenance-only arm.

  If ARM4's FPR is materially above zero (say >10%), the redesign's
  headline number carries the same specificity problem, and the
  manuscript's causal-coupling claim needs qualifying BEFORE submission.
  That is a substantive revision, not a footnote.

  Either way: report the number. Do not omit it because it is
  inconvenient -- that is the failure mode this whole paper is about.
""")
    print("Written to %s (%.1f min)" % (OUT_PATH, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
