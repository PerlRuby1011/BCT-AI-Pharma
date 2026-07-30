"""Execute the pre-registered live-inference experiment (ARMs 0-4).

Protocol frozen in ``Journal Journey/OPTION2_live_inference_PREREGISTRATION.txt``
(approved 2026-07-28). Runs exactly once. Every arm, every seed, no
exclusions.

Usage:
    python run_live_inference_experiment.py
    python run_live_inference_experiment.py --smoke-test
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_modules import cnn_verification as cnn_mod
from ai_modules import isolation_forest_detector as if_mod
from evaluation.live_inference import run_live_inference_seed, split_timeline
from evaluation.run_simulation import _json_default, _sanitize_for_json, run_single_simulation_run
from evaluation.statistical_validation import compute_statistical_validation_from_runs
from simulation.anomaly_injector import inject_anomalies
from simulation.transaction_generator import generate_transactions
from utils import load_config

RESULTS_DIR = REPO_ROOT / "results"
LOG_PATH = RESULTS_DIR / "live_inference.log"
OUT_PATH = RESULTS_DIR / "live_inference_results.json"

SEEDS = list(range(42, 52))

# Arms, exactly as frozen in protocol section 4.
ARMS: List[Dict[str, Any]] = [
    {"id": "ARM0",  "label": "Proxy baseline (published design, full data)",
     "mode": "proxy", "clustered": False, "half": False},
    {"id": "ARM0b", "label": "Proxy baseline, evaluation half only (matched control)",
     "mode": "proxy", "clustered": False, "half": True},
    {"id": "ARM1",  "label": "Live inference, published CNN, uniform injection",
     "mode": "live", "clustered": False, "cnn": "published", "if": "published"},
    {"id": "ARM2",  "label": "Live inference, DEGRADED CNN (overlap 0.30)",
     "mode": "live", "clustered": False, "cnn": "degraded", "if": "published"},
    {"id": "ARM3",  "label": "Live inference, DEGRADED Isolation Forest (contamination 0.20)",
     "mode": "live", "clustered": False, "cnn": "published", "if": "degraded"},
    {"id": "ARM4",  "label": "Live inference, published CNN, CLUSTERED injection",
     "mode": "live", "clustered": True, "cnn": "published", "if": "published"},
    # ARM5 added 2026-07-28 after independent review established that the
    # registered H1 comparison (ARM1 vs ARM2) had ZERO POWER: under uniform
    # injection BCT-AI detection varies by only 0.11 pp across the entire
    # physically reachable range of S8, i.e. below the 1.0 pp threshold
    # regardless of the CNN. ARM5 is ARM4's data condition with ARM2's
    # manipulation -- the 2x2 cell that was not run -- and is the same
    # published-vs-degraded CNN contrast in the one condition where the
    # registered metric demonstrably can move (23.84 pp reachable range).
    # Prediction P5 was recorded BEFORE this arm ran. It is NOT the
    # originally registered H1 test; it is a powered test of the same
    # substantive hypothesis, and must be reported as post-hoc motivated.
    {"id": "ARM5",  "label": "Live inference, DEGRADED CNN, CLUSTERED injection (H1, powered)",
     "mode": "live", "clustered": True, "cnn": "degraded", "if": "published"},
]


def setup_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    log = logging.getLogger("live_inference"); log.setLevel(logging.INFO); log.propagate = False
    if not log.handlers:
        log.addHandler(fh); log.addHandler(sh)
    return log


def build_transactions(config: Dict[str, Any], seed: int, clustered: bool):
    sim = config["simulation"]
    tx = generate_transactions(sim["transactions_per_run"], sim["organizations"], seed=seed)
    return inject_anomalies(
        tx, sim["anomaly_counts"], seed=seed,
        cluster_counterfeit_by_manufacturer=clustered,
        counterfeit_cluster_fraction=config.get("data_quality", {}).get(
            "counterfeit_cluster_fraction", 0.3),
    )


def train_models(config: Dict[str, Any], log: logging.Logger) -> Dict[str, Any]:
    """Train each distinct model exactly once (protocol section 4 reuse rule)."""
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
        log.info("Training CNN [%s] overlap_factor=%.2f ...", name, overlap)
        t0 = time.time()
        model, _ = cnn_mod.train_cnn(mc, cnn_cfg["n_authentic"],
                                     cnn_cfg["n_tampered_per_class"], seed=42)
        metrics = cnn_mod.evaluate_cnn(model, mc, n_test_per_class=50, seed=42 + 999)
        log.info("  CNN [%s] trained in %.1fs | weighted F1 = %.4f",
                 name, time.time() - t0, metrics["weighted_avg"]["f1"])
        models[f"cnn_{name}"] = model
        models[f"cnn_{name}_f1"] = metrics["weighted_avg"]["f1"]

    for name, contamination in (("published", 0.05), ("degraded", 0.20)):
        ic = if_mod.IsolationForestConfig(
            n_estimators=if_cfg["n_estimators"], max_samples=if_cfg["max_samples"],
            contamination=contamination, n_features=if_cfg["n_features"],
            warning_threshold=if_cfg["warning_threshold"],
            critical_threshold=if_cfg["critical_threshold"],
        )
        log.info("Training Isolation Forest [%s] contamination=%.2f ...", name, contamination)
        t0 = time.time()
        model = if_mod.train_isolation_forest(ic, if_cfg["n_train_readings"], seed=42)
        calibration = if_mod.calibrate_score_normalization(model, ic, seed=42 + 1000)
        overall = if_mod.evaluate_all_anomaly_types(model, ic, calibration=calibration,
                                                    seed=42 + 999)["overall"]
        log.info("  IF [%s] trained in %.1fs | overall detection = %.4f",
                 name, time.time() - t0, overall["detection"])
        models[f"if_{name}"] = model
        models[f"if_{name}_calibration"] = calibration
        models[f"if_{name}_detection"] = overall["detection"]

    return models


def run_arm(arm: Dict[str, Any], config: Dict[str, Any], models: Dict[str, Any],
            log: logging.Logger) -> Dict[str, Any]:
    log.info("=" * 78)
    log.info("%s — %s", arm["id"], arm["label"])
    log.info("=" * 78)
    t0 = time.time()
    per_seed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for seed in SEEDS:
        try:
            if arm["mode"] == "proxy":
                cfg = copy.deepcopy(config)
                cfg.setdefault("data_quality", {})["cluster_counterfeit_by_manufacturer"] = arm["clustered"]
                if not arm["half"]:
                    result = run_single_simulation_run(cfg, seed)
                else:
                    # Matched control: published proxy design restricted to the
                    # same evaluation half the live arms use, so ARM1 vs ARM0b
                    # is not confounded by sample restriction.
                    tx = build_transactions(cfg, seed, arm["clustered"])
                    _, evaluation = split_timeline(tx)
                    result = _proxy_on_subset(cfg, seed, evaluation)
            else:
                tx = build_transactions(config, seed, arm["clustered"])
                result = run_live_inference_seed(
                    config, seed, tx,
                    models[f"cnn_{arm['cnn']}"],
                    models[f"if_{arm['if']}"],
                    models[f"if_{arm['if']}_calibration"],
                )
            per_seed.append(result)
            log.info("  seed %d: detection=%.4f  recall=%.4f  (n_cf=%d)",
                     seed, result["bct_ai_counterfeit_detection"],
                     result["bct_ai_recall_efficiency"], result["n_counterfeit"])
        except Exception as exc:
            log.error("  seed %d FAILED: %s", seed, exc)
            log.debug(traceback.format_exc())
            failures.append({"seed": seed, "error": str(exc),
                             "traceback": traceback.format_exc()})

    detection = np.array([r["bct_ai_counterfeit_detection"] for r in per_seed])
    ai_only = np.array([r["ai_only_counterfeit_detection"] for r in per_seed])
    recall = np.array([r["bct_ai_recall_efficiency"] for r in per_seed])

    stats = compute_statistical_validation_from_runs(per_seed) if len(per_seed) >= 2 else None
    summary = {
        "n_seeds": len(per_seed),
        "detection_mean": float(detection.mean()) if len(detection) else None,
        "detection_std": float(detection.std(ddof=1)) if len(detection) > 1 else None,
        "ai_only_detection_mean": float(ai_only.mean()) if len(ai_only) else None,
        "bct_ai_advantage_pp": (
            float((detection.mean() - ai_only.mean()) * 100) if len(detection) else None),
        "recall_mean": float(recall.mean()) if len(recall) else None,
    }
    if stats:
        summary["ci95_detection"] = stats["confidence_intervals_95"]["bct_ai_counterfeit_detection"]
        summary["t_statistic"] = stats["counterfeit_detection_ttest"]["t_statistic"]
        summary["p_value"] = stats["counterfeit_detection_ttest"]["p_value"]

    # NB: `x or default` is wrong here -- a legitimate 0.0 is falsy and would
    # be silently replaced (this printed "advantage=+nan pp" when the true
    # advantage was exactly 0.0). Use explicit None checks.
    def _f(value: Any, fallback: float = float("nan")) -> float:
        return fallback if value is None else float(value)

    log.info("%s SUMMARY: detection=%.4f +/- %.4f | AI-only=%.4f | advantage=%+.2f pp | %.1fs",
             arm["id"], _f(summary["detection_mean"]), _f(summary["detection_std"], 0.0),
             _f(summary["ai_only_detection_mean"]), _f(summary["bct_ai_advantage_pp"]),
             time.time() - t0)

    return {"arm": arm["id"], "label": arm["label"], "config": arm,
            "per_seed": per_seed, "failures": failures, "summary": summary,
            "statistical_validation": stats, "duration_seconds": round(time.time() - t0, 1)}


def _proxy_on_subset(config: Dict[str, Any], seed: int, subset) -> Dict[str, Any]:
    """Published proxy design evaluated on a supplied transaction subset."""
    from pts.product_trust_score import ProductState, compute_pts

    pts_cfg = config["pts"]
    rng = np.random.default_rng(seed + 10_000)
    ai_only_weights = {"ai_confidence": 1.0}

    def _metrics(rows):
        n = len(rows)
        if n == 0:
            nan = float("nan")
            return {"n": 0, "d": nan, "ao_d": nan, "r": nan, "ao_r": nan}
        severity_norm = (rows["anomaly_severity"].to_numpy() - 0.4) / 0.6
        custody = np.clip(1.0 - 0.90 * severity_norm + rng.normal(0, 0.05, n), 0, 1)
        cnn = np.clip(1.0 - 0.30 * severity_norm + rng.normal(0, 0.06, n), 0, 1)
        iso = np.clip(rng.normal(0.38, 0.13, n), 0, 1)
        drug_classes = rows["drug_class"].to_numpy()
        bct, ao = np.empty(n), np.empty(n)
        for i in range(n):
            st = ProductState(custody_chain_trust_scores=[float(custody[i])],
                              temperature_readings_c=[], cnn_authenticity_score=float(cnn[i]),
                              isolation_forest_anomaly_score=float(iso[i]))
            dc = drug_classes[i] if drug_classes[i] in ("A", "B", "C") else "C"
            cc = pts_cfg["drug_classes"][dc]
            bct[i] = compute_pts(st, {"provenance_integrity": cc["w1_provenance_integrity"],
                                      "ai_confidence": cc["w8_ai_confidence"]})["pts"]
            ao[i] = compute_pts(st, ai_only_weights)["pts"]
        a, q = pts_cfg["alert_threshold"], pts_cfg["quarantine_threshold"]
        return {"n": n, "d": float(np.mean(bct < a)), "ao_d": float(np.mean(ao < a)),
                "r": float(np.mean(bct < q)), "ao_r": float(np.mean(ao < q))}

    cf = _metrics(subset.loc[subset["anomaly_type"] == "counterfeit_product"])
    an = _metrics(subset.loc[subset["is_anomaly"]])
    return {"seed": seed, "n_evaluation_transactions": int(len(subset)),
            "n_counterfeit": cf["n"], "n_anomalies": an["n"],
            "bct_ai_counterfeit_detection": cf["d"], "ai_only_counterfeit_detection": cf["ao_d"],
            "bct_ai_recall_efficiency": an["r"], "ai_only_recall_efficiency": an["ao_r"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--arms", type=str, default=None,
                        help="Comma-separated arm ids to run (e.g. ARM5). Models are "
                             "still trained deterministically from the same seeds, so a "
                             "single-arm run reproduces that arm exactly.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output filename under results/ (default live_inference_results.json)")
    args = parser.parse_args()

    log = setup_logging()
    log.info("#" * 78)
    log.info("PRE-REGISTERED LIVE-INFERENCE EXPERIMENT — single execution")
    log.info("Protocol: OPTION2_live_inference_PREREGISTRATION.txt (approved 2026-07-28)")
    log.info("#" * 78)

    config = load_config()
    global SEEDS
    if args.smoke_test:
        SEEDS = [42, 43]
        config["simulation"]["transactions_per_run"] = 20000
        config["cnn"]["n_authentic"] = 40; config["cnn"]["n_tampered_per_class"] = 10
        config["cnn"]["max_epochs"] = 2
        config["isolation_forest"]["n_train_readings"] = 2000
        log.info("SMOKE TEST MODE — results are NOT the pre-registered experiment")

    selected = ARMS
    if args.arms:
        wanted = {a.strip().upper() for a in args.arms.split(",")}
        selected = [a for a in ARMS if a["id"].upper() in wanted]
        log.info("Running subset: %s", [a["id"] for a in selected])

    t0 = time.time()
    models = train_models(config, log)
    arms = [run_arm(arm, config, models, log) for arm in selected]

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "smoke_test": args.smoke_test,
        "seeds": SEEDS,
        "transactions_per_run": config["simulation"]["transactions_per_run"],
        "model_metrics": {k: v for k, v in models.items()
                          if isinstance(v, (float, int, tuple))},
        "arms": arms,
        "total_minutes": round((time.time() - t0) / 60, 1),
    }
    out_path = RESULTS_DIR / args.out if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(output), fh, indent=2, default=_json_default)
    log.info("Wrote %s", out_path)
    log.info("TOTAL: %.1f minutes", output["total_minutes"])


if __name__ == "__main__":
    main()
