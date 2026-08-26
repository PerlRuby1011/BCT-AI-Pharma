"""OPTION7 — chain-length-normalized provenance aggregation.

Measures the raw-product aggregation (registered/published) against a
chain-length-normalized geometric mean, on the same seeds, models and
frozen thresholds.

Efficiency: the CNN and Isolation Forest scores depend only on the
transaction rows and their severity -- never on S1 -- so they are computed
ONCE per seed and reused for both aggregations. Both variants therefore see
numerically identical model outputs, and the only difference between them is
the aggregation itself.

Reports, for each aggregation:
  * TPR/FPR at the frozen alert threshold (0.75), comparable to every
    previously published number; and
  * threshold-free ROC AUC, which is what an aggregation change actually
    improves. These answer different questions and both are reported,
    because an AUC gain does not automatically become a TPR/FPR gain at a
    threshold that was fixed for the old score's range.

Usage:
    python run_aggregation_experiment.py --seeds 42,43,44
    python run_aggregation_experiment.py --seeds 42-51 --out aggregation_dev.json
    python run_aggregation_experiment.py --seeds 52-56 --out aggregation_fresh.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import ai_modules.cnn_verification as cnn_mod
import ai_modules.isolation_forest_detector as if_mod
from evaluation.live_inference import (
    compute_cnn_authenticity, compute_isolation_scores, compute_org_reputation,
    compute_provenance_scores, split_timeline,
)
from simulation.anomaly_injector import inject_anomalies
from simulation.transaction_generator import generate_transactions
from utils import load_config

AGGREGATIONS = ("product", "geometric_mean")
REPO_ROOT = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_transactions(config: Dict[str, Any], seed: int, clustered: bool) -> pd.DataFrame:
    sim = config["simulation"]
    tx = generate_transactions(sim["transactions_per_run"], sim["organizations"], seed=seed)
    return inject_anomalies(
        tx, sim["anomaly_counts"], seed=seed,
        cluster_counterfeit_by_manufacturer=clustered,
        counterfeit_cluster_fraction=config.get("data_quality", {}).get(
            "counterfeit_cluster_fraction", 0.3),
    )


def train_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the published CNN and Isolation Forest once.

    Mirrors run_fpr_diagnostic.train_models() exactly for the published
    configuration, so the models are identical to the ones behind every
    previously reported ARM4 number.
    """
    cnn_cfg, if_cfg = config["cnn"], config["isolation_forest"]
    models: Dict[str, Any] = {}

    mc = cnn_mod.CNNConfig(
        image_size=cnn_cfg["runtime_image_size"],
        dense_units_1=cnn_cfg["dense_units_1"], dense_units_2=cnn_cfg["dense_units_2"],
        dropout=cnn_cfg["dropout"], batch_size=cnn_cfg["batch_size"],
        max_epochs=cnn_cfg["max_epochs"],
        early_stopping_patience=cnn_cfg["early_stopping_patience"],
        pretrained=cnn_cfg.get("pretrained", True), overlap_factor=0.10,
    )
    print("  training CNN (published, overlap=0.10)...", flush=True)
    model, _ = cnn_mod.train_cnn(mc, cnn_cfg["n_authentic"],
                                 cnn_cfg["n_tampered_per_class"], seed=42)
    models["cnn"] = model

    ic = if_mod.IsolationForestConfig(contamination=0.05)
    print("  training Isolation Forest (published, contamination=0.05)...", flush=True)
    iforest = if_mod.train_isolation_forest(ic, if_cfg["n_train_readings"], seed=42)
    models["if"] = iforest
    models["if_cal"] = if_mod.calibrate_score_normalization(iforest, ic, seed=42 + 1000)
    return models


def pts_for(rows: pd.DataFrame, s1: np.ndarray, cnn_scores: np.ndarray,
            iso_scores: np.ndarray, pts_cfg: Dict[str, Any]) -> np.ndarray:
    """Composite PTS over the two counterfeit-relevant components."""
    s8 = (cnn_scores + (1.0 - iso_scores)) / 2.0
    classes = rows["drug_class"].to_numpy()
    w1 = np.array([pts_cfg["drug_classes"][c if c in ("A", "B", "C") else "C"]
                   ["w1_provenance_integrity"] for c in classes])
    w8 = np.array([pts_cfg["drug_classes"][c if c in ("A", "B", "C") else "C"]
                   ["w8_ai_confidence"] for c in classes])
    return (w1 * s1 + w8 * s8) / (w1 + w8)


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC of ``scores`` as a suspicion score (higher = more suspect)."""
    n_pos = int(positive.sum()); n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_seed(config: Dict[str, Any], models: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """One seed, both aggregations, one pass of model inference."""
    pts_cfg = config["pts"]
    alert = pts_cfg["alert_threshold"]
    image_size = config["cnn"]["runtime_image_size"]

    tx = build_transactions(config, seed, clustered=True)
    burn_in, evaluation = split_timeline(tx)
    trust, _ = compute_org_reputation(burn_in)

    counterfeit = (evaluation["anomaly_type"] == "counterfeit_product").to_numpy()
    clean = ~evaluation["is_anomaly"].to_numpy()

    s1 = {a: compute_provenance_scores(evaluation, trust, aggregation=a) for a in AGGREGATIONS}

    # --- model inference ONCE per mask, reused across aggregations ---
    cached: Dict[str, Dict[str, np.ndarray]] = {}
    for label, mask in (("counterfeit", counterfeit), ("clean", clean)):
        rows = evaluation.loc[mask]
        sev = rows["anomaly_severity"].fillna(0.4).to_numpy()
        sev_norm = np.clip((sev - 0.4) / 0.6, 0.0, 1.0)
        cached[label] = {
            "rows": rows,
            "cnn": compute_cnn_authenticity(models["cnn"], rows, sev_norm, image_size),
            "iso": compute_isolation_scores(models["if"], rows, models["if_cal"]),
        }

    cache_dir = REPO_ROOT / "results" / "signal_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"seed{seed}_arm4.npz",
        cnn_cf=cached["counterfeit"]["cnn"], iso_cf=cached["counterfeit"]["iso"],
        cnn_clean=cached["clean"]["cnn"], iso_clean=cached["clean"]["iso"],
        s1_prod_cf=s1["product"][counterfeit], s1_prod_clean=s1["product"][clean],
        s1_geo_cf=s1["geometric_mean"][counterfeit], s1_geo_clean=s1["geometric_mean"][clean],
        dc_cf=cached["counterfeit"]["rows"]["drug_class"].to_numpy().astype("U1"),
        dc_clean=cached["clean"]["rows"]["drug_class"].to_numpy().astype("U1"),
    )

    out: Dict[str, Any] = {"seed": seed,
                           "n_counterfeit": int(counterfeit.sum()),
                           "n_clean": int(clean.sum())}
    stash: Dict[str, Dict[str, np.ndarray]] = {}
    for agg in AGGREGATIONS:
        rates, pts_by_label = {}, {}
        for label, mask in (("counterfeit", counterfeit), ("clean", clean)):
            c = cached[label]
            p = pts_for(c["rows"], s1[agg][mask], c["cnn"], c["iso"], pts_cfg)
            pts_by_label[label] = p
            rates[label] = float(np.mean(p < alert))
        # threshold-free ranking quality over counterfeit-vs-clean
        all_pts = np.concatenate([pts_by_label["counterfeit"], pts_by_label["clean"]])
        is_pos = np.concatenate([np.ones(counterfeit.sum(), bool), np.zeros(clean.sum(), bool)])
        out[agg] = {
            "tpr": rates["counterfeit"],
            "fpr": rates["clean"],
            "tpr_minus_fpr": rates["counterfeit"] - rates["clean"],
            "auc_pts": roc_auc(-all_pts, is_pos),          # lower PTS == more suspect
            "s1_mean_counterfeit": float(s1[agg][counterfeit].mean()),
            "s1_mean_clean": float(s1[agg][clean].mean()),
        }
        stash[agg] = pts_by_label

    # --- matched-TPR comparison -------------------------------------------
    # The two aggregations put PTS on different scales, so their rates at a
    # single frozen threshold are not comparable. The meaningful question is:
    # held at the SAME sensitivity as the published design, does the new
    # aggregation raise fewer false alarms? The threshold is derived from the
    # baseline's own TPR, never chosen to make a number look good.
    target_tpr = out["product"]["tpr"]
    for agg in AGGREGATIONS:
        cf_pts = np.sort(stash[agg]["counterfeit"])
        idx = min(int(np.ceil(target_tpr * len(cf_pts))) - 1, len(cf_pts) - 1)
        thr = cf_pts[max(idx, 0)] if len(cf_pts) else float("nan")
        out[agg]["matched_threshold"] = float(thr)
        out[agg]["matched_tpr"] = float(np.mean(stash[agg]["counterfeit"] <= thr))
        out[agg]["matched_fpr"] = float(np.mean(stash[agg]["clean"] <= thr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--out", default="aggregation_experiment.json")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    started = time.time()
    print("#" * 78)
    print("OPTION7 — chain-length-normalized provenance aggregation")
    print(f"ARM4 configuration (clustered injection) | seeds: {seeds}")
    print("#" * 78)

    config = load_config()
    models = train_models(config)

    per_seed = []
    for seed in seeds:
        r = run_seed(config, models, seed)
        per_seed.append(r)
        p, g = r["product"], r["geometric_mean"]
        print(f"  seed {seed}: product TPR={p['tpr']:.4f} FPR={p['fpr']:.4f} AUC={p['auc_pts']:.4f}"
              f" | geomean TPR={g['tpr']:.4f} FPR={g['fpr']:.4f} AUC={g['auc_pts']:.4f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'aggregation':<20}{'TPR@0.75':>10}{'FPR@0.75':>10}{'TPR-FPR':>10}{'AUC':>10}"
          f"{'TPR@matched':>12}{'FPR@matched':>11}")
    print("-" * 78)
    summary = {}
    for agg in AGGREGATIONS:
        tpr = float(np.mean([r[agg]["tpr"] for r in per_seed]))
        fpr = float(np.mean([r[agg]["fpr"] for r in per_seed]))
        auc = float(np.mean([r[agg]["auc_pts"] for r in per_seed]))
        summary[agg] = {"tpr": tpr, "fpr": fpr, "auc": auc,
                        "tpr_std": float(np.std([r[agg]["tpr"] for r in per_seed], ddof=1)),
                        "fpr_std": float(np.std([r[agg]["fpr"] for r in per_seed], ddof=1))}
        mt = float(np.mean([r[agg]["matched_tpr"] for r in per_seed]))
        mf = float(np.mean([r[agg]["matched_fpr"] for r in per_seed]))
        summary[agg].update({"matched_tpr": mt, "matched_fpr": mf,
                             "matched_fpr_std": float(np.std([r[agg]["matched_fpr"] for r in per_seed], ddof=1))
                             if len(per_seed) > 1 else 0.0})
        print(f"{agg:<20}{tpr*100:>9.2f}%{fpr*100:>9.2f}%{(tpr-fpr):>10.4f}{auc:>10.4f}"
              f"{mt*100:>12.2f}%{mf*100:>10.2f}%")
    d_fpr = (summary["geometric_mean"]["fpr"] - summary["product"]["fpr"]) * 100
    d_tpr = (summary["geometric_mean"]["tpr"] - summary["product"]["tpr"]) * 100
    d_auc = summary["geometric_mean"]["auc"] - summary["product"]["auc"]
    print("-" * 78)
    print(f"{'delta (geo - prod)':<20}{d_tpr:>+9.2f}pp{d_fpr:>+9.2f}pp{'':>10}{d_auc:>+10.4f}")
    print("=" * 78)

    out_path = REPO_ROOT / "results" / args.out
    out_path.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "protocol": "OPTION7 chain-length-normalized aggregation",
         "arm": "ARM4 (clustered injection), published CNN + IF",
         "alert_threshold": config["pts"]["alert_threshold"],
         "seeds": seeds, "per_seed": per_seed, "summary": summary,
         "minutes": (time.time() - started) / 60.0}, indent=2))
    print(f"\nWritten to {out_path} ({(time.time()-started)/60:.1f} min)")


if __name__ == "__main__":
    main()
