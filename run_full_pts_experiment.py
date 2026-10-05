"""Review item 1: the FULL eight-component PTS under live model inference,
scored on counterfeit, other-anomaly AND normal transactions.

Why: the published counterfeit experiment (and the live-inference arms)
score a reduced two-component PTS (provenance + AI confidence). A C&IE
review pointed out this does not test the eight-component enforcement score
the paper proposes, and reports no false-alert rate on normal products.

What this does (additive; nothing existing is modified):
  * same transaction generation, anomaly injection, burn-in/evaluation split,
    organisation reputation (S1), live CNN and live Isolation Forest as
    evaluation/live_inference.py (published models: CNN overlap 0.10,
    IF contamination 0.05);
  * every one of the eight PTS components is filled from an observable
    signal of the transaction -- never from its label:
      S1 provenance       realised custody-chain reputation (unchanged)
      S2 temperature      the per-transaction reading sequence (column 0)
      S3 handling         shock counts in the same sequence (column 14)
      S4 verification     number of custody events for the product so far
      S5 age              days since the product's first custody event
      S6 anomaly history  prior model-flagged events for the same product
      S7 regulatory       'under_investigation' if any hop org has trust < 1
      S8 AI confidence    live CNN P(authentic) and live IF score
  * scores evaluation-half rows: all anomalies + a random sample of normals;
  * reports, per condition, alert (PTS<0.75) and quarantine (PTS<0.50) rates
    for counterfeit, each other anomaly type, and normal rows, overall and
    per drug class, plus paired ablations (full, full minus provenance, full
    minus AI, the published reduced score, provenance-only, AI-only).

Usage:
    python run_full_pts_experiment.py                 # 10 seeds, clustered + uniform
    python run_full_pts_experiment.py --smoke-test    # 1 seed, small sample
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from ai_modules.isolation_forest_detector import _apply_anomaly, _generate_normal_readings, normalized_anomaly_scores  # noqa: E402
from evaluation import live_inference as li  # noqa: E402
from pts.product_trust_score import ProductState, compute_component_scores, weights_for_drug_class  # noqa: E402
from run_live_inference_experiment import build_transactions, setup_logging, train_models  # noqa: E402
from utils import load_config  # noqa: E402

OUT = REPO / "results" / "full_pts_live_results.json"
SEQ = li.IF_SEQUENCE_LENGTH


def sequences(rows: pd.DataFrame) -> np.ndarray:
    """Per-transaction reading sequences, identical to li.compute_isolation_scores."""
    out = np.empty((len(rows), SEQ, 15), dtype=np.float32)
    for k, (tid, at) in enumerate(zip(rows["transaction_id"].to_numpy(), rows["anomaly_type"].to_numpy())):
        rng = np.random.default_rng(li._tx_seed(tid) ^ 0x5EED)
        block = _generate_normal_readings(SEQ, rng)
        if at == "temperature_excursion":
            block = _apply_anomaly(block, "Gradual Drift", rng)
        out[k] = block
    return out


def product_history_features(evaluation: pd.DataFrame, flagged: pd.Series):
    """S4/S5/S6 inputs from the product's own custody history (evaluation half, causal)."""
    ev = evaluation.sort_values("timestamp")
    g = ev.groupby("product_id")
    n_events = g.cumcount() + 1
    first_ts = g["timestamp"].transform("min")
    age_days = (pd.to_datetime(ev["timestamp"]) - pd.to_datetime(first_ts)).dt.total_seconds() / 86400.0
    prior_flags = flagged.reindex(ev.index).fillna(False).astype(int).groupby(ev["product_id"]).cumsum() \
        - flagged.reindex(ev.index).fillna(False).astype(int)
    return (n_events.reindex(evaluation.index), age_days.reindex(evaluation.index),
            prior_flags.reindex(evaluation.index))


def score_seed(config, seed, tx, cnn, ifm, cal, n_normal, rng_seed):
    pts_cfg = config["pts"]
    burn_in, evaluation = li.split_timeline(tx)
    trust, _ = li.compute_org_reputation(burn_in)
    s1_all = pd.Series(li.compute_provenance_scores(evaluation, trust), index=evaluation.index)

    anomalies = evaluation[evaluation["is_anomaly"]]
    normals = evaluation[~evaluation["is_anomaly"]]
    normals = normals.sample(n=min(n_normal, len(normals)), random_state=rng_seed)
    rows = pd.concat([anomalies, normals])

    sev = rows["anomaly_severity"].fillna(1.0).to_numpy()
    sev_norm = np.where(rows["is_anomaly"].to_numpy(), np.clip((sev - 0.4) / 0.6, 0, 1), 1.0)
    cnn_auth = li.compute_cnn_authenticity(cnn, rows, sev_norm, config["cnn"]["runtime_image_size"])
    seqs = sequences(rows)
    if_seq = normalized_anomaly_scores(ifm, seqs.reshape(-1, 15), cal).reshape(len(rows), SEQ)
    if_score = if_seq.mean(axis=1)

    # model-flagged events (observable, not labels) for S6 anomaly history
    flagged = pd.Series((cnn_auth < 0.5) | (if_seq.max(axis=1) > 0.7), index=rows.index)
    n_events, age_days, prior_flags = product_history_features(evaluation, flagged)

    low_trust_orgs = {o for o, t in trust.items() if t < 1.0}
    chain_orgs = evaluation.groupby("product_id")["from_org"].apply(set)

    conds = {"full": None, "full_minus_provenance": "provenance_integrity",
             "full_minus_ai": "ai_confidence"}
    recs = []
    for k, (idx, r) in enumerate(rows.iterrows()):
        dc = r["drug_class"] if r["drug_class"] in ("A", "B", "C") else "C"
        shocks = seqs[k, :, 14]
        state = ProductState(
            custody_chain_trust_scores=[float(s1_all[idx])],
            temperature_readings_c=[float(x) for x in seqs[k, :, 0]],
            shock_events_severity=[0.2 * float(s) for s in shocks if s > 0],
            verification_count=int(n_events[idx]),
            days_since_manufacture=float(age_days[idx]),
            anomaly_count=int(prior_flags[idx]),
            regulatory_status=("under_investigation"
                               if chain_orgs.get(r["product_id"], set()) & low_trust_orgs else "good_standing"),
            cnn_authenticity_score=float(cnn_auth[k]),
            isolation_forest_anomaly_score=float(if_score[k]),
        )
        comp = compute_component_scores(state)
        w = weights_for_drug_class(pts_cfg["drug_classes"][dc])
        out = {"type": r["anomaly_type"] if r["is_anomaly"] else "normal", "drug_class": dc}
        for name, drop in conds.items():
            ww = {c: v for c, v in w.items() if c != drop}
            out[name] = sum(ww[c] * comp[c] for c in ww) / sum(ww.values())
        cw = pts_cfg["drug_classes"][dc]
        a, b = cw["w1_provenance_integrity"], cw["w8_ai_confidence"]
        out["reduced_two_component"] = (a * comp["provenance_integrity"] + b * comp["ai_confidence"]) / (a + b)
        out["provenance_only"] = comp["provenance_integrity"]
        out["ai_only"] = comp["ai_confidence"]
        for c, v in comp.items():
            out["c_" + c] = v
        # OPTION11: per-contract signals (no new random draws -> item-1 numbers unchanged)
        out["sc4_cnn"] = float(cnn_auth[k])
        out["sc5_if_max"] = float(if_seq[k].max())
        recs.append(out)
    df = pd.DataFrame(recs)

    alert, quar = pts_cfg["alert_threshold"], pts_cfg["quarantine_threshold"]
    scores = ["full", "full_minus_provenance", "full_minus_ai", "reduced_two_component",
              "provenance_only", "ai_only"]
    summary = {}
    for t, sub in df.groupby("type"):
        summary[t] = {"n": int(len(sub))}
        for s in scores:
            summary[t][s] = {"alert_rate": float((sub[s] < alert).mean()),
                             "quarantine_rate": float((sub[s] < quar).mean()),
                             "mean": float(sub[s].mean())}
        summary[t]["by_class"] = {dc: {"n": int(len(g)),
                                       "full_alert": float((g["full"] < alert).mean()),
                                       "full_quarantine": float((g["full"] < quar).mean())}
                                  for dc, g in sub.groupby("drug_class")}
        summary[t]["component_means"] = {c: float(sub["c_" + c].mean()) for c in
                                         [x[2:] for x in df.columns if x.startswith("c_")]}
        # OPTION11 unified enforcement (pre-registered thresholds, not tuned)
        sc7_q, sc7_a = sub["full"] < quar, sub["full"] < alert
        sc5_q, sc5_a = sub["sc5_if_max"] > 0.90, sub["sc5_if_max"] > 0.70
        sc4_q, sc4_a = sub["sc4_cnn"] < 0.50, sub["sc4_cnn"] < 0.75
        q = sc7_q | sc5_q | sc4_q
        a = q | sc7_a | sc5_a | sc4_a
        summary[t]["unified"] = {
            "alert_or_worse_rate": float(a.mean()), "quarantine_rate": float(q.mean()),
            "trigger_alert_or_worse": {"sc4": float(sc4_a.mean()), "sc5": float(sc5_a.mean()),
                                       "sc7": float(sc7_a.mean())},
            "trigger_quarantine": {"sc4": float(sc4_q.mean()), "sc5": float(sc5_q.mean()),
                                   "sc7": float(sc7_q.mean())},
            "by_class": {dc: {"n": int(len(g)),
                              "alert_or_worse": float(((g["full"] < alert) | (g["sc5_if_max"] > 0.70)
                                                       | (g["sc4_cnn"] < 0.75)).mean()),
                              "quarantine": float(((g["full"] < quar) | (g["sc5_if_max"] > 0.90)
                                                   | (g["sc4_cnn"] < 0.50)).mean())}
                         for dc, g in sub.groupby("drug_class")},
        }
    return {"seed": seed, "n_rows": int(len(df)), "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--normal-sample", type=int, default=10000)
    ap.add_argument("--enforcement", action="store_true",
                    help="OPTION11: write to unified_enforcement_live_results.json")
    args = ap.parse_args()
    log = setup_logging()
    config = load_config()
    seeds = [42] if args.smoke_test else list(range(42, 52))
    n_normal = 500 if args.smoke_test else args.normal_sample
    if args.smoke_test:
        config["simulation"]["transactions_per_run"] = 20000
        config["simulation"]["anomaly_counts"] = {k: max(1, v // 100) for k, v in
                                                  config["simulation"]["anomaly_counts"].items()}
    t0 = time.time()
    models = train_models(config, log)
    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke_test": args.smoke_test,
               "seeds": seeds, "normal_sample_per_seed": n_normal, "arms": {}}
    for arm, clustered in (("uniform", False), ("clustered", True)):
        per_seed = []
        for seed in seeds:
            tx = build_transactions(config, seed, clustered)
            r = score_seed(config, seed, tx, models["cnn_published"], models["if_published"],
                           models["if_published_calibration"], n_normal, seed)
            full = r["summary"].get("counterfeit_product", {}).get("full", {})
            nrm = r["summary"].get("normal", {}).get("full", {})
            log.info("[%s] seed %d  counterfeit alert=%.4f  normal alert=%.4f  normal quarantine=%.4f",
                     arm, seed, full.get("alert_rate", float("nan")),
                     nrm.get("alert_rate", float("nan")), nrm.get("quarantine_rate", float("nan")))
            uc = r["summary"].get("counterfeit_product", {}).get("unified", {})
            un = r["summary"].get("normal", {}).get("unified", {})
            log.info("[%s] seed %d  UNIFIED counterfeit alert=%.4f quar=%.4f | normal alert=%.4f quar=%.4f",
                     arm, seed, uc.get("alert_or_worse_rate", float("nan")),
                     uc.get("quarantine_rate", float("nan")), un.get("alert_or_worse_rate", float("nan")),
                     un.get("quarantine_rate", float("nan")))
            per_seed.append(r)
        results["arms"][arm] = per_seed
    results["runtime_s"] = time.time() - t0
    out = OUT.with_name("full_pts_live_smoke.json") if args.smoke_test else OUT
    if args.enforcement:
        out = out.with_name("unified_enforcement_live_" + ("smoke.json" if args.smoke_test else "results.json"))
    out.write_text(json.dumps(results, indent=1, default=float))
    log.info("wrote %s (%.0f s)", out, results["runtime_s"])


if __name__ == "__main__":
    main()
