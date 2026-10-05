"""Round-2 review: decision-time features + chronological replay through the
unified state machine.

Fixes three problems a second review found in run_full_pts_experiment.py:
  1. provenance / regulatory features used each product's COMPLETE
     evaluation-period chain (future information);   -> evaluation/causal_features.py
  2. anomaly history treated unscored rows as unflagged;  -> PRODUCT-level sampling:
     every evaluation-period event of each selected product is scored;
  3. "unified enforcement" was a per-transaction Boolean union, not the state
     machine;                                         -> events replayed in time
     order through ProductStateMachine (SC-4, SC-5, SC-7, authorised release).

Protocol (fixed before running; see preregistration/ROUND2_replay_PROTOCOL.txt):
  * same generator, seeds 42-51, uniform + clustered, published CNN / IF;
  * products: every product with an anomaly in the evaluation half, plus
    N_NORMAL clean products (no anomaly anywhere in the run), sampled with the seed;
  * monitoring starts at the evaluation-period start; custody features carry
    burn-in history forward; S6 counts model flags from monitoring start;
  * per event, in order: SC-4 (CNN P(authentic)), SC-5 (max per-reading IF
    score), SC-7 (full eight-component PTS from causal features);
  * release policy: at an event arriving >= RELEASE_DELAY_H after quarantine
    began, QA attempts release iff no anomaly-labelled event of the product has
    occurred so far (a lab test would clear it); the machine may refuse.

Usage:  python run_chronological_replay.py [--smoke-test]
Output: results/chronological_replay_results.json
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

from ai_modules.isolation_forest_detector import normalized_anomaly_scores  # noqa: E402
from blockchain.chaincode.product_state_machine import (  # noqa: E402
    QUARANTINE, ProductStateMachine, TransitionError,
)
from evaluation import live_inference as li  # noqa: E402
from evaluation.causal_features import causal_features, order, prior_flag_counts  # noqa: E402
from pts.product_trust_score import ProductState, compute_component_scores, weights_for_drug_class  # noqa: E402
from run_full_pts_experiment import sequences  # noqa: E402
from run_live_inference_experiment import build_transactions, setup_logging, train_models  # noqa: E402
from utils import load_config  # noqa: E402

N_NORMAL = 1500
RELEASE_DELAY_H = 72.0
OUT = REPO / "results" / "chronological_replay_results.json"


def rate(x) -> float:
    x = np.asarray(x)
    return float(x.mean()) if x.size else float("nan")


def run_seed(config, seed, tx, cnn, ifm, cal, n_normal):
    pts_cfg = config["pts"]
    alert, quar = pts_cfg["alert_threshold"], pts_cfg["quarantine_threshold"]
    burn_in, evaluation = li.split_timeline(tx)
    trust, _ = li.compute_org_reputation(burn_in)
    eval_start = pd.to_datetime(evaluation["timestamp"]).min()

    anom_products = set(evaluation.loc[evaluation["is_anomaly"], "product_id"])
    ever_anom = set(tx.loc[tx["is_anomaly"], "product_id"])
    pool = sorted(set(evaluation["product_id"]) - ever_anom)
    rng = np.random.default_rng(seed)
    normal_products = set(rng.choice(pool, size=min(n_normal, len(pool)), replace=False))
    selected = anom_products | normal_products

    # split_timeline re-indexes, so build history from its two halves (consistent index)
    both = pd.concat([burn_in, evaluation])
    hist = both[both["product_id"].isin(selected)]
    feats = causal_features(hist, trust)
    rows = order(evaluation[evaluation["product_id"].isin(selected)])
    f = feats.loc[rows.index]

    sev = rows["anomaly_severity"].fillna(1.0).to_numpy()
    sev_norm = np.where(rows["is_anomaly"].to_numpy(), np.clip((sev - 0.4) / 0.6, 0, 1), 1.0)
    cnn_auth = li.compute_cnn_authenticity(cnn, rows, sev_norm, config["cnn"]["runtime_image_size"])
    seqs = sequences(rows)
    if_seq = normalized_anomaly_scores(ifm, seqs.reshape(-1, 15), cal).reshape(len(rows), -1)
    if_mean, if_max = if_seq.mean(axis=1), if_seq.max(axis=1)
    flagged = (cnn_auth < 0.5) | (if_max > 0.7)
    prior = prior_flag_counts(rows, flagged)

    pts = np.empty(len(rows))
    s8 = np.empty(len(rows))
    for k, (_, r) in enumerate(rows.iterrows()):
        dc = r["drug_class"] if r["drug_class"] in ("A", "B", "C") else "C"
        state = ProductState(
            custody_chain_trust_scores=[float(f["s1"].iat[k])],
            temperature_readings_c=[float(x) for x in seqs[k, :, 0]],
            shock_events_severity=[0.2 * float(s) for s in seqs[k, :, 14] if s > 0],
            verification_count=int(f["n_events"].iat[k]),
            days_since_manufacture=float(f["elapsed_days"].iat[k]),
            anomaly_count=int(prior[k]),
            regulatory_status="under_investigation" if bool(f["low_trust_seen"].iat[k]) else "good_standing",
            cnn_authenticity_score=float(cnn_auth[k]),
            isolation_forest_anomaly_score=float(if_mean[k]),
        )
        comp = compute_component_scores(state)
        w = weights_for_drug_class(pts_cfg["drug_classes"][dc])
        pts[k] = sum(w[c] * comp[c] for c in w) / sum(w.values())
        s8[k] = comp["ai_confidence"]

    typ = np.where(rows["is_anomaly"].to_numpy(), rows["anomaly_type"].to_numpy(),
                   np.where(rows["product_id"].isin(normal_products).to_numpy(),
                            "normal_product", "normal_row_of_anomalous_product"))
    trig = {
        "full_pts": (pts < alert, pts < quar),
        "s8_ai_confidence": (s8 < alert, s8 < quar),
        "sc4_cnn": (cnn_auth < 0.75, cnn_auth < 0.50),
        "sc5_if": (if_max > 0.70, if_max > 0.90),
    }
    trig["combined"] = (trig["full_pts"][0] | trig["sc4_cnn"][0] | trig["sc5_if"][0],
                        trig["full_pts"][1] | trig["sc4_cnn"][1] | trig["sc5_if"][1])
    transaction_level = {}
    for t in sorted(set(typ)):
        m = typ == t
        transaction_level[t] = {"n": int(m.sum()), **{
            name: {"alert_or_worse": rate(a[m]), "quarantine": rate(q[m])} for name, (a, q) in trig.items()}}

    # ---------------- chronological replay through the state machine
    t_h = ((pd.to_datetime(rows["timestamp"]) - eval_start).dt.total_seconds() / 3600.0).to_numpy()
    pid = rows["product_id"].to_numpy()
    is_anom = rows["is_anomaly"].to_numpy()
    is_cf = (rows["anomaly_type"] == "counterfeit_product").to_numpy()
    prod = {}
    n_events_while_held = {"normal_product": 0, "anomalous_product": 0}
    n_events = {"normal_product": 0, "anomalous_product": 0}
    for k in range(len(rows)):
        p = pid[k]
        st = prod.get(p)
        if st is None:
            st = prod[p] = {"m": ProductStateMachine(p, monitoring_start=0.0), "q_start": None,
                            "anom_seen": False, "entered": 0, "releases": 0, "refused": 0,
                            "holds_h": [], "cf_events": 0, "cf_caught_at_event": 0,
                            "held_at_cf_arrival": 0}
        m = st["m"]
        group = "normal_product" if p in normal_products else "anomalous_product"
        n_events[group] += 1
        # release attempt (QA) before processing the new event
        if m.state == QUARANTINE and st["q_start"] is not None \
                and t_h[k] - st["q_start"] >= RELEASE_DELAY_H and not st["anom_seen"]:
            try:
                m.release(t_h[k], "QA", "lab test clear")
                st["releases"] += 1
                st["holds_h"].append(t_h[k] - st["q_start"])
                st["q_start"] = None
            except TransitionError:
                st["refused"] += 1
        held_on_arrival = m.state == QUARANTINE
        if held_on_arrival:
            n_events_while_held[group] += 1
        if is_cf[k] and held_on_arrival:
            st["held_at_cf_arrival"] += 1
        m.cnn_reading(t_h[k], float(cnn_auth[k]))
        m.cold_chain_reading(t_h[k], float(if_max[k]))
        m.pts_update(t_h[k], float(pts[k]))
        if is_anom[k]:
            st["anom_seen"] = True
        if is_cf[k]:
            st["cf_events"] += 1
            st["cf_caught_at_event"] += int(m.state == QUARANTINE)
        if m.state == QUARANTINE and not held_on_arrival:
            st["entered"] += 1
            st["q_start"] = t_h[k]

    def prod_summary(keys):
        S = [prod[p] for p in keys if p in prod]
        holds = [h for s in S for h in s["holds_h"]]
        return {"n_products": len(S),
                "ever_quarantined": rate([s["entered"] > 0 for s in S]),
                "quarantined_at_end": rate([s["m"].state == QUARANTINE for s in S]),
                "quarantine_entries_per_product": rate([s["entered"] for s in S]),
                "releases": int(sum(s["releases"] for s in S)),
                "refused_release_attempts": int(sum(s["refused"] for s in S)),
                "hold_hours_median": float(np.median(holds)) if holds else None,
                "hold_hours_mean": float(np.mean(holds)) if holds else None}

    cf_products = sorted(set(pid[is_cf]))
    cf_S = [prod[p] for p in cf_products]
    other_anom = sorted(anom_products - set(cf_products))
    replay = {
        "normal_products": prod_summary(sorted(normal_products)),
        "counterfeit_products": {**prod_summary(cf_products),
            "counterfeit_events": int(sum(s["cf_events"] for s in cf_S)),
            "counterfeit_event_quarantined_after_processing": rate(
                [x for s in cf_S for x in [1] * s["cf_caught_at_event"] + [0] * (s["cf_events"] - s["cf_caught_at_event"])]),
            "counterfeit_event_already_held_on_arrival": rate(
                [x for s in cf_S for x in [1] * s["held_at_cf_arrival"] + [0] * (s["cf_events"] - s["held_at_cf_arrival"])])},
        "other_anomalous_products": prod_summary(other_anom),
        "events_processed_while_held": n_events_while_held,
        "events_processed": n_events,
    }
    return {"seed": seed, "n_scored_events": int(len(rows)), "n_products": len(selected),
            "transaction_level": transaction_level, "replay": replay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    config = load_config()
    seeds = [42] if args.smoke_test else list(range(42, 52))
    n_normal = 100 if args.smoke_test else N_NORMAL
    if args.smoke_test:
        config["simulation"]["transactions_per_run"] = 20000
        config["simulation"]["anomaly_counts"] = {k: max(1, v // 100) for k, v in
                                                  config["simulation"]["anomaly_counts"].items()}
    t0 = time.time()
    models = train_models(config, log)
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke_test": args.smoke_test,
           "seeds": seeds, "n_normal_products": n_normal, "release_delay_h": RELEASE_DELAY_H,
           "arms": {}}
    path = OUT.with_name("chronological_replay_smoke.json") if args.smoke_test else OUT
    for arm, clustered in (("uniform", False), ("clustered", True)):
        out["arms"][arm] = []
        for seed in seeds:
            tx = build_transactions(config, seed, clustered)
            r = run_seed(config, seed, tx, models["cnn_published"], models["if_published"],
                         models["if_published_calibration"], n_normal)
            tl, rp = r["transaction_level"], r["replay"]
            log.info("[%s] seed %d | cf rows: full alert %.4f, combined alert %.4f quar %.4f | "
                     "normal-product rows combined quar %.4f | normal products ever quarantined %.4f | "
                     "cf events quarantined %.4f",
                     arm, seed, tl.get("counterfeit_product", {}).get("full_pts", {}).get("alert_or_worse", np.nan),
                     tl.get("counterfeit_product", {}).get("combined", {}).get("alert_or_worse", np.nan),
                     tl.get("counterfeit_product", {}).get("combined", {}).get("quarantine", np.nan),
                     tl.get("normal_product", {}).get("combined", {}).get("quarantine", np.nan),
                     rp["normal_products"]["ever_quarantined"],
                     rp["counterfeit_products"]["counterfeit_event_quarantined_after_processing"])
            out["arms"][arm].append(r)
            out["runtime_s"] = time.time() - t0
            path.write_text(json.dumps(out, indent=1, default=float))   # save after every seed
    log.info("wrote %s (%.0f s)", path, time.time() - t0)


if __name__ == "__main__":
    main()
