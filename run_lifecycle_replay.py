"""Round-3: the central comparison rerun on COHERENT product lifecycles.

Protocol: preregistration/ROUND3_lifecycle_PROTOCOL.txt (written before this code).
Only lifecycle generation and what depends on it change relative to
run_chronological_replay.py (kept unchanged; its results are the archived
event-stream stress test): same seeds, schemes, published CNN/IF, thresholds, rules,
1,500 clean products per seed, chronological replay through the state machine.

Lifecycle-dependent feature definitions:
  S1 provenance  = product of the trust of each custodian handing the unit on (transfers only)
  S4 verification = 'verify' events so far
  S5 age          = time since the product's manufacture event (true age)
  S7 regulatory   = 'under investigation' once any custodian so far has trust < 1
  S6 anomaly hist = model-flagged events since monitoring start (product-level sampling)
  org reputation  = burn-in counterfeit ONSETS per sending organisation

Usage: python run_lifecycle_replay.py [--smoke-test]
Output: results/lifecycle_replay_results.json (saved after every seed)
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
from blockchain.chaincode.product_state_machine import QUARANTINE, ProductStateMachine, TransitionError  # noqa: E402
from evaluation import live_inference as li  # noqa: E402
from evaluation.causal_features import order, prior_flag_counts  # noqa: E402
from pts.product_trust_score import ProductState, compute_component_scores, weights_for_drug_class  # noqa: E402
from run_full_pts_experiment import sequences  # noqa: E402
from run_live_inference_experiment import setup_logging, train_models  # noqa: E402
from simulation.lifecycle_generator import build_lifecycle_transactions  # noqa: E402
from simulation.lifecycle_invariants import lifecycle_violations  # noqa: E402
from utils import load_config  # noqa: E402

N_NORMAL = 1500
RELEASE_DELAY_H = 72.0
OUT = REPO / "results" / "lifecycle_replay_results.json"


def rate(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else float("nan")


def lifecycle_features(hist: pd.DataFrame, trust) -> pd.DataFrame:
    """Decision-time features from each product's own history (prefix-invariant)."""
    h = order(hist)
    default = float(np.mean(list(trust.values()))) if trust else 1.0
    tr = h["from_org"].map(lambda o: trust.get(o, default)).astype(float).where(h["is_transfer"], 1.0)
    g = h["product_id"]
    s1 = np.exp(np.log(tr.clip(lower=1e-9)).groupby(g, sort=False).cumsum())
    n_verify = (h["transaction_type"] == "verify").astype(int).groupby(g, sort=False).cumsum()
    ts = pd.to_datetime(h["timestamp"])
    age = (ts - ts.groupby(g, sort=False).transform("min")).dt.total_seconds() / 86400.0
    low = ((tr < 1.0) & h["is_transfer"]).astype(int).groupby(g, sort=False).cummax().astype(bool)
    return pd.DataFrame({"s1": s1.clip(0, 1), "n_verify": n_verify, "age_days": age,
                         "low_trust_seen": low}, index=h.index).reindex(hist.index)


def run_seed(config, seed, tx, cnn, ifm, cal, n_normal):
    pts_cfg = config["pts"]
    alert, quar = pts_cfg["alert_threshold"], pts_cfg["quarantine_threshold"]
    compromised = list(tx.attrs.get("compromised_manufacturers", []))
    onsets_all = tx[tx["counterfeit_onset"]]
    concentration = (float(onsets_all["from_org"].isin(compromised).mean())
                     if compromised and len(onsets_all) else None)
    viol = lifecycle_violations(tx)
    labels = tx.groupby("product_id")["anomaly_type"].agg(set)
    unexplained = int(sum(1 for p in viol.product_id.unique()
                          if not labels[p] & {"custody_break", "unauthorized_transfer"}))

    burn_in, evaluation = li.split_timeline(tx)
    rep = burn_in.assign(anomaly_type=np.where(burn_in["counterfeit_onset"], "counterfeit_product", "none"))
    trust, _ = li.compute_org_reputation(rep)
    eval_start = pd.to_datetime(evaluation["timestamp"]).min()

    anom_products = set(evaluation.loc[evaluation["is_anomaly"], "product_id"])
    ever_anom = set(tx.loc[tx["is_anomaly"], "product_id"])
    pool = sorted(set(evaluation["product_id"]) - ever_anom)
    rng = np.random.default_rng(seed)
    normal_products = set(rng.choice(pool, size=min(n_normal, len(pool)), replace=False))
    selected = anom_products | normal_products
    both = pd.concat([burn_in, evaluation])
    feats = lifecycle_features(both[both["product_id"].isin(selected)], trust)
    rows = order(evaluation[evaluation["product_id"].isin(selected)])
    f = feats.loc[rows.index]

    sev = rows["anomaly_severity"].fillna(1.0).to_numpy()
    sev_norm = np.where(rows["is_anomaly"].to_numpy(), np.clip((sev - 0.4) / 0.6, 0, 1), 1.0)
    cnn_auth = li.compute_cnn_authenticity(cnn, rows, sev_norm, config["cnn"]["runtime_image_size"])
    seqs = sequences(rows)
    if_seq = normalized_anomaly_scores(ifm, seqs.reshape(-1, 15), cal).reshape(len(rows), -1)
    if_mean, if_max = if_seq.mean(axis=1), if_seq.max(axis=1)
    prior = prior_flag_counts(rows, (cnn_auth < 0.5) | (if_max > 0.7))

    pts = np.empty(len(rows)); s8 = np.empty(len(rows))
    dcs = rows["drug_class"].to_numpy()
    for k in range(len(rows)):
        dc = dcs[k] if dcs[k] in ("A", "B", "C") else "C"
        state = ProductState(
            custody_chain_trust_scores=[float(f["s1"].iat[k])],
            temperature_readings_c=[float(x) for x in seqs[k, :, 0]],
            shock_events_severity=[0.2 * float(s) for s in seqs[k, :, 14] if s > 0],
            verification_count=int(f["n_verify"].iat[k]),
            days_since_manufacture=float(f["age_days"].iat[k]),
            anomaly_count=int(prior[k]),
            regulatory_status="under_investigation" if bool(f["low_trust_seen"].iat[k]) else "good_standing",
            cnn_authenticity_score=float(cnn_auth[k]),
            isolation_forest_anomaly_score=float(if_mean[k]))
        comp = compute_component_scores(state)
        w = weights_for_drug_class(pts_cfg["drug_classes"][dc])
        pts[k] = sum(w[c] * comp[c] for c in w) / sum(w.values())
        s8[k] = comp["ai_confidence"]

    is_cf = (rows["anomaly_type"] == "counterfeit_product").to_numpy()
    onset = rows["counterfeit_onset"].to_numpy()
    typ = np.where(rows["is_anomaly"].to_numpy(), rows["anomaly_type"].to_numpy(),
                   np.where(rows["product_id"].isin(normal_products).to_numpy(),
                            "normal_product", "normal_row_of_anomalous_product"))
    typ = np.where(onset, "counterfeit_onset", typ)
    trig = {"full_pts": (pts < alert, pts < quar), "s8_ai_confidence": (s8 < alert, s8 < quar),
            "sc4_cnn": (cnn_auth < 0.75, cnn_auth < 0.50), "sc5_if": (if_max > 0.70, if_max > 0.90)}
    trig["combined"] = (trig["full_pts"][0] | trig["sc4_cnn"][0] | trig["sc5_if"][0],
                        trig["full_pts"][1] | trig["sc4_cnn"][1] | trig["sc5_if"][1])
    groups = {t: typ == t for t in sorted(set(typ))}
    groups["counterfeit_all_events"] = is_cf
    transaction_level = {t: {"n": int(m.sum()), **{name: {"alert_or_worse": rate(a[m]), "quarantine": rate(q[m])}
                                                    for name, (a, q) in trig.items()}}
                         for t, m in groups.items()}

    # ---------------- chronological replay (hours since evaluation start)
    t_h = ((pd.to_datetime(rows["timestamp"]) - eval_start).dt.total_seconds() / 3600.0).to_numpy()
    pid = rows["product_id"].to_numpy(); is_anom = rows["is_anomaly"].to_numpy()
    prod = {}
    held_events = {"normal_product": 0, "anomalous_product": 0}
    n_events = {"normal_product": 0, "anomalous_product": 0}
    for k in range(len(rows)):
        p = pid[k]
        st = prod.get(p)
        if st is None:
            st = prod[p] = {"m": ProductStateMachine(p, monitoring_start=0.0), "q_start": None,
                            "anom_seen": False, "entered": 0, "releases": 0, "refused": 0,
                            "holds_h": [], "onset_t": None, "onset_k": None, "n_ev": 0,
                            "first_q_t": None, "first_q_ev": None, "cls": dcs[k]}
        m = st["m"]; st["n_ev"] += 1
        grp = "normal_product" if p in normal_products else "anomalous_product"
        n_events[grp] += 1
        if (m.state == QUARANTINE and st["q_start"] is not None
                and t_h[k] - st["q_start"] >= RELEASE_DELAY_H and not st["anom_seen"]):
            try:
                m.release(t_h[k], "QA", "lab test clear")
                st["releases"] += 1; st["holds_h"].append(t_h[k] - st["q_start"]); st["q_start"] = None
            except TransitionError:
                st["refused"] += 1
        held = m.state == QUARANTINE
        if held:
            held_events[grp] += 1
        if onset[k]:
            st["onset_t"] = t_h[k]; st["onset_k"] = st["n_ev"]
        m.cnn_reading(t_h[k], float(cnn_auth[k]))
        m.cold_chain_reading(t_h[k], float(if_max[k]))
        m.pts_update(t_h[k], float(pts[k]))
        if is_anom[k]:
            st["anom_seen"] = True
        if m.state == QUARANTINE and not held:
            st["entered"] += 1; st["q_start"] = t_h[k]
            if st["first_q_t"] is None:
                st["first_q_t"] = t_h[k]; st["first_q_ev"] = st["n_ev"]

    cf_products = sorted(set(pid[is_cf]))
    other_anom = sorted(anom_products - set(cf_products))

    def summary(keys):
        S = [prod[p] for p in keys if p in prod]
        holds = [h for s in S for h in s["holds_h"]]
        out = {"n_products": len(S), "ever_quarantined": rate([s["entered"] > 0 for s in S]),
               "quarantined_at_end": rate([s["m"].state == QUARANTINE for s in S]),
               "releases": int(sum(s["releases"] for s in S)),
               "refused_release_attempts": int(sum(s["refused"] for s in S)),
               "hold_hours_median": float(np.median(holds)) if holds else None}
        out["by_class"] = {c: {"n": int(sum(s["cls"] == c for s in S)),
                               "ever_quarantined": rate([s["entered"] > 0 for s in S if s["cls"] == c])}
                           for c in ("A", "B", "C")}
        return out

    cf = summary(cf_products)
    with_onset = [prod[p] for p in cf_products if prod[p]["onset_t"] is not None]
    caught = [s for s in with_onset if s["first_q_t"] is not None and s["first_q_t"] >= s["onset_t"]]
    cf.update({
        "with_onset_in_window": len(with_onset),
        "quarantined_after_onset": rate([s in caught for s in with_onset]),
        "quarantined_at_onset_event": rate([s["first_q_ev"] == s["onset_k"] for s in with_onset]),
        "delay_events_median": float(np.median([s["first_q_ev"] - s["onset_k"] for s in caught])) if caught else None,
        "delay_hours_median": float(np.median([s["first_q_t"] - s["onset_t"] for s in caught])) if caught else None,
    })
    replay = {"normal_products": summary(sorted(normal_products)), "counterfeit_products": cf,
              "other_anomalous_products": summary(other_anom),
              "events_processed_while_held": held_events, "events_processed": n_events}
    return {"seed": seed, "n_scored_events": int(len(rows)), "n_products": len(selected),
            "compromised_manufacturers": compromised, "counterfeit_onset_concentration": concentration,
            "invariant_violations": viol.rule.value_counts().to_dict(),
            "unexplained_violation_products": unexplained,
            "transaction_level": transaction_level, "replay": replay}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep (arm, seed) results already in the output file; run only the missing ones")
    args = ap.parse_args()
    log = setup_logging(); config = load_config()
    seeds = [42] if args.smoke_test else list(range(42, 52))
    n_normal = 100 if args.smoke_test else N_NORMAL
    if args.smoke_test:
        config["simulation"]["transactions_per_run"] = 20000
    t0 = time.time()
    models = None
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke_test": args.smoke_test,
           "protocol": "preregistration/ROUND3_lifecycle_PROTOCOL.txt", "seeds": seeds,
           "n_normal_products": n_normal, "release_delay_h": RELEASE_DELAY_H, "arms": {}}
    path = OUT.with_name("lifecycle_replay_smoke.json") if args.smoke_test else OUT
    prev = json.loads(path.read_text()) if (args.resume and path.exists()) else {}
    if prev:
        out["resumed_from"] = prev.get("generated_at")
        out["resume_note"] = ("models retrained in the resumed session with the same seeds; "
                              "seeds listed in resumed_seeds come from the new session")
        out["resumed_seeds"] = {}
    for arm, clustered in (("uniform", False), ("clustered", True)):
        done = {r["seed"]: r for r in prev.get("arms", {}).get(arm, [])}
        out["arms"][arm] = [done[s] for s in seeds if s in done]
        for seed in seeds:
            if seed in done:
                continue
            if prev:
                out["resumed_seeds"].setdefault(arm, []).append(seed)
            if models is None:
                models = train_models(config, log)
            tx = build_lifecycle_transactions(config, seed, clustered)
            r = run_seed(config, seed, tx, models["cnn_published"], models["if_published"],
                         models["if_published_calibration"], n_normal)
            tl, rp = r["transaction_level"], r["replay"]
            log.info("[%s] seed %d | cf events: full %.4f sc4 %.4f comb %.4f | clean events comb alert %.4f "
                     "| cf products quarantined %.4f | clean products quarantined %.4f | unexplained viol %d",
                     arm, seed, tl["counterfeit_all_events"]["full_pts"]["alert_or_worse"],
                     tl["counterfeit_all_events"]["sc4_cnn"]["alert_or_worse"],
                     tl["counterfeit_all_events"]["combined"]["alert_or_worse"],
                     tl.get("normal_product", {}).get("combined", {}).get("alert_or_worse", float("nan")),
                     rp["counterfeit_products"]["ever_quarantined"], rp["normal_products"]["ever_quarantined"],
                     r["unexplained_violation_products"])
            out["arms"][arm].append(r)
            out["arms"][arm].sort(key=lambda x: x["seed"])
            out["runtime_s"] = time.time() - t0
            path.write_text(json.dumps(out, indent=1, default=float))
    if prev and "clustered" in out.get("resumed_seeds", {}):
        # consistency check: re-run the last clustered seed finished in the earlier
        # session with this session's retrained models and record both.
        old = {r["seed"]: r for r in prev["arms"].get("clustered", [])}
        if old:
            s_chk = max(old)
            tx = build_lifecycle_transactions(config, s_chk, True)
            r = run_seed(config, s_chk, tx, models["cnn_published"], models["if_published"],
                         models["if_published_calibration"], n_normal)
            key = lambda x: (x["transaction_level"]["counterfeit_all_events"]["combined"]["alert_or_worse"],
                             x["replay"]["counterfeit_products"]["ever_quarantined"])
            out["consistency_check"] = {"arm": "clustered", "seed": s_chk,
                                        "earlier_session": key(old[s_chk]), "resumed_session": key(r)}
            log.info("consistency check seed %d: earlier %s resumed %s", s_chk, key(old[s_chk]), key(r))
            path.write_text(json.dumps(out, indent=1, default=float))
    log.info("wrote %s (%.0f s)", path, time.time() - t0)


if __name__ == "__main__":
    main()
