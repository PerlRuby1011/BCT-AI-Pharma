"""V1: instrumented re-run of the archived round-3 lifecycle replay.

Spec: C&IE/new_experiments/ROUND4_SPEC.md (revision 2), section V1.

What is reused unchanged
    run_lifecycle_replay.run_seed (identical to commit 859e1e9), the generator,
    thresholds, release policy, train_models (CNN seed 42; IF seed 42,
    calibration seed 1042), config/config.yaml.

What changes (bookkeeping and the unit interface only)
    * The state machine used inside run_seed is replaced by
      InstrumentedMachine, which converts the replay's HOURS to the machine's
      MINUTES at the interface (hours_to_machine_time) and logs every call.
      Logged times are converted back to hours for reporting.
    * Per counterfeit product, interception fields are derived from that log
      joined with the regenerated event table (no ties, so (product, time)
      identifies an event).

Model provenance
    No round-3 checkpoints exist, so models are retrained with the archived
    procedure. Fingerprints and checkpoints are saved to results/models_v1/.
    Agreement with the archive is judged on OUTPUTS: every numeric field of
    every archived per-seed record is compared (counts exact; rates to 1e-12).
    Agreement on all archived SUMMARY outputs does not establish identical
    per-product trajectories (no per-product archive or checkpoints exist), so
    the interception measurements are attributed to THIS instrumented rerun,
    reported together with its agreement with the original summaries.
    The archived results file is never written.

Exit status
    0 only if every requested run is present and agrees with the archived
    summaries; with the default (all) arguments, also requires all 20
    (arm, seed) runs. Any mismatch or missing run -> exit 1.

Usage
    python run_v1_interception.py --seeds 42 --arms uniform   # first check (~40 min)
    python run_v1_interception.py                             # all 20 runs
    python run_v1_interception.py --resume                    # continue after interruption
Output
    results/lifecycle_replay_interception.json (saved after every seed)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import run_lifecycle_replay as rlr  # noqa: E402
from ai_modules.isolation_forest_detector import (  # noqa: E402
    _generate_normal_readings, normalized_anomaly_scores)
from blockchain.chaincode.product_state_machine import (  # noqa: E402
    QUARANTINE, ProductStateMachine, hours_to_machine_time, machine_time_to_hours)
from evaluation import live_inference as li  # noqa: E402
from evaluation.causal_features import order  # noqa: E402
from run_live_inference_experiment import setup_logging, train_models  # noqa: E402
from simulation.lifecycle_generator import build_lifecycle_transactions  # noqa: E402
from utils import load_config  # noqa: E402

OUT = REPO / "results" / "lifecycle_replay_interception.json"
ARCHIVE = REPO / "results" / "lifecycle_replay_results.json"
MODELS_DIR = REPO / "results" / "models_v1"
INSPECT = {"verify", "quality_check"}
CALL_LOG: list = []


class InstrumentedMachine(ProductStateMachine):
    """Current machine; replay hours converted to minutes; every call logged."""

    def _rec(self, method, t_h, before):
        # time as stored by the machine (minutes), converted back to hours
        t_back = machine_time_to_hours(self.history[-1]["t"])
        assert abs(t_back - t_h) < 1e-9, (t_back, t_h)
        CALL_LOG.append((self.product_id, float(t_h), t_back, method, before, self.state,
                         tuple(sorted(self.quarantine_reasons))))

    def cnn_reading(self, t, p_authentic, **kw):
        b = self.state; r = super().cnn_reading(hours_to_machine_time(t), p_authentic, **kw)
        self._rec("cnn", t, b); return r

    def cold_chain_reading(self, t, if_score, **kw):
        b = self.state; r = super().cold_chain_reading(hours_to_machine_time(t), if_score, **kw)
        self._rec("cold", t, b); return r

    def pts_update(self, t, pts, **kw):
        b = self.state; r = super().pts_update(hours_to_machine_time(t), pts, **kw)
        self._rec("pts", t, b); return r

    def release(self, t, role, reason_code):
        b = self.state; r = super().release(hours_to_machine_time(t), role, reason_code)
        self._rec("release", t, b); return r

    def check_telemetry(self, t):
        return super().check_telemetry(hours_to_machine_time(t))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def state_dict_sha(model):
    sd = model.state_dict()
    return sha(b"".join(sd[k].cpu().numpy().tobytes() for k in sorted(sd)))


def probe_hashes(models, config):
    """Hashes of model outputs on fixed probe inputs (IF: 2,000 readings; CNN: 512 rows)."""
    X = _generate_normal_readings(2000, np.random.default_rng(2024))
    s = normalized_anomaly_scores(models["if_published"], X, models["if_published_calibration"])
    tx = build_lifecycle_transactions(config, 42, False)
    _, ev = li.split_timeline(tx)
    rows = order(ev).iloc[:512]
    sev = rows["anomaly_severity"].fillna(1.0).to_numpy()
    sev_norm = np.where(rows["is_anomaly"].to_numpy(), np.clip((sev - 0.4) / 0.6, 0, 1), 1.0)
    p = li.compute_cnn_authenticity(models["cnn_published"], rows, sev_norm,
                                    config["cnn"]["runtime_image_size"])
    return {"if": sha(np.round(s, 12).tobytes()), "cnn": sha(np.round(p, 9).tobytes()),
            "cnn_mean": float(p.mean())}


def load_models(mdir: Path, config, log):
    """Load saved checkpoints instead of retraining, and VERIFY them.

    Weights must match fingerprints.json exactly (state-dict / pickle SHA-256),
    and the probe outputs computed on this machine are compared with the
    recorded ones. Weight mismatch aborts. A probe-output mismatch (possible
    across platforms/library builds) is recorded and also aborts, because the
    run would then not be an inference reproduction of the reference.
    """
    import torch
    from ai_modules import cnn_verification as cnn_mod
    fp = json.loads((mdir / "fingerprints.json").read_text())
    cnn_cfg = config["cnn"]; models = {}
    for name, overlap in (("cnn_published", 0.10), ("cnn_degraded", 0.30)):
        mc = cnn_mod.CNNConfig(
            image_size=cnn_cfg["runtime_image_size"], dense_units_1=cnn_cfg["dense_units_1"],
            dense_units_2=cnn_cfg["dense_units_2"], dropout=cnn_cfg["dropout"],
            batch_size=cnn_cfg["batch_size"], max_epochs=cnn_cfg["max_epochs"],
            early_stopping_patience=cnn_cfg["early_stopping_patience"],
            pretrained=False, overlap_factor=overlap)          # architecture only; weights loaded below
        m = cnn_mod.build_cnn_model(mc)
        m.load_state_dict(torch.load(mdir / f"{name}.pt", map_location="cpu"))
        m.to("cpu").eval()
        if state_dict_sha(m) != fp[name]["state_dict_sha256"]:
            raise SystemExit(f"checkpoint {name}: weights do not match fingerprints.json")
        models[name], models[f"{name}_f1"] = m, fp[name]["train_f1"]
    for name in ("if_published", "if_degraded"):
        blob = (mdir / f"{name}.pkl").read_bytes()
        if sha(blob) != fp[name]["pickle_sha256"]:
            raise SystemExit(f"checkpoint {name}: pickle does not match fingerprints.json")
        models[name], models[f"{name}_calibration"] = pickle.loads(blob)
        models[f"{name}_detection"] = fp[name]["train_detection"]
    pr = probe_hashes(models, config)
    ok = {"cnn": pr["cnn"] == fp["cnn_published"]["probe_p_authentic_sha256"],
          "if": pr["if"] == fp["if_published"]["probe_scores_sha256"]}
    log.info("loaded checkpoints from %s | weights verified | probe outputs match: %s", mdir, ok)
    if not all(ok.values()):
        raise SystemExit(f"probe outputs differ from the reference on this machine: {ok}")
    return models, {"loaded_from": str(mdir.relative_to(REPO)), "weights_verified": True,
                    "probe_outputs_match_reference": ok}


def fingerprint_and_save(models, config, log, session):
    import torch
    mdir = MODELS_DIR / f"session_{session}"
    mdir.mkdir(parents=True, exist_ok=True)
    fp = {}
    for name in ("cnn_published", "cnn_degraded"):
        buf = io.BytesIO(); torch.save(models[name].state_dict(), buf)
        (mdir / f"{name}.pt").write_bytes(buf.getvalue())
        fp[name] = {"state_dict_sha256": state_dict_sha(models[name]),
                    "train_f1": models[f"{name}_f1"]}
    for name in ("if_published", "if_degraded"):
        blob = pickle.dumps((models[name], models[f"{name}_calibration"]))
        (mdir / f"{name}.pkl").write_bytes(blob)
        fp[name] = {"pickle_sha256": sha(blob), "calibration": list(models[f"{name}_calibration"]),
                    "train_detection": models[f"{name}_detection"]}
    pr = probe_hashes(models, config)
    fp["if_published"]["probe_scores_sha256"] = pr["if"]
    fp["cnn_published"]["probe_p_authentic_sha256"] = pr["cnn"]
    fp["cnn_published"]["probe_p_authentic_mean"] = pr["cnn_mean"]
    fp["round3_log_reference"] = {"cnn_published_f1": 1.0, "if_published_detection": 0.96}
    (mdir / "fingerprints.json").write_text(json.dumps(fp, indent=1, default=float))
    log.info("model fingerprints written: %s", mdir / "fingerprints.json")
    return fp


def compare(archived, new, path=""):
    """List differences between two per-seed records (counts exact, floats 1e-12)."""
    diffs = []
    if isinstance(archived, dict):
        for k in archived:
            if k not in new:
                diffs.append({"field": f"{path}/{k}", "archived": "present", "new": "missing"})
            else:
                diffs += compare(archived[k], new[k], f"{path}/{k}")
    elif isinstance(archived, list):
        if len(archived) != len(new):
            diffs.append({"field": path, "archived": archived, "new": new})
        else:
            for i, (a, b) in enumerate(zip(archived, new)):
                diffs += compare(a, b, f"{path}[{i}]")
    elif isinstance(archived, bool) or archived is None or isinstance(archived, str):
        if archived != new:
            diffs.append({"field": path, "archived": archived, "new": new})
    elif isinstance(archived, int):
        if archived != new:
            diffs.append({"field": path, "archived": archived, "new": new})
    elif isinstance(archived, float):
        both_nan = isinstance(new, float) and np.isnan(archived) and np.isnan(new)
        if not both_nan and abs(archived - float(new)) > 1e-12:
            diffs.append({"field": path, "archived": archived, "new": float(new)})
    return diffs


def interception(config, seed, clustered):
    """Join the call log with the regenerated event table, per counterfeit product."""
    tx = build_lifecycle_transactions(config, seed, clustered)
    _, ev = li.split_timeline(tx)
    eval_start = pd.to_datetime(ev["timestamp"]).min()
    ev = order(ev)
    ev = ev.assign(t_h=(pd.to_datetime(ev["timestamp"]) - eval_start).dt.total_seconds() / 3600.0)
    log = pd.DataFrame(CALL_LOG, columns=["product_id", "t_h", "t_back_h", "method", "before",
                                         "after", "reasons"])
    cf_ids = set(ev.loc[ev["anomaly_type"] == "counterfeit_product", "product_id"])
    per, recs = log[log.product_id.isin(cf_ids)].groupby("product_id"), []
    for p, g in ev[ev.product_id.isin(cf_ids)].groupby("product_id", sort=False):
        typ = g["transaction_type"].tolist(); th = g["t_h"].tolist()
        on = np.flatnonzero(g["counterfeit_onset"].to_numpy())
        L = per.get_group(p) if p in per.groups else log.iloc[0:0]
        first_cnn = L[L.method == "cnn"].groupby("t_h")["before"].first()   # status before event readings
        q = L[(L.after == QUARANTINE) & (L.before != QUARANTINE)]
        first_q_t = float(q.t_h.iloc[0]) if len(q) else None          # join key (replay clock)
        first_q_back = float(q.t_back_h.iloc[0]) if len(q) else None  # machine time, converted back
        r = {"product_id": p, "onset_in_window": bool(len(on)),
             "first_hold_t_h": first_q_t,
             "first_hold_reasons": list(q.reasons.iloc[0]) if len(q) else None}
        if first_q_t is not None:
            r["first_hold_index"] = th.index(first_q_t)
        if len(on):
            i0 = int(on[0]); r["onset_index"], r["onset_t_h"] = i0, th[i0]
            disp = [i for i, t in enumerate(typ) if t == "dispense" and i >= i0]
            if disp:
                j = disp[0]; r["dispense_index"], r["dispense_t_h"] = j, th[j]
                r["held_at_dispense"] = bool(first_cnn.get(th[j]) == QUARANTINE)
                if first_q_t is not None and r["held_at_dispense"]:
                    k = r["first_hold_index"]
                    r["delay_events"] = k - i0
                    r["delay_h"] = first_q_back - th[i0]   # hours, from machine time converted back
                    span = typ[i0:k + 1]
                    r["scored_events_onset_to_hold"] = len(span)
                    r["inspections_onset_to_hold"] = sum(t in INSPECT for t in span)
        recs.append(r)
    sub = [r for r in recs if r.get("dispense_index") is not None]
    held = [r for r in sub if r["held_at_dispense"]]
    summ = {"n_substituted_in_window": len(sub), "held_at_dispense": len(held),
            "missed": len(sub) - len(held),
            "held_at_dispense_share": len(held) / len(sub) if sub else None,
            "delay_events_median": float(np.median([r["delay_events"] for r in held])) if held else None,
            "delay_h_median": float(np.median([r["delay_h"] for r in held])) if held else None,
            "n_substituted_before_window": sum(not r["onset_in_window"] for r in recs),
            "held_before_window_substituted_any_time": sum((not r["onset_in_window"]) and r["first_hold_t_h"] is not None for r in recs)}
    return summ, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(42, 52)))
    ap.add_argument("--arms", nargs="*", default=["uniform", "clustered"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--load-models", type=Path, default=None,
                    help="checkpoint directory (e.g. results/models_v1/session_1); verified, no retraining")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file; REQUIRED with --load-models so the frozen V1 reference is never touched")
    args = ap.parse_args()
    global OUT
    if args.load_models is not None:
        if args.out is None or args.out.resolve() == OUT.resolve():
            raise SystemExit("--load-models requires --out pointing to a NEW file (V1 reference is frozen)")
    if args.out is not None:
        OUT = args.out if args.out.is_absolute() else REPO / args.out
    log = setup_logging(); config = load_config()
    archive = json.loads(ARCHIVE.read_text())
    out = json.loads(OUT.read_text()) if (args.resume and OUT.exists()) else \
        {"spec": "ROUND4_SPEC.md rev 2, V1", "archive": str(ARCHIVE.name), "runs": {}}
    rlr.ProductStateMachine = InstrumentedMachine          # interface + bookkeeping only
    t0 = time.time()
    sessions = out.setdefault("model_sessions", [])
    session = len(sessions)
    if args.load_models is not None:
        mdir = args.load_models if args.load_models.is_absolute() else REPO / args.load_models
        models, info = load_models(mdir, config, log)
        sessions.append({"session": session, "started": time.strftime("%Y-%m-%d %H:%M:%S"), **info})
    else:
        models = train_models(config, log)
        sessions.append({"session": session, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "fingerprints": fingerprint_and_save(models, config, log, session)})
    if session and "fingerprints" in sessions[session] and "fingerprints" in sessions[0]:   # same procedure retrained; record whether outputs of the models match session 0
        f0, f1 = sessions[0]["fingerprints"], sessions[session]["fingerprints"]
        sessions[session]["probe_outputs_match_session_0"] = {
            "cnn": f0["cnn_published"]["probe_p_authentic_sha256"] == f1["cnn_published"]["probe_p_authentic_sha256"],
            "if": f0["if_published"]["probe_scores_sha256"] == f1["if_published"]["probe_scores_sha256"]}
    for arm in args.arms:
        clustered = arm == "clustered"
        old = {r["seed"]: r for r in archive["arms"][arm]}
        for seed in args.seeds:
            key = f"{arm}/{seed}"
            if key in out["runs"]:
                continue
            CALL_LOG.clear()
            tx = build_lifecycle_transactions(config, seed, clustered)
            r = rlr.run_seed(config, seed, tx, models["cnn_published"], models["if_published"],
                             models["if_published_calibration"], rlr.N_NORMAL)
            diffs = compare(old[seed], json.loads(json.dumps(r, default=float)))
            summ, recs = interception(config, seed, clustered)
            out["runs"][key] = {"model_session": session, "archived_summaries_agree": not diffs, "n_differences": len(diffs),
                                "differences": diffs[:200], "interception": summ, "products": recs}
            log.info("[%s] seed %d | archived summaries agree %s (%d diffs) | held at dispense %d/%d",
                     arm, seed, not diffs, len(diffs), summ["held_at_dispense"], summ["n_substituted_in_window"])
            out["runtime_s"] = time.time() - t0
            OUT.write_text(json.dumps(out, indent=1, default=float))
    requested = [f"{a}/{s}" for a in args.arms for s in args.seeds]
    full = [f"{a}/{s}" for a in ("uniform", "clustered") for s in range(42, 52)]
    missing_req = [k for k in requested if k not in out["runs"]]
    bad = [k for k, v in out["runs"].items() if not v["archived_summaries_agree"]]
    missing_full = [k for k in full if k not in out["runs"]]
    out["requested_runs_complete_and_agree"] = not missing_req and not [k for k in requested if k in bad]
    out["all_20_runs_complete_and_agree"] = not missing_full and not bad
    out["attribution"] = (("Instrumented rerun with checkpoints loaded from %s (verified; no retraining); "
                           "per-product agreement with the V1 reference is checked separately by check_v2_baseline.py."
                           % args.load_models) if args.load_models is not None else
                          ("Interception measurements come from this instrumented rerun (retrained models); "
                           "agreement is with the archived per-seed SUMMARY outputs only."))
    OUT.write_text(json.dumps(out, indent=1, default=float))
    log.info("requested runs complete and agree: %s | all 20 complete and agree: %s | mismatched: %s | missing: %s",
             out["requested_runs_complete_and_agree"], out["all_20_runs_complete_and_agree"], bad, missing_full)
    if not out["requested_runs_complete_and_agree"] or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
