"""Item 3: separate the two things the 'contamination 0.10 -> 0.05' change moved.

In the repo, IsolationForestConfig.contamination sets BOTH
  (a) the fraction of injected anomalies in the training mixture, and
  (b) sklearn's `contamination` parameter (which only sets offset_, used by
      predict/decision_function; NOT by score_samples).
The detector's alert path uses score_samples -> fixed reference-bound
normalisation -> warning_threshold 0.7. This script varies (a) and (b)
independently with the evaluation episodes held fixed, and adds a plain
temperature rule as an operational comparator. Repo is imported, not modified.

Usage: python3 if_factorial.py <repo_dir> <out_json> [episodes_per_type]
"""
import json
import sys
import time

import numpy as np
from sklearn.ensemble import IsolationForest

REPO = sys.argv[1]
OUT = sys.argv[2]
N_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 100
sys.path.insert(0, REPO)
from ai_modules import isolation_forest_detector as ifd  # noqa: E402

MIXES = [0.0, 0.025, 0.05, 0.10, 0.20]
PARAMS = [0.05, 0.10, "auto"]
SEEDS = list(range(42, 52))
N_TRAIN = 20000
EP_LEN = 60
WARN = 0.7
INTERVAL_MIN = 0.2


def train(mix, param, seed):
    """Copy of ifd.train_isolation_forest with mixture and parameter decoupled."""
    rng = np.random.default_rng(seed)
    n_anom = int(N_TRAIN * mix)
    normal = ifd._generate_normal_readings(N_TRAIN - n_anom, rng)
    blocks, remaining, k = [], n_anom, len(ifd.ANOMALY_TYPES)
    for i, t in enumerate(ifd.ANOMALY_TYPES):
        b = remaining // (k - i)
        remaining -= b
        if b < 2:  # mix = 0.0: no anomalies in training (np.gradient needs >= 2 rows)
            continue
        blocks.append(ifd._apply_anomaly(ifd._generate_normal_readings(b, rng), t, rng))
    X = np.concatenate([normal] + blocks, axis=0)
    rng.shuffle(X)
    m = IsolationForest(n_estimators=100, max_samples=256, contamination=param, random_state=seed)
    m.fit(X)
    return m, X


def episodes(seed):
    """Fixed evaluation episodes (same for every cell of a seed)."""
    rng = np.random.default_rng(seed + 999)
    out = {}
    for t in ifd.ANOMALY_TYPES:
        an, ctl = [], []
        for _ in range(N_EP):
            nb = ifd._generate_normal_readings(EP_LEN, rng)
            an.append(ifd._apply_anomaly(nb, t, rng))
            ctl.append(ifd._generate_normal_readings(EP_LEN, rng))
        out[t] = (an, ctl)
    return out


def first_alert(flags):
    idx = np.flatnonzero(flags)
    return int(idx[0]) if idx.size else -1


def evaluate(scorer, eps):
    res = {}
    for t, (an, ctl) in eps.items():
        det, lat, fp = [], [], []
        for a, c in zip(an, ctl):
            ia = first_alert(scorer(a))
            det.append(ia >= 0)
            if ia >= 0:
                lat.append(ia * INTERVAL_MIN)
            fp.append(first_alert(scorer(c)) >= 0)
        res[t] = {"detection": float(np.mean(det)), "control_alarm_rate": float(np.mean(fp)),
                  "latency_min": float(np.mean(lat)) if lat else None,
                  "n_anomalous_episodes": len(an), "n_control_episodes": len(ctl)}
    res["overall"] = {
        "detection": float(np.mean([res[t]["detection"] for t in eps])),
        "control_alarm_rate": float(np.mean([res[t]["control_alarm_rate"] for t in eps])),
    }
    return res


def dump(cells, invariance, rule_rows, t0, done):
    json.dump({"design": {"mixes": MIXES, "params": PARAMS, "seeds": SEEDS, "n_train": N_TRAIN,
                          "episodes_per_type": N_EP, "episode_len": EP_LEN, "warning_threshold": WARN},
               "seeds_completed": done,
               "score_samples_max_abs_diff_across_params": max(invariance) if invariance else None,
               "cells": cells, "temperature_rule": rule_rows,
               "runtime_s": time.time() - t0}, open(OUT, "w"), indent=1)


def main():
    import os
    t0 = time.time()
    cells, invariance, rule_rows, done = [], [], [], []
    if os.path.exists(OUT):  # resume: keep completed seeds, skip them
        prev = json.load(open(OUT))
        cells, rule_rows = prev.get("cells", []), prev.get("temperature_rule", [])
        done = prev.get("seeds_completed", [])
        d = prev.get("score_samples_max_abs_diff_across_params")
        invariance = [d] if d is not None else []
        print(f"resuming; already done: {done}", flush=True)
    for seed in SEEDS:
        if seed in done:
            continue
        eps = episodes(seed)
        # temperature rule: alert when temperature leaves the 2-8 C band
        rule = evaluate(lambda X: (X[:, 0] > 8.0) | (X[:, 0] < 2.0), eps)
        rule_rows.append({"seed": seed, **{k: v for k, v in rule.items()}})
        for mix in MIXES:
            ref = None
            for param in PARAMS:
                m, _ = train(mix, param, seed)
                cal = ifd.calibrate_score_normalization(m, ifd.IsolationForestConfig(), seed=seed + 1000)
                r = evaluate(lambda X: ifd.normalized_anomaly_scores(m, X, cal) > WARN, eps)
                probe = ifd._generate_normal_readings(500, np.random.default_rng(7))
                ss = m.score_samples(probe)
                if ref is None:
                    ref = ss
                else:
                    invariance.append(float(np.max(np.abs(ss - ref))))
                cells.append({"seed": seed, "train_mix": mix, "sklearn_contamination": param,
                              "calibration": list(cal), "results": r})
        done.append(seed)
        dump(cells, invariance, rule_rows, t0, done)  # save after every seed
        print(f"seed {seed} done {time.time()-t0:.0f}s", flush=True)
    print("ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
