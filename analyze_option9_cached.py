"""OPTION9 analysis from cached model signals (design seeds).

CNN/IF scores depend only on rows+severity, never on S1, so the cached
scores from OPTION7 (baseline generator) and OPTION8 (shortcut generator)
are valid for any S1 composition. Alignment is re-verified per seed
against the cached product-S1 before any metric is computed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config
from run_aggregation_experiment import build_transactions as build_base, pts_for, roc_auc
from run_product_signal_experiment import build_transactions as build_short
from evaluation.live_inference import (split_timeline, compute_org_reputation,
                                       compute_provenance_scores, compute_hop_deficit_factor)

CFG = load_config(); PTS = CFG["pts"]; ALERT = PTS["alert_threshold"]


def seed_metrics(seed: int, condition: str):
    if condition == "C1":
        tag, key, builder, use_deficit = "arm4", "s1_prod", build_base, False
    else:
        tag, key, builder, use_deficit = "opt8", "s1_base", build_short, True
    f = Path("results/signal_cache") / f"seed{seed}_{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f)

    tx = builder(CFG, seed, True); bi, ev = split_timeline(tx)
    trust, _ = compute_org_reputation(bi)
    cf = (ev["anomaly_type"] == "counterfeit_product").to_numpy()
    cl = ~ev["is_anomaly"].to_numpy()

    s1_prod = compute_provenance_scores(ev, trust)
    assert np.allclose(s1_prod[cf], d[f"{key}_cf"]), f"cache misaligned seed {seed} {tag}"
    s1_mean = compute_provenance_scores(ev, trust, aggregation="mean")
    if use_deficit:
        fac = compute_hop_deficit_factor(bi, ev)
        s1_prod, s1_mean = s1_prod * fac, s1_mean * fac

    out = {}
    stash = {}
    for name, s1 in (("control_product", s1_prod), ("treatment_mean", s1_mean)):
        pcf = pts_for(ev.loc[cf], s1[cf], d["cnn_cf"], d["iso_cf"], PTS)
        pcl = pts_for(ev.loc[cl], s1[cl], d["cnn_clean"], d["iso_clean"], PTS)
        stash[name] = (pcf, pcl)
        allp = np.concatenate([pcf, pcl])
        pos = np.concatenate([np.ones(cf.sum(), bool), np.zeros(cl.sum(), bool)])
        out[name] = {"tpr": float(np.mean(pcf < ALERT)), "fpr": float(np.mean(pcl < ALERT)),
                     "auc": roc_auc(-allp, pos)}
    target = out["control_product"]["tpr"]
    for name in out:
        pcf, pcl = stash[name]
        s = np.sort(pcf); i = min(int(np.ceil(target * len(s))) - 1, len(s) - 1)
        thr = s[max(i, 0)]
        out[name]["matched_tpr"] = float(np.mean(pcf <= thr))
        out[name]["matched_fpr"] = float(np.mean(pcl <= thr))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--out", default="option9_dev.json"); a = ap.parse_args()
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-"); seeds += list(range(int(lo), int(hi) + 1))
        else: seeds.append(int(part))

    allres = {}
    for cond in ("C1", "C2"):
        per = [r for s in seeds if (r := seed_metrics(s, cond)) is not None]
        if not per: continue
        summ = {}
        for arm in ("control_product", "treatment_mean"):
            summ[arm] = {k: float(np.mean([p[arm][k] for p in per]))
                         for k in ("tpr", "fpr", "auc", "matched_fpr")}
        deltas = [(p["treatment_mean"]["matched_fpr"] - p["control_product"]["matched_fpr"]) * 100
                  for p in per]
        allres[cond] = {"n": len(per), "summary": summ, "matched_fpr_deltas": deltas,
                        "per_seed": per}
        lbl = {"C1": "aggregation_only (baseline gen)",
               "C2": "aggregation_plus_signal (shortcut gen + deficit)"}[cond]
        print(f"\n=== {cond}: {lbl} | n={len(per)} ===")
        print(f"{'arm':<18}{'TPR@0.75':>10}{'FPR@0.75':>10}{'AUC':>9}{'FPR@matchedTPR':>16}")
        for arm in ("control_product", "treatment_mean"):
            v = summ[arm]
            print(f"{arm:<18}{v['tpr']*100:>9.2f}%{v['fpr']*100:>9.2f}%{v['auc']:>9.4f}{v['matched_fpr']*100:>15.2f}%")
        dd = np.array(deltas)
        print(f"  matched-FPR delta {dd.mean():+.2f}pp (std {dd.std(ddof=1):.2f}) "
              f"worse on {int((dd>0).sum())}/{len(dd)} seeds")
        print(f"  per-seed: {[round(x,2) for x in deltas]}")
    Path("results", a.out).write_text(json.dumps(allres, indent=2, default=float))
    print(f"\nWritten to results/{a.out}")


if __name__ == "__main__":
    main()
