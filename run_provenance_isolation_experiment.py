"""Execute the pre-registered provenance-isolation experiment (ARM6a/ARM6b).

Protocol: preregistration/OPTION4_provenance_isolation_PREREGISTRATION.txt
(approved 2026-08-23). Runs exactly once.

ARM6a  CONTROL   proxy design, CLUSTERED injection, evaluation half only
ARM6b  TREATMENT live provenance (S1) + proxy S8, same data condition

The two arms share one random stream. The proxy custody draw is taken in
ARM6b as well and then discarded, so the cnn and iso draws that feed S8 are
bit-identical across arms. Provenance is therefore the only channel that
differs -- which is the entire point of the experiment.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.live_inference import (
    compute_org_reputation,
    compute_provenance_scores,
    split_timeline,
)
from pts.product_trust_score import ProductState, compute_pts
from simulation.anomaly_injector import inject_anomalies
from simulation.transaction_generator import generate_transactions
from utils import load_config

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
OUT_PATH = Path(__file__).resolve().parent / "results" / "provenance_isolation.json"

# Measured previously, not re-run (protocol section 3).
ARM0B_MEAN = 0.7219
ARM4_MEAN = 0.9229


def build_transactions(config: Dict[str, Any], seed: int, clustered: bool):
    sim = config["simulation"]
    tx = generate_transactions(sim["transactions_per_run"], sim["organizations"], seed=seed)
    return inject_anomalies(
        tx, sim["anomaly_counts"], seed=seed,
        cluster_counterfeit_by_manufacturer=clustered,
        counterfeit_cluster_fraction=config.get("data_quality", {}).get(
            "counterfeit_cluster_fraction", 0.3),
    )


def score_rows(rows, pts_cfg, rng, s1_override=None) -> Dict[str, Any]:
    """Score one subset under the proxy design, optionally overriding S1.

    Args:
        rows: Transaction subset to score.
        pts_cfg: The ``pts`` configuration section.
        rng: Shared random stream. Draws occur in a fixed order regardless
            of ``s1_override`` so both arms consume the stream identically.
        s1_override: Live provenance scores aligned to ``rows``. When None,
            the proxy custody formula is used (ARM6a).

    Returns:
        Detection and recall metrics for this subset.
    """
    n = len(rows)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "detection": nan, "recall": nan}

    severity_norm = (rows["anomaly_severity"].to_numpy() - 0.4) / 0.6

    # Draw order is fixed. custody is drawn even when overridden, so the
    # cnn and iso streams below are identical across arms.
    custody_proxy = np.clip(1.0 - 0.90 * severity_norm + rng.normal(0, 0.05, n), 0, 1)
    cnn = np.clip(1.0 - 0.30 * severity_norm + rng.normal(0, 0.06, n), 0, 1)
    iso = np.clip(rng.normal(0.38, 0.13, n), 0, 1)

    custody = custody_proxy if s1_override is None else np.clip(s1_override, 0, 1)

    drug_classes = rows["drug_class"].to_numpy()
    bct = np.empty(n)
    for i in range(n):
        st = ProductState(
            custody_chain_trust_scores=[float(custody[i])],
            temperature_readings_c=[],
            cnn_authenticity_score=float(cnn[i]),
            isolation_forest_anomaly_score=float(iso[i]),
        )
        dc = drug_classes[i] if drug_classes[i] in ("A", "B", "C") else "C"
        cc = pts_cfg["drug_classes"][dc]
        bct[i] = compute_pts(st, {
            "provenance_integrity": cc["w1_provenance_integrity"],
            "ai_confidence": cc["w8_ai_confidence"],
        })["pts"]

    alert = pts_cfg["alert_threshold"]
    quarantine = pts_cfg["quarantine_threshold"]
    return {
        "n": n,
        "detection": float(np.mean(bct < alert)),
        "recall": float(np.mean(bct < quarantine)),
        "pts_mean": float(np.mean(bct)),
        "s1_mean": float(np.mean(custody)),
    }


def run_seed(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Run both arms on one seed, sharing the transaction table."""
    pts_cfg = config["pts"]
    tx = build_transactions(config, seed, clustered=True)
    burn_in, evaluation = split_timeline(tx)

    trust, base_rate = compute_org_reputation(burn_in)
    s1_live_all = compute_provenance_scores(evaluation, trust)

    counterfeit_mask = (evaluation["anomaly_type"] == "counterfeit_product").to_numpy()
    cf_rows = evaluation.loc[counterfeit_mask]
    s1_live_cf = s1_live_all[counterfeit_mask]

    # Independent streams, identical seeding, so each arm sees the same draws.
    a = score_rows(cf_rows, pts_cfg, np.random.default_rng(seed + 10_000), None)
    b = score_rows(cf_rows, pts_cfg, np.random.default_rng(seed + 10_000), s1_live_cf)

    return {
        "seed": seed,
        "n_evaluation_transactions": int(len(evaluation)),
        "n_counterfeit": a["n"],
        "base_rate_burn_in": float(base_rate),
        "arm6a": a,
        "arm6b": b,
        "paired_difference_pp": (b["detection"] - a["detection"]) * 100,
    }


def main() -> None:
    t0 = time.time()
    print("#" * 74)
    print("PRE-REGISTERED PROVENANCE-ISOLATION EXPERIMENT — single execution")
    print("Protocol: OPTION4_provenance_isolation_PREREGISTRATION.txt (2026-08-23)")
    print("POST-HOC relative to OPTION2. Reported as such regardless of outcome.")
    print("#" * 74)

    config = load_config()
    per_seed: List[Dict[str, Any]] = []

    for seed in SEEDS:
        r = run_seed(config, seed)
        per_seed.append(r)
        print("  seed %d: ARM6a=%.4f  ARM6b=%.4f  diff=%+.2f pp  (n_cf=%d)" % (
            seed, r["arm6a"]["detection"], r["arm6b"]["detection"],
            r["paired_difference_pp"], r["n_counterfeit"]))

    a_vals = np.array([r["arm6a"]["detection"] for r in per_seed])
    b_vals = np.array([r["arm6b"]["detection"] for r in per_seed])
    diffs = b_vals - a_vals

    arm6a_mean, arm6b_mean = float(a_vals.mean()), float(b_vals.mean())
    d_prov = arm6b_mean - arm6a_mean
    d_total = ARM4_MEAN - arm6a_mean

    # --- H6.2 positive control, evaluated FIRST (protocol section 2) ---
    h62_gap_pp = abs(arm6a_mean - ARM0B_MEAN) * 100
    h62_holds = bool(h62_gap_pp < 2.0)

    # --- Power guard (protocol section 4) ---
    guard_tripped = bool(abs(d_total * 100) < 2.0)
    share = None if guard_tripped else float(d_prov / d_total)

    # --- H6.1 primary test ---
    t_stat, p_val = scipy_stats.ttest_rel(b_vals, a_vals)
    sd_diff = float(diffs.std(ddof=1))
    n = len(diffs)
    ci_half = 1.96 * sd_diff / np.sqrt(n) if sd_diff > 0 else 0.0

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "OPTION4_provenance_isolation_PREREGISTRATION.txt",
        "post_hoc_relative_to": "OPTION2 (approved 2026-07-28)",
        "seeds": SEEDS,
        "transactions_per_run": config["simulation"]["transactions_per_run"],
        "per_seed": per_seed,
        "arm6a_mean_detection": arm6a_mean,
        "arm6a_sd": float(a_vals.std(ddof=1)),
        "arm6b_mean_detection": arm6b_mean,
        "arm6b_sd": float(b_vals.std(ddof=1)),
        "reference_arm0b_mean": ARM0B_MEAN,
        "reference_arm4_mean": ARM4_MEAN,
        "d_prov_pp": d_prov * 100,
        "d_total_pp": d_total * 100,
        "share": share,
        "power_guard_tripped": guard_tripped,
        "h62_positive_control": {
            "gap_vs_arm0b_pp": h62_gap_pp,
            "threshold_pp": 2.0,
            "holds": h62_holds,
        },
        "h61_test": {
            "paired_t": float(t_stat),
            "df": n - 1,
            "p_value": float(p_val),
            "mean_diff_pp": float(diffs.mean() * 100),
            "ci95_diff_pp": [float((diffs.mean() - ci_half) * 100),
                             float((diffs.mean() + ci_half) * 100)],
            "predicted_share_gt": 0.50,
            "holds": None if guard_tripped else bool(share > 0.50),
        },
        "duration_seconds": round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 74)
    print("H6.2 positive control: ARM6a=%.2f%% vs ARM0b=%.2f%% | gap=%.2f pp | %s" % (
        arm6a_mean * 100, ARM0B_MEAN * 100, h62_gap_pp,
        "HOLDS" if h62_holds else "*** FALSIFIED — H6.1 NOT INTERPRETABLE ***"))
    print("=" * 74)
    print("ARM6a (proxy, clustered):        %.2f%% +/- %.2f" % (
        arm6a_mean * 100, a_vals.std(ddof=1) * 100))
    print("ARM6b (live provenance only):    %.2f%% +/- %.2f" % (
        arm6b_mean * 100, b_vals.std(ddof=1) * 100))
    print("ARM4  (full redesign, measured): %.2f%%" % (ARM4_MEAN * 100))
    print("-" * 74)
    print("d_prov  = %+.2f pp" % (d_prov * 100))
    print("d_total = %+.2f pp" % (d_total * 100))
    if guard_tripped:
        print("*** POWER GUARD TRIPPED: |d_total| < 2.0 pp — share NOT reported ***")
    else:
        print("share   = %.3f  (predicted > 0.50 -> %s)" % (
            share, "CONFIRMED" if share > 0.50 else "FALSIFIED"))
    print("paired t(%d) = %.3f, p = %.6f" % (n - 1, t_stat, p_val))
    print("mean paired diff = %+.2f pp, 95%% CI [%+.2f, %+.2f]" % (
        diffs.mean() * 100, (diffs.mean() - ci_half) * 100, (diffs.mean() + ci_half) * 100))
    print("=" * 74)
    print("Written to %s (%.1fs)" % (OUT_PATH, time.time() - t0))


if __name__ == "__main__":
    main()
