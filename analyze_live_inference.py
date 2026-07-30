"""Evaluate the pre-registered gates and hypotheses from the live-inference run.

Written BEFORE the run's results were available, so that the analysis is not
shaped by them. Applies exactly the rules frozen in
``OPTION2_live_inference_PREREGISTRATION.txt`` sections 2 and 6.

Usage:
    python analyze_live_inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results" / "live_inference_results.json"

# Frozen thresholds (pre-registration sections 2 and 6)
H1_THRESHOLD_PP = 1.0
H3_THRESHOLD_PP = 1.0
R1_TARGET, R1_TOLERANCE = 0.714, 0.014
R7_DEGENERATE_MAX = 0.90
R8_MIN_CLEAN_PTS = 0.75


def arm_by_id(data: Dict[str, Any], arm_id: str) -> Optional[Dict[str, Any]]:
    for arm in data["arms"]:
        if arm["arm"] == arm_id:
            return arm
    return None


def mean_diag(arm: Dict[str, Any], key: str) -> Optional[float]:
    vals = [s["diagnostics"][key] for s in arm["per_seed"]
            if "diagnostics" in s and s["diagnostics"].get(key) is not None
            and np.isfinite(s["diagnostics"][key])]
    return float(np.mean(vals)) if vals else None


def fmt(x: Optional[float], pct: bool = False, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x*100:.{dp}f}%" if pct else f"{x:.{dp}f}"


def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS}")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    print("=" * 92)
    print("PRE-REGISTERED LIVE-INFERENCE EXPERIMENT — RESULTS")
    print(f"generated {data['generated_at']} | seeds {data['seeds']} | "
          f"{data['transactions_per_run']:,} tx/seed | {data['total_minutes']} min")
    if data.get("smoke_test"):
        print("*** SMOKE TEST DATA — NOT THE PRE-REGISTERED EXPERIMENT ***")
    print("=" * 92)

    # ---- all arms, no exclusions (reporting obligation, section 7) ----
    print(f"\n{'arm':6s} {'detection':>18s} {'AI-only':>10s} {'advantage':>11s} "
          f"{'recall':>9s} {'n_cf':>7s} {'seeds':>6s}")
    print("-" * 92)
    for arm in data["arms"]:
        s = arm["summary"]
        n_cf = int(np.mean([x["n_counterfeit"] for x in arm["per_seed"]])) if arm["per_seed"] else 0
        det = f"{fmt(s['detection_mean'], True)} ± {fmt(s.get('detection_std'), True)}"
        print(f"{arm['arm']:6s} {det:>18s} {fmt(s['ai_only_detection_mean'], True):>10s} "
              f"{(f'{s[chr(39)+chr(39)]}' if False else fmt((s['bct_ai_advantage_pp'] or 0)/100, True)):>11s} "
              f"{fmt(s['recall_mean'], True):>9s} {n_cf:>7,} {s['n_seeds']:>6d}")
        if arm["failures"]:
            print(f"       !! {len(arm['failures'])} seed failure(s)")

    a0, a0b = arm_by_id(data, "ARM0"), arm_by_id(data, "ARM0b")
    a1, a2 = arm_by_id(data, "ARM1"), arm_by_id(data, "ARM2")
    a3, a4 = arm_by_id(data, "ARM3"), arm_by_id(data, "ARM4")

    # ---- R1 harness parity ----
    print("\n" + "=" * 92)
    print("GATES")
    print("=" * 92)
    d0 = a0["summary"]["detection_mean"] if a0 else None
    r1 = d0 is not None and abs(d0 - R1_TARGET) <= R1_TOLERANCE * 2
    print(f"R1 harness parity   ARM0 = {fmt(d0, True)} vs published "
          f"{R1_TARGET*100:.1f}% ± {R1_TOLERANCE*100:.1f}%  -> "
          f"{'PASS' if r1 else 'FAIL — NOTHING ELSE IS INTERPRETED'}")

    # ---- R7 degeneracy (live) ----
    dg1 = mean_diag(a1, "cnn_degenerate_mass") if a1 else None
    sc1 = mean_diag(a1, "cnn_severity_corr") if a1 else None
    r7 = (dg1 is not None and dg1 <= R7_DEGENERATE_MAX) and (sc1 is not None and sc1 < -0.2)
    print(f"R7 CNN degeneracy   mass = {fmt(dg1, True)} (fails >90%), "
          f"corr(P_auth,severity) = {fmt(sc1)}  -> {'PASS' if r7 else 'FAIL — R4 VOID'}")

    # ---- R8 provenance degeneracy ----
    print(f"R8 provenance       ARM1 S1(cf) = {fmt(mean_diag(a1,'s1_mean_counterfeit'))}, "
          f"S1(other) = {fmt(mean_diag(a1,'s1_mean_other'))} | "
          f"ARM4 S1(cf) = {fmt(mean_diag(a4,'s1_mean_counterfeit'))}, "
          f"S1(other) = {fmt(mean_diag(a4,'s1_mean_other'))}")

    # ---- H1 causal coupling ----
    print("\n" + "=" * 92)
    print("HYPOTHESES")
    print("=" * 92)
    d1 = a1["summary"]["detection_mean"] if a1 else None
    d2 = a2["summary"]["detection_mean"] if a2 else None
    d3 = a3["summary"]["detection_mean"] if a3 else None
    for label, other in (("CNN  (ARM1 vs ARM2)", d2), ("IF   (ARM1 vs ARM3)", d3)):
        if d1 is None or other is None:
            print(f"H1 {label}: N/A"); continue
        delta = abs(d1 - other) * 100
        print(f"H1 {label}: |{fmt(d1,True)} - {fmt(other,True)}| = {delta:.2f} pp  "
              f"-> {'CONFIRMED' if delta > H1_THRESHOLD_PP else 'FALSIFIED'} "
              f"(threshold {H1_THRESHOLD_PP} pp)")

    # H1 on the AI-Only channel, which is where CNN influence is unmasked
    ao1 = a1["summary"]["ai_only_detection_mean"] if a1 else None
    ao2 = a2["summary"]["ai_only_detection_mean"] if a2 else None
    if ao1 is not None and ao2 is not None:
        d_ao = abs(ao1 - ao2) * 100
        print(f"   [diagnostic, not the registered test] AI-Only channel CNN sensitivity: "
              f"|{fmt(ao1,True)} - {fmt(ao2,True)}| = {d_ao:.2f} pp")

    # ---- H3 clustering ----
    adv1 = a1["summary"]["bct_ai_advantage_pp"] if a1 else None
    adv4 = a4["summary"]["bct_ai_advantage_pp"] if a4 else None
    if adv1 is not None and adv4 is not None:
        margin = adv4 - adv1
        print(f"H3 provenance-needs-clustering: advantage clustered ({adv4:+.2f} pp) "
              f"- uniform ({adv1:+.2f} pp) = {margin:+.2f} pp  -> "
              f"{'CONFIRMED' if margin > H3_THRESHOLD_PP else 'FALSIFIED'} "
              f"(threshold {H3_THRESHOLD_PP} pp)")

    # ---- matched control ----
    d0b = a0b["summary"]["detection_mean"] if a0b else None
    if d0b is not None and d1 is not None:
        print(f"\nMatched control: proxy on evaluation half (ARM0b) = {fmt(d0b, True)} "
              f"vs live (ARM1) = {fmt(d1, True)}  -> shift {(d1-d0b)*100:+.2f} pp")

    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
