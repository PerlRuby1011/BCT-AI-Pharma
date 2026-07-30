"""Verify round-2 finding: is R-C's objective monotone increasing in r,
so that "maximize reachable range" degenerates to r = 1 (i.e. AI-Only)?

Reviewer's analytic claim: the S1-window in which a row can be swung by S8
has width 0.5r/(1-r), zero at r=0 and unbounded as r->1.

Checked here two ways: (a) the closed form, (b) empirically against the real
per-row S1 values, computing actual reachable detection range at each r.
"""
import os, sys
REPO = "/Users/mm/Articles Manusript/CODE/BCT-AI-Pharma"
sys.path.insert(0, REPO); os.chdir(REPO)

import numpy as np
from utils import load_config
from simulation.transaction_generator import generate_transactions
from simulation.anomaly_injector import inject_anomalies
from evaluation.live_inference import (
    split_timeline, compute_org_reputation, compute_provenance_scores)

cfg = load_config("config/config.yaml")
SIM = cfg["simulation"]
ALERT = cfg["pts"]["alert_threshold"]
S8_LO, S8_HI = 0.38, 0.88
SEEDS = list(range(42, 52))


def s1_counterfeit(seed, clustered):
    tx = generate_transactions(SIM["transactions_per_run"], SIM["organizations"], seed=seed)
    tx = inject_anomalies(tx, SIM["anomaly_counts"], seed=seed,
                          cluster_counterfeit_by_manufacturer=clustered,
                          counterfeit_cluster_fraction=0.3)
    burn, ev = split_timeline(tx)
    trust, _ = compute_org_reputation(burn)
    s1 = compute_provenance_scores(ev, trust)
    return s1[(ev["anomaly_type"] == "counterfeit_product").to_numpy()]


print("caching per-row S1 ...", flush=True)
S1 = {m: np.concatenate([s1_counterfeit(s, c) for s in SEEDS])
      for m, c in (("uniform", False), ("clustered", True))}
for m, v in S1.items():
    print(f"  {m}: {len(v):,} counterfeit rows, mean S1 {v.mean():.4f}", flush=True)


def detection(s1, r, s8):
    """PTS = (1-r)*S1 + r*S8 ; detected iff < ALERT."""
    return float(np.mean(((1 - r) * s1 + r * s8) < ALERT))


print(f"\n{'r':>6s} {'window 0.5r/(1-r)':>19s} {'uniform range':>15s} {'clustered range':>16s}")
print("-" * 62)
rows = []
for r in (0.05, 0.10, 0.20, 0.286, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
    window = 0.5 * r / (1 - r)
    out = {}
    for m in ("uniform", "clustered"):
        lo = detection(S1[m], r, S8_HI)   # high S8 -> fewer detected
        hi = detection(S1[m], r, S8_LO)   # low  S8 -> more detected
        out[m] = (hi - lo) * 100
    rows.append((r, window, out["uniform"], out["clustered"]))
    tag = "  <- published Class A" if abs(r - 0.286) < 1e-9 else ""
    print(f"{r:6.3f} {window:19.4f} {out['uniform']:14.2f}pp {out['clustered']:15.2f}pp{tag}")

print("-" * 62)
u = [x[2] for x in rows]; c = [x[3] for x in rows]
print(f"uniform range monotone non-decreasing in r? {all(b >= a - 1e-9 for a, b in zip(u, u[1:]))}")
print(f"argmax r (uniform)  = {rows[int(np.argmax(u))][0]}")
print(f"argmax r (clustered)= {rows[int(np.argmax(c))][0]}")
print()
print("At r = 1.0 the composite is PTS = S8 exactly, i.e. the AI-Only condition:")
for m in ("uniform", "clustered"):
    lo = detection(S1[m], 1.0, S8_HI); hi = detection(S1[m], 1.0, S8_LO)
    print(f"  {m:>10s}: reachable range {(hi-lo)*100:.2f} pp")
