"""Independent verification of review finding 10.2: was the H1 test powered?

Holds S8 constant across its physically reachable range and recomputes
BCT-AI detection from the ACTUAL per-row S1 values, all 10 seeds, both
injection modes. If the reachable range under uniform injection is below
the 1.0 pp H1 threshold, the registered H1 test had zero power.

No CNN inference required.
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
SIM, PTS = cfg["simulation"], cfg["pts"]
ALERT = PTS["alert_threshold"]
SEEDS = list(range(42, 52))


def s1_and_classes(seed, clustered):
    tx = generate_transactions(SIM["transactions_per_run"], SIM["organizations"], seed=seed)
    tx = inject_anomalies(tx, SIM["anomaly_counts"], seed=seed,
                          cluster_counterfeit_by_manufacturer=clustered,
                          counterfeit_cluster_fraction=0.3)
    burn, ev = split_timeline(tx)
    trust, _ = compute_org_reputation(burn)
    s1 = compute_provenance_scores(ev, trust)
    cf = (ev["anomaly_type"] == "counterfeit_product").to_numpy()
    return s1[cf], ev["drug_class"].to_numpy()[cf]


def detection_at(s1, classes, s8):
    """BCT-AI detection with S8 pinned to a constant, using real per-row S1."""
    w1 = np.array([PTS["drug_classes"][c if c in "ABC" else "C"]["w1_provenance_integrity"]
                   for c in classes])
    w8 = np.array([PTS["drug_classes"][c if c in "ABC" else "C"]["w8_ai_confidence"]
                   for c in classes])
    pts = (w1 * s1 + w8 * s8) / (w1 + w8)
    return float(np.mean(pts < ALERT))


print("Recomputing per-row S1 for 10 seeds x 2 injection modes ...", flush=True)
data = {}
for mode, clustered in (("uniform", False), ("clustered", True)):
    data[mode] = [s1_and_classes(s, clustered) for s in SEEDS]
    print(f"  {mode}: done", flush=True)

S8_GRID = [0.38, 0.45, 0.50, 0.52, 0.60, 0.70, 0.80, 0.88]
print(f"\n{'S8':>6s}   {'ARM1 (uniform)':>16s}   {'ARM4 (clustered)':>18s}")
print("-" * 48)
results = {"uniform": [], "clustered": []}
for s8 in S8_GRID:
    row = {}
    for mode in ("uniform", "clustered"):
        det = np.mean([detection_at(s1, cl, s8) for s1, cl in data[mode]])
        results[mode].append(det); row[mode] = det
    print(f"{s8:6.2f}   {row['uniform']*100:15.2f}%   {row['clustered']*100:17.2f}%")

print("-" * 48)
for mode in ("uniform", "clustered"):
    rng = (max(results[mode]) - min(results[mode])) * 100
    verdict = "BELOW 1.0 pp H1 threshold -> ZERO POWER" if rng < 1.0 else "powered"
    print(f"{mode:>10s} total reachable range = {rng:6.2f} pp   {verdict}")
