"""Build the per-seed supplement draft from saved records (read-only).
Usage: python3 -I build_per_seed_supplement.py <repo_results_dir> <out_dir>
Reads JSON records only; runs no model and no simulation."""
import sys, json, csv, hashlib, statistics as st, os, math
R, OUT = sys.argv[1], sys.argv[2]
def load(f): return json.load(open(os.path.join(R, f)))
def sha(f): return hashlib.sha256(open(os.path.join(R, f), "rb").read()).hexdigest()
rows, used, checks = [], set(), []
def add(table, seed, measure, value, source, note="", record_position="", protocol_seed=""):
    rows.append(dict(table=table, seed=seed, record_position=record_position, protocol_seed=protocol_seed, measure=measure, value=value, source=source, note=note)); used.add(source)

# S1 ablation phases
PH = [("phase1_baseline.json","Phase 1 baseline"),("phase2_data_quality.json","Phase 2 data quality"),
      ("phase3_contamination.json","Phase 3 IF contamination"),("phase4_lstm_tuning.json","Phase 4 LSTM search"),
      ("phase5_cnn_overlap.json","Phase 5 CNN overlap"),("phase6_final.json","Phase 6 combined")]
abl = {}
for f, lab in PH:
    d = load(f); abl[lab] = {}
    for r in d["per_run_results"]:
        abl[lab][r["seed"]] = r["bct_ai_counterfeit_detection"]
        add("S1 ablation", r["seed"], f"{lab}: integrated detection", r["bct_ai_counterfeit_detection"], f)
        add("S1 ablation", r["seed"], f"{lab}: analytics-only detection", r["ai_only_counterfeit_detection"], f)
base = abl["Phase 1 baseline"]
for lab in abl:
    diffs = [abl[lab][s]-base[s] for s in base]
    checks.append((f"S1 {lab}: max |per-seed change vs baseline| (pp)", round(100*max(abs(x) for x in diffs),4)))

# S2 redesign arms
li = load("live_inference_results.json"); a5 = load("arm5_h1_powered.json")
arms = {a["arm"]: a for a in li["arms"]}
arms["ARM5"] = [a for a in a5["arms"] if a["arm"]=="ARM5"][0]
rerun4 = [a for a in a5["arms"] if a["arm"]=="ARM4"][0]
for name, a in arms.items():
    src = "arm5_h1_powered.json" if name=="ARM5" else "live_inference_results.json"
    for r in a["per_seed"]:
        add("S2 redesign arms", r["seed"], f"{name}: integrated detection", r["bct_ai_counterfeit_detection"], src)
        add("S2 redesign arms", r["seed"], f"{name}: analytics-only detection", r["ai_only_counterfeit_detection"], src)
        add("S2 redesign arms", r["seed"], f"{name}: evaluated counterfeit rows", r["n_counterfeit"], src)
    m = st.mean(r["bct_ai_counterfeit_detection"] for r in a["per_seed"])
    checks.append((f"S2 {name} mean integrated detection (%)", round(100*m,2)))
same = all(abs(x["bct_ai_counterfeit_detection"]-y["bct_ai_counterfeit_detection"])<1e-15
           for x,y in zip(arms["ARM4"]["per_seed"], rerun4["per_seed"]))
checks.append(("S2 ARM4 rerun (Arm 5 execution) identical per seed to first execution", same))
# S3 paired contrast
d = [100*(x["bct_ai_counterfeit_detection"]-y["bct_ai_counterfeit_detection"])
     for x,y in zip(arms["ARM4"]["per_seed"], arms["ARM5"]["per_seed"])]
for r,v in zip(arms["ARM4"]["per_seed"], d):
    add("S3 Arm 4 minus Arm 5", r["seed"], "integrated detection difference (pp)", v, "live_inference_results.json + arm5_h1_powered.json")
mu, sd = st.mean(d), st.stdev(d); se = sd/math.sqrt(len(d))
checks += [("S3 mean difference (pp)", round(mu,3)), ("S3 t(9)", round(mu/se,2)), ("S3 d_z", round(mu/sd,2)),
           ("S3 95% CI (pp, t=2.262)", (round(mu-2.2622*se,2), round(mu+2.2622*se,2))), ("S3 positive in all seeds", all(x>0 for x in d))]
# S4 FPR diagnostic
fd = load("fpr_diagnostic.json")
for arm in ("ARM4","ARM5"):
    for r in fd["arms"][arm]["per_seed"]:
        for k in ("tpr","fpr"):
            add("S4 operating point", r["seed"], f"{arm}: {k.upper()}", r[k], "fpr_diagnostic.json")
    checks.append((f"S4 {arm} mean FPR (%)", round(100*fd["arms"][arm]["fpr_mean"],2)))
# S5 provenance isolation
pi = load("provenance_isolation.json")
for r in pi["per_seed"]:
    add("S5 provenance isolation", r["seed"], "Arm 6b (provenance channel isolated): detection", r["arm6b"]["detection"], "provenance_isolation.json")
checks.append(("S5 Arm 6b mean detection (%)", round(100*pi["arm6b_mean_detection"],2)))
# S6 repairs: matched-FPR change (treatment - control), pp
def rep(label, f, ctrl, trt, seeds_from_protocol=None):
    q = load(f)
    if "per_seed" in q:
        ps = q["per_seed"]; vals=[]
        for r in ps:
            v = 100*(r[trt]["matched_fpr"]-r[ctrl]["matched_fpr"]); vals.append(v)
            add("S6 FPR repairs", r["seed"], f"{label}: matched-FPR change (pp)", v, f)
    else:
        c = q["C1"]; vals = c["matched_fpr_deltas"]
        assert c["n"] == len(seeds_from_protocol) == len(vals)
        for i,(s,v) in enumerate(zip(seeds_from_protocol, vals), 1):
            add("S6 FPR repairs", "", f"{label}: matched-FPR change (pp)", v, f,
                "record stores no seed field; observations in record order. The analysis script iterates the --seeds range in ascending order and the record's n equals the block size (no seed dropped); the protocol specifies the block; the invoked range is not stored in the record",
                record_position=i, protocol_seed=s)
    checks.append((f"S6 {label} mean (pp)", round(st.mean(vals),2)))
rep("Geometric mean, design", "aggregation_dev.json", "product", "geometric_mean")
rep("Geometric mean, held-out", "aggregation_fresh.json", "product", "geometric_mean")
rep("Shortcut routing, design", "option8_dev.json", "s1_only", "s1_x_deficit")
rep("Shortcut routing, held-out", "option8_heldout.json", "s1_only", "s1_x_deficit")
rep("Arithmetic mean, design", "option9_dev.json", None, None, list(range(42,52)))
rep("Arithmetic mean, held-out", "option9_heldout.json", None, None, list(range(62,67)))
rep("Unclipped trust, design", "option10_dev.json", None, None, list(range(42,52)))
rep("Unclipped trust, held-out", "option10_heldout.json", None, None, list(range(67,72)))
sh = load("shrinkage_experiment.json")["conditions"]["ARM4"]
for b,s_ in zip(sh["baseline"]["per_seed"], sh["shrunk"]["per_seed"]):
    add("S6 FPR repairs", b["seed"], "Shrinkage, design: FPR change (pp, operating point)", 100*(s_["fpr"]-b["fpr"]), "shrinkage_experiment.json")
checks.append(("S6 shrinkage max |FPR change| (pp)", max(abs(100*(s_["fpr"]-b["fpr"])) for b,s_ in zip(sh["baseline"]["per_seed"], sh["shrunk"]["per_seed"]))))
used.add("shrinkage_experiment.json")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT,"per_seed_supplement.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=["table","seed","record_position","protocol_seed","measure","value","source","note"]); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT,"checks.json"),"w") as fh: json.dump(checks, fh, indent=1, default=str)
with open(os.path.join(OUT,"sources_sha256.txt"),"w") as fh:
    for f in sorted(s for s in used if s.endswith(".json") and "+" not in s): fh.write(f"{sha(f)}  results/{f}\n")
print(len(rows), "rows"); [print(c) for c in checks]
