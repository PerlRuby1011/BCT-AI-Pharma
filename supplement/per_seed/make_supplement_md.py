"""Render per_seed_supplement.md from per_seed_supplement.csv (same folder)."""
import csv, collections
rows=list(csv.DictReader(open("per_seed_supplement.csv")))
by=collections.defaultdict(lambda: collections.defaultdict(dict))
pos=collections.defaultdict(list)
for r in rows:
    if r["seed"]: by[r["table"]][r["measure"]][int(r["seed"])]=float(r["value"])
    else: pos[r["measure"]].append((int(r["record_position"]), int(r["protocol_seed"]), float(r["value"])))
def table(name, measures, fmt, title, cols):
    seeds=sorted({s for m in measures for s in by[name][m]})
    out=([f"### {title}\n"] if title else [])+["| Seed | "+" | ".join(cols)+" |", "|"+"---|"*(len(measures)+1)]
    for s in seeds: out.append(f"| {s} | "+" | ".join(fmt(by[name][m][s]) if s in by[name][m] else "—" for m in measures)+" |")
    return "\n".join(out)+"\n"
def ordered(measure, title):
    out=[f"### {title}\n","| Record position | Protocol-specified seed | Change |","|---|---|---|"]
    for p,s,v in sorted(pos[measure]): out.append(f"| {p} | {s} | {v:+.2f} |")
    return "\n".join(out)+"\n"
pct=lambda v:f"{100*v:.2f}"; pp=lambda v:f"{v:+.2f}"; n=lambda v:f"{int(v):,}"
A=["ARM0","ARM0b","ARM1","ARM2","ARM3","ARM4","ARM5"]; AC=["Arm 0","Arm 0b","Arm 1","Arm 2","Arm 3","Arm 4","Arm 5"]
md=["# Per-seed supplement (DRAFT, not yet in the staged package)\n",
"Built 5 November 2026 from the saved result records in the repository's `results/` directory. Records were only read; no model or simulation was run.\n",
"- `per_seed_supplement.csv`: one row per observation, with its source file.\n- `sources_sha256.txt`: hashes of the source records.\n- `checks.json`: recomputed summaries.\n- To rebuild: `python3 -I build_per_seed_supplement.py <repo>/results <out_dir>`, then `python3 -I make_supplement_md.py`.\n",
"Detection values are the percentage of counterfeit rows that raise an alert at τ = 0.75. All seeds are simulation seeds. Each model was trained from one fixed training seed (42), and the per-transaction field of classifier and Isolation Forest inputs is reused across seeds. These values therefore show variation in simulation, injection and evaluation only.\n",
"## S1. Ablation (Table 2): integrated detection per seed (%)\n",
table("S1 ablation",[f"{p}: integrated detection" for p in ["Phase 1 baseline","Phase 2 data quality","Phase 3 IF contamination","Phase 4 LSTM search","Phase 5 CNN overlap","Phase 6 combined"]],pct,"",["Baseline","Data quality","IF setting","LSTM search","CNN overlap","Combined"]),
"Phases 3–5 equal the baseline on every seed. Analytics-only values are in the CSV.\n",
"## S2. Redesign arms (Table 4), per seed (%)\n",
table("S2 redesign arms",[f"{a}: integrated detection" for a in A],pct,"Integrated detection",AC),
table("S2 redesign arms",[f"{a}: analytics-only detection" for a in A],pct,"Analytics-only detection",AC),
table("S2 redesign arms",[f"{a}: evaluated counterfeit rows" for a in ["ARM1","ARM4","ARM5"]],n,"Evaluated counterfeit rows",["Arm 1 (uniform)","Arm 4 (clustered)","Arm 5 (clustered)"]),
"Arm 4 was re-executed in the run that added Arm 5. Its per-seed values are identical in both records.\n",
"## S3. Model-swap contrast: Arm 4 minus Arm 5 (pp)\n",
table("S3 Arm 4 minus Arm 5",["integrated detection difference (pp)"],pp,"",["Difference"]),
"Mean 4.757 pp; t(9) = 19.74; d_z = 6.24; 95% CI [4.21, 5.30] (`checks.json`). The difference is positive in all ten seeds.\n",
"## S4. Operating point (Table 5), per seed (%)\n",
table("S4 operating point",["ARM4: TPR","ARM4: FPR","ARM5: TPR","ARM5: FPR"],pct,"",["Arm 4 TPR","Arm 4 FPR","Arm 5 TPR","Arm 5 FPR"]),
"## S5. Provenance channel isolated (Section VII-E): detection per seed (%)\n",
table("S5 provenance isolation",["Arm 6b (provenance channel isolated): detection"],pct,"",["Detection"]),
"The record gives detection (TPR) only. The 40.78% FPR quoted in the submitted manuscript appears in `fpr_diagnostic.json` only as text in its `prompted_by` field, not as a per-seed result, so the supplement does not report it.\n",
"## S6. False-positive repairs (Table 6): change in matched-TPR FPR, treatment minus control (pp)\n",
"Pre-run documentation, as stated in Section VII-E:\n- **Shrinkage:** pre-registered (protocol committed before execution).\n- **Shortcut routing, arithmetic mean, unclipped trust:** self-dated precommitments, first committed with their results.\n- **Geometric mean:** exploratory (no pre-run plan located).\n",
table("S6 FPR repairs",["Geometric mean, design: matched-FPR change (pp)"],pp,"Geometric mean (exploratory): design seeds",["Change"]),
table("S6 FPR repairs",["Geometric mean, held-out: matched-FPR change (pp)"],pp,"Geometric mean (exploratory): held-out seeds",["Change"]),
table("S6 FPR repairs",["Shortcut routing, design: matched-FPR change (pp)"],pp,"Shortcut routing: design seeds",["Change"]),
table("S6 FPR repairs",["Shortcut routing, held-out: matched-FPR change (pp)"],pp,"Shortcut routing: held-out seeds",["Change"]),
ordered("Arithmetic mean, design: matched-FPR change (pp)","Arithmetic mean: design block (protocol: 42–51)"),
ordered("Arithmetic mean, held-out: matched-FPR change (pp)","Arithmetic mean: held-out block (protocol: 62–66)"),
ordered("Unclipped trust, design: matched-FPR change (pp)","Unclipped trust: design block (protocol: 42–51)"),
ordered("Unclipped trust, held-out: matched-FPR change (pp)","Unclipped trust: held-out block (protocol: 67–71)"),
table("S6 FPR repairs",["Shrinkage, design: FPR change (pp, operating point)"],pp,"Shrinkage: design seeds (no held-out evaluation)",["Change"]),
"""**Seed labels for the arithmetic-mean and unclipped-trust records.** These records store their per-seed observations without a seed field, so the tables show them in record order. The protocol-specified seed is listed separately. The evidence for that pairing:
- `analyze_option9_cached.py` and `analyze_option10_cached.py` iterate the `--seeds` range in ascending order.
- Each record's `n` equals its block size, so no seed was dropped.
- The only cached inputs in the held-out ranges are for seeds 62–66 (created 29 Aug 2026) and 67–71 (created 30 Aug 2026), in `results/signal_cache/`.

The `--seeds` value used for each run is not stored in the record. Block means do not depend on the pairing, and every mean matches Table 6 (`checks.json`).

The shortcut-routing control is that experiment's own registered baseline (`s1_only`).

## Not included
- **Threshold sweep (Table 3):** its record stores the mean and SD per threshold, not per-seed values.
- **Post-review analyses** (N0, N1, N2): their per-seed outputs are in the staged evidence package (`analyses/post_review/`) and are linked, not duplicated.
"""]
open("per_seed_supplement.md","w").write("\n".join(md)); print("ok")
