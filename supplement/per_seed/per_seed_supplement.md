# Per-seed supplement

Built 5 November 2026 from the saved result records in the repository's `results/` directory. Records were only read; no model or simulation was run.

- `per_seed_supplement.csv`: one row per observation, with its source file.
- `sources_sha256.txt`: hashes of the source records.
- `checks.json`: recomputed summaries.
- To rebuild: `python3 -I build_per_seed_supplement.py <repo>/results <out_dir>`, then `python3 -I make_supplement_md.py`.

Detection values are the percentage of counterfeit rows that raise an alert at τ = 0.75. All seeds are simulation seeds. Each model was trained from one fixed training seed (42), and the per-transaction field of classifier and Isolation Forest inputs is reused across seeds. These values therefore show variation in simulation, injection and evaluation only.

## S1. Ablation (Table 2): integrated detection per seed (%)

| Seed | Baseline | Data quality | IF setting | LSTM search | CNN overlap | Combined |
|---|---|---|---|---|---|---|
| 42 | 70.87 | 70.83 | 70.87 | 70.87 | 70.87 | 70.83 |
| 43 | 70.13 | 70.74 | 70.13 | 70.13 | 70.13 | 70.74 |
| 44 | 73.00 | 73.52 | 73.00 | 73.00 | 73.00 | 73.52 |
| 45 | 73.65 | 71.22 | 73.65 | 73.65 | 73.65 | 71.22 |
| 46 | 71.91 | 70.52 | 71.91 | 71.91 | 71.91 | 70.52 |
| 47 | 73.13 | 74.30 | 73.13 | 73.13 | 73.13 | 74.30 |
| 48 | 72.39 | 70.65 | 72.39 | 72.39 | 72.39 | 70.65 |
| 49 | 71.17 | 70.61 | 71.17 | 71.17 | 71.17 | 70.61 |
| 50 | 72.78 | 71.09 | 72.78 | 72.78 | 72.78 | 71.09 |
| 51 | 70.09 | 70.57 | 70.09 | 70.09 | 70.09 | 70.57 |

Phases 3–5 equal the baseline on every seed. Analytics-only values are in the CSV.

## S2. Redesign arms (Table 4), per seed (%)

### Integrated detection

| Seed | Arm 0 | Arm 0b | Arm 1 | Arm 2 | Arm 3 | Arm 4 | Arm 5 |
|---|---|---|---|---|---|---|---|
| 42 | 70.87 | 71.60 | 0.00 | 0.00 | 0.00 | 91.32 | 87.55 |
| 43 | 70.13 | 70.34 | 0.00 | 0.00 | 0.00 | 92.35 | 87.11 |
| 44 | 73.00 | 73.17 | 0.00 | 0.00 | 0.00 | 92.83 | 88.43 |
| 45 | 73.65 | 73.41 | 0.00 | 0.00 | 0.00 | 91.56 | 88.00 |
| 46 | 71.91 | 71.01 | 0.00 | 0.00 | 0.00 | 92.70 | 87.36 |
| 47 | 73.13 | 74.24 | 0.00 | 0.00 | 0.00 | 92.63 | 88.47 |
| 48 | 72.39 | 72.44 | 0.00 | 0.00 | 0.00 | 91.27 | 86.03 |
| 49 | 71.17 | 71.18 | 0.00 | 0.00 | 0.00 | 92.70 | 87.49 |
| 50 | 72.78 | 72.14 | 0.00 | 0.00 | 0.00 | 92.80 | 86.90 |
| 51 | 70.09 | 72.42 | 0.00 | 0.00 | 0.00 | 92.74 | 87.99 |

### Analytics-only detection

| Seed | Arm 0 | Arm 0b | Arm 1 | Arm 2 | Arm 3 | Arm 4 | Arm 5 |
|---|---|---|---|---|---|---|---|
| 42 | 57.09 | 55.50 | 73.51 | 71.17 | 73.77 | 74.67 | 70.82 |
| 43 | 57.35 | 56.98 | 74.50 | 70.25 | 74.85 | 75.86 | 70.70 |
| 44 | 59.91 | 59.48 | 77.55 | 73.70 | 77.73 | 76.86 | 74.26 |
| 45 | 59.04 | 58.33 | 75.07 | 72.62 | 75.94 | 73.96 | 71.11 |
| 46 | 57.91 | 57.97 | 74.63 | 71.19 | 74.98 | 74.53 | 69.90 |
| 47 | 59.00 | 61.02 | 76.23 | 73.03 | 76.66 | 75.13 | 72.44 |
| 48 | 55.87 | 57.77 | 73.50 | 72.17 | 74.12 | 74.15 | 71.27 |
| 49 | 58.09 | 56.26 | 75.50 | 72.04 | 75.93 | 74.46 | 71.24 |
| 50 | 57.61 | 58.16 | 75.43 | 72.22 | 75.87 | 76.06 | 70.08 |
| 51 | 57.17 | 56.83 | 74.96 | 71.72 | 75.22 | 74.68 | 70.44 |

### Evaluated counterfeit rows

| Seed | Arm 1 (uniform) | Arm 4 (clustered) | Arm 5 (clustered) |
|---|---|---|---|
| 42 | 1,155 | 1,141 | 1,141 |
| 43 | 1,153 | 1,164 | 1,164 |
| 44 | 1,118 | 1,115 | 1,115 |
| 45 | 1,147 | 1,125 | 1,125 |
| 46 | 1,135 | 1,123 | 1,123 |
| 47 | 1,157 | 1,154 | 1,154 |
| 48 | 1,132 | 1,145 | 1,145 |
| 49 | 1,159 | 1,151 | 1,151 |
| 50 | 1,152 | 1,153 | 1,153 |
| 51 | 1,142 | 1,157 | 1,157 |

Arm 4 was re-executed in the run that added Arm 5. Its per-seed values are identical in both records.

## S3. Model-swap contrast: Arm 4 minus Arm 5 (pp)

| Seed | Difference |
|---|---|
| 42 | +3.77 |
| 43 | +5.24 |
| 44 | +4.39 |
| 45 | +3.56 |
| 46 | +5.34 |
| 47 | +4.16 |
| 48 | +5.24 |
| 49 | +5.21 |
| 50 | +5.90 |
| 51 | +4.75 |

Mean 4.757 pp; t(9) = 19.74; d_z = 6.24; 95% CI [4.21, 5.30] (`checks.json`). The difference is positive in all ten seeds.

## S4. Operating point (Table 5), per seed (%)

| Seed | Arm 4 TPR | Arm 4 FPR | Arm 5 TPR | Arm 5 FPR |
|---|---|---|---|---|
| 42 | 91.32 | 12.49 | 87.55 | 12.48 |
| 43 | 92.35 | 12.30 | 87.11 | 12.29 |
| 44 | 92.83 | 12.26 | 88.43 | 12.25 |
| 45 | 91.56 | 12.25 | 88.00 | 12.24 |
| 46 | 92.70 | 12.46 | 87.36 | 12.45 |
| 47 | 92.63 | 12.25 | 88.47 | 12.24 |
| 48 | 91.27 | 12.34 | 86.03 | 12.33 |
| 49 | 92.70 | 12.44 | 87.49 | 12.43 |
| 50 | 92.80 | 12.06 | 86.90 | 12.05 |
| 51 | 92.74 | 12.58 | 87.99 | 12.56 |

## S5. Provenance channel isolated (Section VII-E): detection per seed (%)

| Seed | Detection |
|---|---|
| 42 | 98.33 |
| 43 | 98.11 |
| 44 | 99.64 |
| 45 | 98.93 |
| 46 | 99.38 |
| 47 | 99.05 |
| 48 | 98.69 |
| 49 | 98.52 |
| 50 | 99.05 |
| 51 | 98.44 |

The record gives detection (TPR) only. The 40.78% FPR quoted in the submitted manuscript appears in `fpr_diagnostic.json` only as text in its `prompted_by` field, not as a per-seed result, so the supplement does not report it.

## S6. False-positive repairs (Table 6): change in matched-TPR FPR, treatment minus control (pp)

Pre-run documentation, as stated in Section VII-E:
- **Shrinkage:** pre-registered (protocol committed before execution).
- **Shortcut routing, arithmetic mean, unclipped trust:** self-dated precommitments, first committed with their results.
- **Geometric mean:** exploratory (no pre-run plan located).

### Geometric mean (exploratory): design seeds

| Seed | Change |
|---|---|
| 42 | +1.26 |
| 43 | +3.39 |
| 44 | -0.06 |
| 45 | -0.05 |
| 46 | +0.83 |
| 47 | +2.04 |
| 48 | +1.77 |
| 49 | +2.20 |
| 50 | +1.64 |
| 51 | +2.30 |

### Geometric mean (exploratory): held-out seeds

| Seed | Change |
|---|---|
| 52 | +1.52 |
| 53 | +0.53 |
| 54 | +3.07 |
| 55 | +3.04 |
| 56 | +1.70 |

### Shortcut routing: design seeds

| Seed | Change |
|---|---|
| 42 | +4.59 |
| 43 | +4.89 |
| 44 | +4.47 |
| 45 | +3.86 |
| 46 | +4.68 |
| 47 | +4.83 |
| 48 | +4.83 |
| 49 | +4.72 |
| 50 | +4.68 |
| 51 | +4.73 |

### Shortcut routing: held-out seeds

| Seed | Change |
|---|---|
| 57 | +3.14 |
| 58 | +3.21 |
| 59 | +2.38 |
| 60 | +4.61 |
| 61 | +4.28 |

### Arithmetic mean: design block (protocol: 42–51)

| Record position | Protocol-specified seed | Change |
|---|---|---|
| 1 | 42 | +0.65 |
| 2 | 43 | +2.95 |
| 3 | 44 | -0.50 |
| 4 | 45 | -0.29 |
| 5 | 46 | +0.35 |
| 6 | 47 | +1.71 |
| 7 | 48 | +0.88 |
| 8 | 49 | +1.59 |
| 9 | 50 | +1.03 |
| 10 | 51 | +1.51 |

### Arithmetic mean: held-out block (protocol: 62–66)

| Record position | Protocol-specified seed | Change |
|---|---|---|
| 1 | 62 | +1.23 |
| 2 | 63 | +1.43 |
| 3 | 64 | +1.15 |
| 4 | 65 | +3.87 |
| 5 | 66 | +0.50 |

### Unclipped trust: design block (protocol: 42–51)

| Record position | Protocol-specified seed | Change |
|---|---|---|
| 1 | 42 | +7.47 |
| 2 | 43 | +11.46 |
| 3 | 44 | +9.67 |
| 4 | 45 | +8.20 |
| 5 | 46 | +11.14 |
| 6 | 47 | +10.29 |
| 7 | 48 | +9.58 |
| 8 | 49 | +11.03 |
| 9 | 50 | +9.75 |
| 10 | 51 | +10.78 |

### Unclipped trust: held-out block (protocol: 67–71)

| Record position | Protocol-specified seed | Change |
|---|---|---|
| 1 | 67 | +11.44 |
| 2 | 68 | +11.20 |
| 3 | 69 | +9.27 |
| 4 | 70 | +10.78 |
| 5 | 71 | +6.12 |

### Shrinkage: design seeds (no held-out evaluation)

| Seed | Change |
|---|---|
| 42 | +0.00 |
| 43 | +0.00 |
| 44 | +0.00 |
| 45 | +0.00 |
| 46 | +0.00 |
| 47 | +0.00 |
| 48 | +0.00 |
| 49 | +0.00 |
| 50 | +0.00 |
| 51 | +0.00 |

**Seed labels for the arithmetic-mean and unclipped-trust records.** These records store their per-seed observations without a seed field, so the tables show them in record order. The protocol-specified seed is listed separately. The evidence for that pairing:
- `analyze_option9_cached.py` and `analyze_option10_cached.py` iterate the `--seeds` range in ascending order.
- Each record's `n` equals its block size, so no seed was dropped.
- The only cached inputs in the held-out ranges are for seeds 62–66 (created 29 Aug 2026) and 67–71 (created 30 Aug 2026), in `results/signal_cache/`.

The `--seeds` value used for each run is not stored in the record. Block means do not depend on the pairing, and every mean matches Table 6 (`checks.json`).

The shortcut-routing control is that experiment's own registered baseline (`s1_only`).

## Not included
- **Threshold sweep (Table 3):** its record stores the mean and SD per threshold, not per-seed values.
- **Post-review analyses** (N0, N1, N2): their per-seed outputs are in the staged evidence package (`analyses/post_review/`) and are linked, not duplicated.
