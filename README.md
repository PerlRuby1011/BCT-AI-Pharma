# BCT-AI-Pharma

Reference implementation of the blockchain–AI pharmaceutical supply chain
system of the BCCA 2026 conference paper, and the records cited by the IEEE
Access article *When the Metric Cannot See the Model* (Access-2026-50542),
mapped in the section
[Records for the IEEE Access methods article](#records-for-the-ieee-access-methods-article-revision).
The archived evidence package is version DOI
[10.5281/zenodo.24097169](https://doi.org/10.5281/zenodo.24097169)
(concept DOI 10.5281/zenodo.21987357).

This public release (1.1.0) contains the code and records needed for that
article, and code shared with a separate manuscript by the same authors.
Experiments, results and tooling used only by that separate manuscript are not
part of this release. Three code files of its coherent-lifecycle replay
(`run_lifecycle_replay.py`, `simulation/lifecycle_generator.py`,
`simulation/lifecycle_invariants.py`) are included only because
`run_v1_interception.py` imports them, and `smoke_load_checkpoints.py`, which
the post-review reproduction uses to load and fingerprint the saved model
checkpoints, imports `run_v1_interception.py`; their results are not included.

## Historical pipeline

`python main.py --config config/config.yaml` and
`python evaluation/run_simulation.py` run the older component/proxy pipeline.
They may train models and overwrite shared result filenames; use a separate
copy if investigating them. The network benchmark samples assumed targets;
it is not a deployed-network benchmark. The counterfeit metric uses severity
proxies and does not establish current live detector effectiveness.

The BCCA 2026 conference figures (96.8% integrated detection; 78.2 / 41.7 /
23.4% baselines; t(9) = 6.42 / 7.11) are conference-reported values that this
code does not reproduce. Earlier localization and throughput values were
stipulated or calculated from assumptions. Historical component results are kept in `results/` as records. The
older full-score/event-stream records remain archived as superseded evidence,
not interchangeable with later evaluations.

## Conference-reported reference values (BCCA 2026) — not computed results

Up to release 1.0.x, `evaluation/baseline_comparison.py` built a "Table VI
baseline comparison" by importing the conference-reported percentages from
`config.yaml` instead of recomputing them from transaction-level predictions,
and a helper generated per-run samples by adding Gaussian jitter around those
values. The approximately 71.4% detection reported by this repository is
calculated separately, from transaction-level scores, with the implemented
proxy-score decision rule. The two are therefore not two measurements of the
same evaluation.

From release 1.1.0:

- All conference-reported values sit in `config.yaml:bcca2026_reference_values`
  and are labelled as reference values, not computed results: the Table VI
  comparison (renamed from `baseline_comparison`), the LSTM/CNN/Isolation Forest
  metrics (formerly `target_metrics`), the paired t-statistics (formerly in
  `statistical_validation`), and the Monte Carlo degradation figures and
  "real-world range" (formerly in `monte_carlo`).
- Computed outputs no longer contain them. The component pipelines return
  `metrics` only, the Monte Carlo results have no `paper_reference` or
  `real_world_range`, and `run_full_pipeline` has no `baseline_comparison` or
  ratio fields. The PTS calibration diagnostic compares each variant with the
  AI-Only detection computed in the same runs instead of the hard-coded 78.2%.
  The notebook plots only computed metrics.
- The historical implementation is preserved, separately, in
  `historical/bcca2026_reference.py`. The table builder and jitter helper have
  the same algorithm, and every return value carries a provenance marker. Run
  `python -m historical.bcca2026_reference` to display the table with its label.
- Checks:
  - (a) Static tests: no current module imports `historical`, reads the
    reference section, or uses the removed keys.
  - (b) A runtime marker check (`evaluation/result_guards.py`) in four
    results writers. It detects the marker string; it does not detect an
    untagged bare number.
  - (c) Stub-based output tests (`tests/test_reference_outputs_stubbed.py`,
    no model training): they check pipeline outputs, including written
    files, for both the marker and the reference numbers themselves.

Archived files under `results/` that were produced before 1.1.0 are kept
unchanged as historical records; any "baseline_comparison" block inside
them contains the conference-reported values, not computed results. See
`docs/releases/1.1.0.md`.

`config/config.yaml` retains historical target fields and larger training-size
settings for traceability. These are not acceptance criteria for current
results. Raising those settings changes the experiment and provides no
assurance of reaching the old targets. The historical notebook follows `main.py`. Dockerfile and the default Compose
service run `main.py` (the historical pipeline); the Docker environment has not
been independently validated for exact model reproduction.

## Records for the IEEE Access methods article (revision)

This section maps each item cited by *When the Metric Cannot See the Model*
(IEEE Access, revised submission) to its location. Original records are
posted alongside the results. Later analyses are kept separate from them.

Paths are relative to the root of this release. In the archived evidence
package this release is the directory `release/1.1.0/`, which holds the
complete tracked tree of this commit (including the tracked `results/` and
`preregistration/` files). The package keeps three other things separately:
the result records as saved, including untracked signal caches, model files
and logs (`records/original/`); the frozen code snapshots used by the analyses
(`code/snapshots/`); and the post-review analyses (`analyses/`).

**Code version used for each analysis.** The current release is not always the
code that produced a record. Where the scripts differ, use the snapshot.

| Records | Produced with | Same scripts in this release? |
|---|---|---|
| Ablation (Table 2), PTS calibration diagnostic, threshold sweep (Table 3), redesign arms and Arm 5 (Table 4) | Uncommitted code of 27–28 July 2026, first committed in efe5c99 (29 July); snapshot efe5c99 | No: `orchestrate.py`, `run_live_inference_experiment.py`, `evaluation/run_simulation.py` and `pts/calibration_diagnostic.py` differ. The script that wrote `results/alert_threshold_sensitivity.json` is not in any commit |
| FPR diagnostic, provenance isolation, shrinkage, geometric mean | Snapshot aaef45c | Yes (identical) |
| Shortcut routing, arithmetic mean | Commits 5dd4c45, 89d4ab2 (scripts committed with their results) | Yes (identical) |
| Unclipped trust | `analyze_option10_cached.py`, first committed in 859e1e9 (6 Oct 2026), after the run of 30 Aug 2026 | Yes, identical to that first commit. The version executed on 30 Aug is not independently established |
| Isolation Forest factorial | Commit 8103bab; snapshot 99c605b | Yes (identical) |
| N0, N1, N2 post-review analyses | Their own code copies under `analyses/post_review/` | Not applicable |

**Protocols, the withdrawn verdict and the internal reviews**

| Item | Location |
|---|---|
| Written protocol for the redesign (hypotheses, gates, decision rules, feasibility run §10a, predictions §13, results §14, added Arm 5 §15); dated 28 July 2026, first committed alongside the results on 29 July 2026 | [`preregistration/OPTION2_live_inference_PREREGISTRATION.txt`](preregistration/OPTION2_live_inference_PREREGISTRATION.txt) |
| Withdrawn verdict and round-1 internal review (§10.2, §10.6; Arm 5 prediction §10.4) | [`preregistration/LIVE_INFERENCE_RESULTS_for_review.txt`](preregistration/LIVE_INFERENCE_RESULTS_for_review.txt) |
| Round-2 internal review | [`preregistration/REVIEW_ROUND2_request.txt`](preregistration/REVIEW_ROUND2_request.txt) |
| Reading guide, and what establishes each document's timing | [`preregistration/README.md`](preregistration/README.md) |

Both internal reviews were run by separate AI-assisted (Claude) sessions. They
are not external peer review. The zero-power issue in the uniform-injection
arm was flagged by a co-author review and, separately, by the round-1 internal
AI-assisted review (§10.2 of its record).

**Per-seed records of the original experiments**

| Article item | Location |
|---|---|
| Table 2 (ablation) | [`results/phase1_baseline.json`](results/phase1_baseline.json) to [`results/phase6_final.json`](results/phase6_final.json); [`results/EXPERIMENT_REPORT.md`](results/EXPERIMENT_REPORT.md) |
| Table 3 (threshold sweep; mean and SD per threshold) | [`results/alert_threshold_sensitivity.json`](results/alert_threshold_sensitivity.json) |
| Table 4 (redesign arms 0–4) | [`results/live_inference_results.json`](results/live_inference_results.json) |
| Arm 5 and the Arm 4 rerun | [`results/arm5_h1_powered.json`](results/arm5_h1_powered.json) |
| Execution log containing the 28 July 2026 runs: lines 1–141 are the two executions of 28 July (redesign arms; Arm 5 with the Arm 4 rerun); later entries are runs from October 2026 | [`results/live_inference.log`](results/live_inference.log) |
| Table 5 (operating point) | [`results/fpr_diagnostic.json`](results/fpr_diagnostic.json) |
| Provenance isolation | [`results/provenance_isolation.json`](results/provenance_isolation.json); protocol [`preregistration/OPTION4_provenance_isolation_PREREGISTRATION.txt`](preregistration/OPTION4_provenance_isolation_PREREGISTRATION.txt) |
| Isolation Forest factorial | [`results/if_factorial_results.json`](results/if_factorial_results.json) |

**False-positive repairs (Table 6)**

| Repair | Pre-run document | Design seeds | Held-out seeds |
|---|---|---|---|
| Shrinkage | [`preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt`](preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt) | [`results/shrinkage_experiment.json`](results/shrinkage_experiment.json) | none |
| Geometric mean (exploratory) | none located | [`results/aggregation_dev.json`](results/aggregation_dev.json) | [`results/aggregation_fresh.json`](results/aggregation_fresh.json) (52–56) |
| Shortcut routing | [`preregistration/OPTION8_product_level_signal_PRECOMMITMENT.txt`](preregistration/OPTION8_product_level_signal_PRECOMMITMENT.txt) | [`results/option8_dev.json`](results/option8_dev.json) | [`results/option8_heldout.json`](results/option8_heldout.json) (57–61) |
| Arithmetic mean | [`preregistration/OPTION9_nonmultiplicative_aggregation_PRECOMMITMENT.txt`](preregistration/OPTION9_nonmultiplicative_aggregation_PRECOMMITMENT.txt) | [`results/option9_dev.json`](results/option9_dev.json) | [`results/option9_heldout.json`](results/option9_heldout.json) (62–66) |
| Unclipped trust | [`preregistration/OPTION10_continuous_trust_PRECOMMITMENT.txt`](preregistration/OPTION10_continuous_trust_PRECOMMITMENT.txt) | [`results/option10_dev.json`](results/option10_dev.json) | [`results/option10_heldout.json`](results/option10_heldout.json) (67–71) |

Scripts: [`run_shrinkage_experiment.py`](run_shrinkage_experiment.py),
[`run_aggregation_experiment.py`](run_aggregation_experiment.py),
[`run_product_signal_experiment.py`](run_product_signal_experiment.py),
[`analyze_option9_cached.py`](analyze_option9_cached.py),
[`analyze_option10_cached.py`](analyze_option10_cached.py).
The timing of each pre-run document is stated in
[`preregistration/README.md`](preregistration/README.md).

**Per-seed supplement:** [`supplement/per_seed/per_seed_supplement.md`](supplement/per_seed/per_seed_supplement.md),
with machine-readable values in [`supplement/per_seed/per_seed_supplement.csv`](supplement/per_seed/per_seed_supplement.csv)
and the build script. It reads the records above only.

**Code and configuration:** [`config/config.yaml`](config/config.yaml).
Detection is computed in [`evaluation/run_simulation.py`](evaluation/run_simulation.py)
(proxy design) and [`evaluation/live_inference.py`](evaluation/live_inference.py)
(redesign), with [`pts/product_trust_score.py`](pts/product_trust_score.py).
The models are in [`ai_modules/`](ai_modules/). Article Table 9 lists the parameters.

**Post-review analyses** (N0 reproduction of the saved models' outputs, N1
blend removal, N2 classifier comparison, model-free sweep and bounds, record
rechecks, numerical verification) are in the archived evidence package, not in
this repository. Start with its `START_HERE.md`. Its `REPRODUCE.md` separates
model-free steps, steps that run saved models, and steps that retrain.

**Archive:** version DOI [10.5281/zenodo.24097169](https://doi.org/10.5281/zenodo.24097169), concept DOI [10.5281/zenodo.21987357](https://doi.org/10.5281/zenodo.21987357). Version 1.0.0
(10.5281/zenodo.21987358) is the original-submission snapshot. It does not
contain the repair records, the factorial or the post-review analyses.

## Working notes kept in the repository root

`FIX_PROMPT_*.md` and `DEBUG_PROMPT_*.md` are the authors' working
specifications, written to plan the false-positive repairs and to diagnose
the shrinkage null result. They are kept unchanged as part of the record
(`preregistration/OPTION8_product_level_signal_PRECOMMITMENT.txt` cites one
of them). They are not protocols and carry no timing claim; the timing of
each repair is documented in `preregistration/README.md`.

## License

MIT — see `LICENSE`.
