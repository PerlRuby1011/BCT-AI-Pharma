# Changes Log

## Phase 2 — realistic distributions (2026-07-27T22:02:02)
- Temperature reading std dev: 1.2C -> 0.8C (N(5.0, 0.8) cold-chain distribution)
- Counterfeit injection: uniform random -> clustered by manufacturer (30% of manufacturer nodes treated as a compromised cluster)
- Custody transfer timing: uniform arrival -> exponential(lambda=0.1) inter-arrival
Rationale: Realistic pharmaceutical distributions improve model generalization: uniform placement/timing is an unrealistic simplification of real supply-chain data.

## Phase 3 — IF contamination 0.10->0.05 (2026-07-27T22:02:13)
- isolation_forest.contamination: 0.10 -> 0.05
Rationale: Matches the true anomaly injection rate of 5% (simulation.anomaly_rate).

## Phase 4 — best hyperparams (grid search) (2026-07-27T22:03:56)
- LSTM hyperparameters: grid search over units_layer1, units_layer2, dropout, learning_rate (winning combination recorded below)
- Grid search winner: {'units_layer1': 96, 'units_layer2': 48, 'dropout': 0.2, 'learning_rate': 0.001}
Rationale: Grid search selects the architecture/optimizer configuration that minimizes validation MAPE on held-out synthetic sequences.

## Phase 5 — CNN overlap 0.18->0.10 (2026-07-27T22:14:48)
- cnn.overlap_factor: 0.18 -> 0.10
Rationale: Lower synthetic class-overlap reduces label noise in the packaging-verification training data, improving CNN discriminability.

## Phase 6 — all improvements combined (2026-07-27T22:24:51)
- All Phase 2-5 improvements applied simultaneously
Rationale: Combined ablation: measures the total effect and any interaction (redundancy or synergy) between the four individually-validated improvements.

## PTS Calibration Fixes A/B/C — implemented and diagnosed (2026-07-28)

Goal: raise measured BCT-AI counterfeit detection above the AI-Only Table VI
baseline of 78.2% (measured Phase 6 value: 71.4% ± 1.4%).

### Code changes (all default-OFF; baseline behaviour byte-identical)

- `pts/product_trust_score.py`
  - `validate_drug_class_weights()` — asserts each drug class's 8 expanded
    weights sum to 1.0. Enforces the sum-to-1.0 constraint mechanically
    rather than by inspection.
  - `quarantine_threshold_for_drug_class()` — resolves a per-class
    quarantine threshold (**Fix C**), falling back to the global
    `pts.quarantine_threshold`. Raises if the resolved threshold is not
    strictly below `pts.alert_threshold`, so the quarantine zone can never
    be made unreachable.
  - `quarantine_override_triggered()` — **Fix B**. Forces quarantine when
    the *raw* CNN authenticity score < `cnn_authenticity_max` AND the
    temperature-compliance score S2 < `temperature_compliance_max`, for the
    configured drug classes. Uses `ProductState.cnn_authenticity_score`,
    not `components["ai_confidence"]` (the latter is the average of the CNN
    score and the inverse Isolation Forest score, so it is not a substitute).
  - `compute_pts_for_drug_class()` — now consults the per-class threshold
    and the override, and reports `quarantine_threshold` /
    `quarantine_override` in its result for auditability.
- `config/config.yaml` — added `pts.quarantine_override` (`enabled: false`)
  and documented the optional per-class `quarantine_threshold` key. No
  baseline weight or threshold value was changed.
- `evaluation/run_simulation.py` — `_detection_and_recall()` now applies the
  per-class quarantine threshold and the Fix B override when computing
  `bct_ai_recall_efficiency`, and reports `bct_ai_override_rate`. The
  AI-Only comparison condition retains the single global threshold, since it
  has no drug-class-aware policy layer.
- `pts/calibration_diagnostic.py` — new. Runs baseline, A, B, C, A+B, A+C,
  B+C, A+B+C over 5 seeds x 50,000 transactions, plus two extra probes
  (see below). Writes `results/pts_calibration_diagnostic.json`.
- `tests/test_pts.py` — 8 new tests covering weight-sum validation, Fix A
  proportional rescaling, Fix B gate/class/enable semantics, the Fix B
  no-readings case, per-class thresholds, and the threshold-ordering guard.
  Full suite: 61 passed.

### Fix definitions as implemented

- **Fix A** — Class A `w8_ai_confidence` 0.10 → 0.20. The residual 0.80 is
  distributed over w1/w2/`remaining_total` by the scale factor
  `(1 - 0.20) / (1 - 0.10) = 0.8889`, preserving their relative proportions:
  w1 0.25 → 0.2222, w2 0.35 → 0.3111, remaining_total 0.30 → 0.2667.
  Sum = 1.0 exactly (asserted).
- **Fix B** — Class A override at CNN < 0.30 AND S2 < 0.20.
- **Fix C** — Class A quarantine threshold 0.50 → 0.45.

### Two added probes (not in the original three)

- **Fix C-up** — Class A quarantine threshold 0.50 → **0.55**. The 0.518
  excursion that motivated Fix C is a case that *failed* to quarantine at
  0.50; catching it requires a threshold **above** 0.518. Lowering to 0.45
  moves away from the motivating finding. C-up is measured alongside C so
  the direction of the effect is on the record.
- **Fix B-obs** — Fix B with gates widened to values the
  counterfeit-detection product states can actually reach
  (`cnn_authenticity_max: 0.90`, `temperature_compliance_max: 1.01`). Used
  only to distinguish "override never fires" (coverage) from "override is
  not wired in" (bug).

### Diagnostic result — 5 seeds (42-46) x 50,000 transactions

Detection = fraction of counterfeit-labeled products with PTS below the
**alert** threshold (0.75). Quarantine = fraction below the class quarantine
threshold or forced by the Fix B override.

| Variant | Detection | Δ vs base | Quarantine | Δ vs base | >78.2%? |
|---|---|---|---|---|---|
| baseline | 70.00% | +0.00 | 36.65% | +0.00 | no |
| A (w8 0.10→0.20) | 70.00% | +0.00 | 34.99% | −1.65 | no |
| B (quarantine override) | 70.00% | +0.00 | 36.65% | +0.00 | no |
| C (quar. thr 0.50→0.45) | 70.00% | +0.00 | 35.47% | −1.18 | no |
| C-up (quar. thr 0.50→0.55) | 70.00% | +0.00 | 37.66% | +1.01 | no |
| A+B | 70.00% | +0.00 | 34.99% | −1.65 | no |
| A+C | 70.00% | +0.00 | 33.60% | −3.05 | no |
| B+C | 70.00% | +0.00 | 35.47% | −1.18 | no |
| **A+B+C (combined)** | **70.00%** | **+0.00** | **33.60%** | **−3.05** | **no** |
| A+B+C-up | 70.00% | +0.00 | 36.42% | −0.23 | no |
| B-obs (reachable gates) | 70.00% | +0.00 | 41.30% | +4.66 | no |

Per-seed detection, baseline: [0.714, 0.718, 0.678, 0.680, 0.710]
Per-seed detection, Fix A:   [0.714, 0.718, 0.672, 0.686, 0.710]
(The identical 70.00% means are a coincidence of these 5 seeds summing to
3.500 in both cases; Fix A does move individual seeds by ±0.6 pp, within
per-seed noise of ~±2 pp at n=500 counterfeits/seed.)

### Conclusion: none of the three fixes can raise the detection rate

1. **Fixes B and C cannot affect detection by construction.** Detection is
   thresholded at the **alert** boundary (`run_simulation.py`,
   `bct_ai_detection = mean(pts < alert_threshold)`); B and C both act on
   the **quarantine** decision, which drives `recall_efficiency`. Their
   detection delta is structurally zero, not empirically zero.
2. **Fix B never fires in the counterfeit path.** That path constructs
   `ProductState(temperature_readings_c=[])`, and
   `score_temperature_compliance()` returns exactly 1.0 with no readings, so
   the `S2 < 0.20` gate is unreachable. The B-obs probe (gates widened to
   reachable values) moved quarantine by +4.66 pp, confirming the override
   is correctly wired — the null result is a coverage artifact of the
   evaluation harness, not a bug. Fix B remains meaningful in
   `run_pts_pipeline()`, where real temperature readings exist.
3. **Fix A shifts weight from the strong signal to the weak one.** In the
   counterfeit path, provenance responds to severity with slope 0.90 while
   AI confidence responds with slope ≈0.15 (it is diluted by an
   Isolation Forest score irrelevant to counterfeiting). Raising w8
   therefore cannot help detection; it measurably *reduces* quarantine
   (−1.65 pp).
4. **Fix C points the wrong way.** −1.18 pp quarantine. C-up (0.55), the
   direction the 0.518 finding actually motivates, gives +1.01 pp.
5. **Combined A+B+C is strictly worse than baseline**: detection unchanged
   at 70.00%, quarantine −3.05 pp.

Because the combined fix does not exceed 78.2%, the full 10-seed
orchestrator was **not** run and no journal-paper numbers were revised —
per the pre-registered decision rule.

### Separate finding: the 78.2% target is not a measured quantity

The AI-Only 78.2% figure is a Table VI constant in
`config.yaml:baseline_comparison`, carried over from the conference paper's
design targets. This harness's own AI-Only condition — PTS with all weight
on `ai_confidence`, computed from the same draws as BCT-AI — measures
**55.92%** on these seeds (57.59% at full scale).

**CORRECTION (added after reviewing the manuscript directly).** An earlier
version of this section stated that "the current RQ1 comparison places a
measured number (71.4%) against a non-measured one (78.2%)." That is
**wrong**, and the error was mine. The manuscript already does the right
thing on every count:

- Table~\ref{tab:detection} lists the row as "AI-Only (measured) 57.6%",
  with a footnote stating that the conference version's 78.2% "is a design
  target ... not reproduced here."
- **RQ1's threshold is 90%, not 78.2%** ("counterfeit detection above 90\%
  at national supply chain scale"). Exceeding 78.2% would therefore not
  have flipped RQ1 to "Yes" even if a fix had worked.
- RQ2 — the question that actually asks about beating AI-only — is already
  answered **"Yes"**: 71.4% vs 57.6%, t(9)=52.20, p<0.0001, a 13.8 pp gap
  stable across all ten seeds.
- The manuscript's RQ1 answer already carries the correct qualification,
  that 71.4% "measures the sensitivity of the PTS aggregation ... not the
  accuracy of live model inference," and already names the per-transaction
  inference redesign as Future Work item 1.

So the premise of this whole calibration exercise — "the AI-only baseline
is 78.2% and we must exceed it to flip RQ1" — was mistaken on both halves.
No manuscript change is warranted, and none was made.

### Full-scale confirmation — 10 seeds (42-51) x 230,000 transactions

The 5-seed/50K diagnostic above was re-run at the *exact* scale and
data-quality settings of the published Phase 6 result, since the detection
metric costs only ~12s per 10-seed sweep (the 28-minute Phase 6 runtime was
AI-module training, which none of these fixes touch). Baseline reproduces
the published **71.40% ± 1.35%** exactly, confirming the
`run_simulation.py` wiring change is metric-neutral.

| Variant | Detection | Δ | Quarantine | Δ | >78.2%? |
|---|---|---|---|---|---|
| baseline | 71.40% ± 1.35% | +0.00 | 36.59% ± 0.26% | +0.00 | no |
| A | 71.24% ± 1.32% | −0.16 | 35.04% ± 0.26% | −1.55 | no |
| B | 71.40% ± 1.35% | +0.00 | 36.59% ± 0.26% | +0.00 | no |
| C | 71.40% ± 1.35% | +0.00 | 35.46% ± 0.25% | −1.13 | no |
| C-up (0.55) | 71.40% ± 1.35% | +0.00 | 37.72% ± 0.24% | **+1.13** | no |
| A+B | 71.24% ± 1.32% | −0.16 | 35.04% ± 0.26% | −1.55 | no |
| A+C | 71.24% ± 1.32% | −0.16 | 33.67% ± 0.27% | −2.93 | no |
| B+C | 71.40% ± 1.35% | +0.00 | 35.46% ± 0.25% | −1.13 | no |
| **A+B+C (combined)** | **71.24% ± 1.32%** | **−0.16** | **33.67% ± 0.27%** | **−2.93** | **no** |
| A+B+C-up | 71.24% ± 1.32% | −0.16 | 36.41% ± 0.26% | −0.18 | no |

Measured AI-Only condition on the same paired draws: **57.59%**.

At full scale Fix A resolves to a small but consistent **decrease**
(−0.16 pp), matching the analytic prediction in point 3 above; the 5-seed
diagnostic's exact +0.00 was a sampling coincidence. The conclusions are
otherwise unchanged: **no fix, alone or combined, raises counterfeit
detection, and the combined fix is strictly worse than baseline on both
metrics.** The only change that improves anything is C-up (+1.13 pp
quarantine) — the *opposite* of Fix C as specified.

Written to `results/pts_calibration_diagnostic_fullscale.json`.

## Alert-threshold sensitivity sweep (2026-07-28)

Run because two independent sessions identified the alert threshold as the
*only* parameter capable of moving the detection number, making its
sensitivity the thing most worth documenting. **This is a sensitivity
analysis. The operating point remains 0.75. No threshold was selected on
the basis of the detection value it produces.**

10 seeds x 230,000 transactions, Phase 6 data-quality settings.

| Alert threshold | BCT-AI detection | AI-Only detection | Gap (pp) |
|---|---|---|---|
| 0.60 | 50.91% | 5.57% | **+45.34** |
| 0.65 | 57.85% | 16.21% | **+41.64** |
| 0.70 | 64.49% | 34.16% | **+30.33** |
| **0.75 (operating point)** | **71.40%** | **57.59%** | **+13.82** |
| 0.80 | 78.24% | 78.53% | **−0.29** |
| 0.85 | 85.31% | 92.39% | −7.08 |
| 0.90 | 92.47% | 98.23% | −5.77 |

### The decisive result

**At alert = 0.80, BCT-AI detection is 78.24% — it clears the 78.2% target.
And at that same threshold the AI-Only condition reaches 78.53%, overtaking
it.** The single setting that would have delivered the requested number is
precisely the setting at which the paper's central comparative claim (RQ2)
inverts and dies.

This is the strongest available evidence that threshold-tuning toward a
target is not a weaker version of the right method — it is a different
operation that destroys the result it is aiming at. Anyone who had searched
parameters with "exceeds 78.2%" as the stopping rule would have stopped
here, reported success, and silently inverted RQ2.

The mechanism: the AI-Only score is tightly concentrated
(S8 ~ 0.81 − 0.15·s, range [0.456, 0.973]), so it crosses a high threshold
almost all at once, whereas BCT-AI's provenance term spreads its
distribution (S1 range [0.000, 1.000]). Above ~0.80 the whole AI-Only mass
falls below the line. **The 13.8 pp BCT-AI advantage is therefore specific
to the 0.75 operating point and reverses above ~0.80** — a genuine
qualification of RQ2 that the manuscript does not currently state, and the
one finding from this exercise that may be worth adding to it.

Written to `results/alert_threshold_sensitivity.json`.

## Manuscript: RQ2 operating-point qualification added (2026-07-28)

Following the alert-threshold sweep, two additions to
`Journal Journey/BCT_AI_Pharma_CompInd.tex` (the only manuscript edits
made in this exercise):

1. **Section 5.8** — new subsubsection "Operating-point dependence of the
   comparative advantage", with `Table~\ref{tab:threshold}` reporting the
   full sweep, the sign change between 0.75 and 0.80, the distributional
   mechanism, and an explicit statement that the 0.75 operating point was
   fixed from the conference specification before any value in the table
   was computed.
2. **Section 7, RQ2 answer** — a qualifying paragraph recording that the
   13.8 pp advantage holds at every threshold at or below 0.75, narrows to
   zero at 0.80, and reverses above it, with the recommendation that the
   threshold be cited alongside the comparison.

No other manuscript text was changed. No published number was revised.
Compiles clean: pdflatex x3, exit 0, 0 undefined refs/citations, 83 pages
(was 81), 11 tables (was 10). The one float-size warning is pre-existing
(line 789, smart-contract table, unrelated to these edits).

## Option 2 pre-registration drafted, NOT run (2026-07-28)

`Journal Journey/OPTION2_live_inference_PREREGISTRATION.txt` — protocol
for the per-transaction live-inference redesign (Future Work item 1),
written for approval before execution. Nothing was run.

Contains: the causal-coupling hypothesis H1 and its falsification
criterion; signal-construction spec for all three PTS inputs; four arms
including two deliberate model-degradation arms that test coupling; six
pre-registered decision rules; reporting obligations covering every
outcome; effort/compute estimate; and a risk register.

Two findings from scoping worth recording independently of whether the
experiment is approved:

- **The manuscript overstates the cost of its own top-priority follow-up.**
  Future Work item 1 claims "an order-of-magnitude increase in
  computational budget". The redesign requires no per-seed retraining;
  the detection metric re-runs all ten seeds at full scale in ~12s. The
  real cost is 4-8 hours of implementation, dominated by custody-chain
  reconstruction and streaming, not by compute. If the redesign is not
  undertaken, this sentence should still be corrected.
- **The most likely failure mode is asymmetric implementation.** Making
  the CNN and Isolation Forest live while leaving provenance as the
  severity-linked proxy (0.90 slope) would rig the comparison in favour
  of the paper's own thesis and be *worse* than the current symmetric
  proxy design. Deriving provenance from realized custody chains is
  therefore mandatory, not optional; the pre-registration requires
  abandoning the redesign rather than shipping the partial version.
