# Protocols and review record

These are the working documents behind the journal paper's protocol-based
experiments. They are posted alongside the results, including the parts that record
mistakes, withdrawn conclusions, and disagreements between review rounds.
That is deliberate: a written protocol is only evidence if the record of
what changed after results were seen is visible.

Read in this order.

| File | What it is |
|---|---|
| `PTS_CALIBRATION_findings_and_open_questions.txt` | First episode. Why PTS recalibration could not raise the counterfeit detection rate, and why the 78.2% target it was aimed at was not a measured quantity. |
| `PTS_CALIBRATION_response_and_resolution.txt` | Independent second session confirming the above, plus the alert-threshold sweep showing that the one setting which reaches the target is the setting at which the paper's comparative claim inverts. |
| `OPTION2_live_inference_PREREGISTRATION.txt` | The written protocol for the live-inference redesign: hypotheses, gates, decision rules, amendments, the predictions (§13), and the recorded results (§14). Read §13 before §14. The protocol is dated 28 July 2026, the day of the run, and was first committed together with its results on 29 July 2026 (efe5c99); that the predictions preceded execution therefore rests on the document's own dating, not on a repository timestamp. |
| `LIVE_INFERENCE_RESULTS_for_review.txt` | Results, round-1 internal AI-assisted review (§10), and the ARM 5 follow-up (§11). |
| `REVIEW_ROUND2_request.txt` | Round-2 review request with answers filled in, the R-C geometry finding (§9), and the action list (§10, §11). |
| `OPTION3_weight_sweep_PREREGISTRATION.txt` | Successor experiment. **Not approved, not run.** Its guards were substantially rewritten after review found the original selection criterion degenerate. |

## False-positive repair protocols, and what establishes their timing

| Repair | Document | Timing |
|---|---|---|
| Provenance isolation (diagnostic) | `OPTION4_provenance_isolation_PREREGISTRATION.txt` | Committed 2026-08-23 17:37, before its run (record dated 17:39 the same day) |
| Shrinkage | `OPTION5_provenance_shrinkage_PREREGISTRATION.txt` | Committed 2026-08-24 13:47, before its run (record dated 2026-08-25 02:08) |
| Geometric-mean aggregation ("OPTION7") | none located | Exploratory: no pre-run plan located; code and results committed together on 2026-08-26 |
| Shortcut routing | `OPTION8_product_level_signal_PRECOMMITMENT.txt` | Dated 2026-08-25 in the document; first committed with its results on 2026-08-29 |
| Arithmetic-mean aggregation | `OPTION9_nonmultiplicative_aggregation_PRECOMMITMENT.txt` | Dated 2026-08-26 in the document; first committed with its results on 2026-08-30 |
| Unclipped trust | `OPTION10_continuous_trust_PRECOMMITMENT.txt` | Dated 2026-08-27 in the document; first committed with its results on 2026-08-31 |

The written protocol for the redesign, `OPTION2_live_inference_PREREGISTRATION.txt`,
is dated 2026-07-28 and was first committed alongside the results on
2026-07-29. The execution log `results/live_inference.log` contains the
28 July 2026 runs in its first 141 lines (later entries are runs from October
2026). At the start of each 28 July execution it names the protocol file and
records it as approved that day; it does not record the file's contents, so it
does not establish which version of the protocol existed before execution. Where a document was first committed together with its results, its
pre-run status rests on the document's own dates and content, not on a
repository timestamp.

## Things the record shows that a polished write-up would hide

- The conference version's headline numbers (96.8% detection, 2.8 min recall,
  t = 6.42) are **not computed by this repository**. In the audited
  implementation they are supplied through configuration (from release 1.1.0,
  as labelled conference-reported reference values), and the audited records
  do not establish the run that generated them. The journal paper does not use
  them as evidence; it reports values computed from transaction-level records,
  which are lower.
- A hypothesis was initially graded **FALSIFIED** when the test that produced
  it had **zero statistical power by construction**. A co-author review and
  a separate internal AI-assisted review both flagged it; the verdict was withdrawn and restated as *not testable in that
  arm*, and the missing arm was then run.
- The weight-sweep selection criterion (`R-C`) would, unconstrained, have
  selected the allocation that removes the blockchain entirely. Its objective
  saturates across ~59.7% of the feasible simplex, so a tie-break would have
  performed the entire selection while the stated objective performed none.
  The criterion was withdrawn before the experiment was approved.
- The same class of error — a correction pass missing an instance in a
  different *modality* from the one that triggered it — recurred four times,
  and was caught by an independent reader every time, never by self-check.

## Reproduction

```bash
python verify_power.py          # reachable-range / zero-power check
python verify_rc_geometry.py    # R-C plateau geometry
python run_live_inference_experiment.py    # full arms, ~50 min
python analyze_live_inference.py           # gates and hypotheses
```
