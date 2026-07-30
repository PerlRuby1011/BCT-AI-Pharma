# Pre-registration and review record

These are the working documents behind the journal paper's pre-registered
experiments. They are published unedited, including the parts that record
mistakes, withdrawn conclusions, and disagreements between review rounds.
That is deliberate: a pre-registration is only evidence if the record of
what changed after results were seen is visible.

Read in this order.

| File | What it is |
|---|---|
| `PTS_CALIBRATION_findings_and_open_questions.txt` | First episode. Why PTS recalibration could not raise the counterfeit detection rate, and why the 78.2% target it was aimed at was not a measured quantity. |
| `PTS_CALIBRATION_response_and_resolution.txt` | Independent second session confirming the above, plus the alert-threshold sweep showing that the one setting which reaches the target is the setting at which the paper's comparative claim inverts. |
| `OPTION2_live_inference_PREREGISTRATION.txt` | The frozen protocol for the live-inference redesign: hypotheses, gates, decision rules, amendments, the pre-run predictions (§13), and the recorded results (§14). Read §13 before §14 — the predictions were written before execution. |
| `LIVE_INFERENCE_RESULTS_for_review.txt` | Results, round-1 independent review (§10), and the ARM 5 follow-up (§11). |
| `REVIEW_ROUND2_request.txt` | Round-2 review request with answers filled in, the R-C geometry finding (§9), and the action list (§10, §11). |
| `OPTION3_weight_sweep_PREREGISTRATION.txt` | Successor experiment. **Not approved, not run.** Its guards were substantially rewritten after review found the original selection criterion degenerate. |

## Things the record shows that a polished write-up would hide

- The conference version's headline numbers (96.8% detection, 2.8 min recall,
  t = 6.42) were **design targets, not measured results**. The journal paper
  reports only measured values, which are lower.
- A hypothesis was initially graded **FALSIFIED** when the test that produced
  it had **zero statistical power by construction**. An independent reviewer
  caught it; the verdict was withdrawn and restated as *not testable in that
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
