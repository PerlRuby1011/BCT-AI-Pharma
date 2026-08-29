# Prompt — add a real per-product signal to close the provenance FPR gap

This is a bigger, different kind of change than the last two fixes: it
modifies the **data generator**, not just the detector. Read Section 4
(integrity constraints) before writing any code — it's the part that
matters most here, more than in the previous two attempts.

## Context (don't re-derive)

Two prior fix attempts on this same problem, both in the codebase already:
1. Empirical-Bayes shrinkage of per-org rate estimates (`compute_org_reputation(..., shrinkage=True)`)
   — clean null, explained by clip saturation.
2. Chain-length-normalized geometric mean aggregation
   (`compute_provenance_scores(..., aggregation="geometric_mean")`) —
   genuinely improves AUC (0.915→0.971 dev, 0.913→0.970 held-out) but does
   NOT improve FPR at the paper's actual operating point (worse by
   1.53–1.97pp, confirmed on 5 held-out seeds). Both are additive, default
   behavior unchanged, already committed.

Root cause, now fully understood: trust is a property of an
**organization**; the label is a property of a **product**. One
organization ("hot org") has a genuinely elevated ~12% counterfeit rate;
88% of its traffic is clean and gets the same penalty. No rescaling or
reaggregation of per-organization trust can fix this, because the
information needed to tell two products apart — something specific to
*that product's* journey, not its handler's history — doesn't exist
anywhere in the current transaction data for counterfeit items.
`simulation/anomaly_injector.py`: a `counterfeit_product` row currently
only gets `anomaly_severity` set (0.4–1.0), which feeds the CNN's synthetic
image and nothing else. No route, timing, quantity, or hop-count
perturbation happens.

## The task

Give counterfeit products a genuine, individually-varying, causally
plausible signature in their custody chain — something a real counterfeit
product plausibly WOULD exhibit — then let the provenance scorer read that
signature as an observable, never as the label itself.

Plausible real-world-motivated candidates (pick one, or propose your own
with the same justification standard):
- **Shortcut/irregular routing**: counterfeit goods entering the legitimate
  supply chain via fewer or unusual intermediate handoffs (a known
  real-world pattern — counterfeiters often skip legitimate distribution
  tiers).
- **Atypical timing**: unusually fast or unusually delayed handoff gaps
  relative to the typical timing for that route/org-type pair.
- **Quantity irregularities**: split-shipment or batch-size patterns that
  deviate from the typical pattern for that product/org.

Whichever you pick, implement it as a new, clearly-named perturbation in
`inject_anomalies()` (or a new function it calls) for `counterfeit_product`
rows only, alongside the existing `anomaly_severity` assignment. Then add a
new, genuinely per-product feature computation (in `live_inference.py` or a
new module) that reads this observable and folds it into S1 or a new
signal — again, reading the observable column, never `is_anomaly` /
`anomaly_type` directly.

## Practical harness

Same scaffolding as the last two experiments
(`run_fpr_diagnostic.py`, `run_aggregation_experiment.py`) —
`train_models`, `build_transactions`, `pts_for`, `split_timeline`, etc.
Fast iteration on a couple of seeds first, full comparison later.

## Section 4 — integrity constraints (read this part twice)

This is a bigger deal than the last two fixes because you're changing what
"counterfeit" looks like in the simulation, not just how it's read. Two
rules, both non-negotiable:

1. **Choose the signal's existence and its magnitude for real-world
   plausibility, BEFORE checking what it does to FPR.** State the
   justification in a comment or a short note before running anything
   (e.g. "counterfeit chains skip on average 1 legitimate handoff, based on
   X real-world pattern" or "timing gap distribution shifted by Y, chosen
   because Z"). If you find yourself adjusting the magnitude because a
   different value gives a better FPR, STOP — that recreates exactly the
   "detection rate as a function of an injection constant" trap already
   flagged and rejected for this project (see
   `preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt`'s
   framing of Option B in the earlier discussion). Pick the value for a
   stated real-world reason, run it once, report what happens — don't tune
   it toward an outcome.
2. **Same held-out discipline as before.** Iterate/debug on seeds 42–51.
   Freeze the final design (the perturbation AND the detector feature that
   reads it) before looking at seeds 52–56 (or a fresh block if those are
   now "used"; ask the author which seeds are still clean if unsure).
   Report both.

## What to hand back

The perturbation added (exact mechanism + real-world justification stated
in advance, not after), the detector feature added, TPR/FPR (and AUC) at
the paper's registered operating point on both the design seeds and the
held-out confirmation seeds, and an explicit note that this is a
data-generation change requiring disclosure — one sentence describing
exactly what changed in the generator, suitable for dropping into a
manuscript's methods/limitations section. If it doesn't work, report that
as plainly as the last two attempts were reported — a third honest null,
on top of two others, would still be a genuinely strong result for this
paper, not a failure.
