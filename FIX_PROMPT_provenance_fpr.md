# Prompt — iterate on a real fix for the provenance false-positive rate

You may modify code freely in this session. This is exploration, not a
frozen pre-registered protocol — move fast, try things, no ceremony needed
for the iteration itself. There is exactly one hard rule at the end before
anything goes in a paper (Section 5). Read that before you report a number
to the author.

## Context (what's already known, don't re-derive)

Files: `evaluation/live_inference.py` (`compute_org_reputation`,
`compute_provenance_scores`), `run_fpr_diagnostic.py`,
`run_shrinkage_experiment.py`. Full history in
`preregistration/OPTION4_provenance_isolation_PREREGISTRATION.txt` and
`OPTION5_provenance_shrinkage_PREREGISTRATION.txt` if you want the complete
trail — you don't need to read those to start, this section has the gist.

- Published ARM4 config (live inference, published CNN, clustered/
  source-concentrated counterfeit injection): **92.29% TPR, 12.34% FPR**,
  stable across 10 seeds (42–51).
- Diagnosed root cause: trust is computed **per organization**
  (`compute_org_reputation`) from that org's burn-in counterfeit rate, but
  the label is **per product**. One organization ("hot org") has a
  genuinely elevated ~12% counterfeit rate; the other 88% of its traffic is
  clean and inherits the same trust penalty. Mean S1 for clean-but-touches-
  hot-org (0.6384) is barely distinguishable from mean S1 for counterfeit
  (0.6037).
- S1 is computed as a **product** of per-org trust scores along a custody
  chain (~12 hops, `compute_provenance_scores`, log-sum internally). One
  low-trust hop can dominate a chain's S1 regardless of the other ~11 hops.
- Already tried and failed cleanly: empirical-Bayes shrinkage of the
  per-org rate estimate (`compute_org_reputation(..., shrinkage=True)`,
  already in the codebase). Null result — the excess-over-base-rate term
  is clipped (`PROVENANCE_EXCESS_CAP=3.0`, and a lower clip at 0.0), and
  every organization sits at one clip boundary or the other, so shrinking
  an already-clipped value doesn't change the output. This rules out
  "noisy rate estimate" as the cause. The rate is measured correctly; the
  scoring design just can't use per-product information because it doesn't
  have any per-product information to use.
- **The transaction generator currently gives counterfeit products no
  distinguishing per-transaction signal at all**, other than which
  organization handled them. `simulation/anomaly_injector.py`: when a row
  is marked `counterfeit_product`, only `anomaly_severity` (0.4–1.0) is
  set; nothing about route, timing, quantity, or hop count changes. (Note:
  `anomaly_severity` DOES feed the CNN's synthetic packaging image — that's
  a legitimate observable proxy, not label leakage, because the CNN only
  ever sees the rendered image, not the severity number. Keep that same
  discipline for anything you add: the detector must read an *observable*
  the generator produces, never the ground-truth label or a value derived
  directly from it.)

## What's genuinely open — pick a direction and try it

You have two different levers available, and they're not mutually
exclusive:

**A. Fix the aggregation, not just the estimate.** S1 is currently a raw
product across the whole chain, so one bad hop is fatal regardless of
chain length or the quality of the other hops. Consider alternatives:
a bounded/damped aggregation (e.g. geometric mean instead of raw product,
or a a capped per-chain penalty so one hop can't fully zero out a long
clean chain), or weighting hops by recency/position, or a hop-count-aware
normalization. This doesn't require touching the data generator — it's a
scoring-function change, evaluate it the same way ARM4 already is.

**B. Give the detector a real per-product signal.** This means modifying
`simulation/anomaly_injector.py` so that a `counterfeit_product` row gets
some additional, causally-motivated, per-transaction perturbation beyond
severity — e.g. an atypical handoff timing gap, a shortcut/irregular route
(real counterfeit goods often skip legitimate distribution steps), or a
quantity/documentation irregularity. Then give `compute_provenance_scores`
or a new scoring function access to that observable (never to
`is_anomaly`/`anomaly_type` directly). This is a **data-generation
change** — flag it clearly as such in your own notes, because it changes
what "counterfeit" looks like in the simulation, not just how cleverly it's
detected. That's fine to do, but it must be disclosed as exactly that in
the manuscript later, the same way the existing severity/CNN mechanism
already is.

Try either or both. Fast iteration is fine — print TPR/FPR on a couple of
seeds, adjust, repeat, however you'd normally debug this.

## Practical harness

`run_fpr_diagnostic.py` and `run_shrinkage_experiment.py` already have the
train/build/measure scaffolding (`train_models`, `build_transactions`,
`pts_for`, `split_timeline`, `compute_cnn_authenticity`,
`compute_isolation_scores`). Fastest path is probably a new small script
that reuses those imports, computes your candidate S1 (via a new function
or a new `compute_org_reputation`/`compute_provenance_scores` variant), and
prints TPR/FPR the same way. Use `--seeds 42,43,44` (or edit directly) for
fast iteration — don't run all 10 seeds every loop.

## Section 5 — the one hard rule before anything goes in the paper

Iterate and tune on seeds 42–51 all you want — that's the data the FPR
problem was originally found on, and using it to explore is fine. But
**do not report a final TPR/FPR number to the author, or put one in a
paper, that was measured only on seeds you also used to pick the design.**
That's tuning-on-the-test-set, and the improvement could be real or could
just be the variant that happened to fit those 10 runs.

Before reporting a final number: pick ONE design (stop iterating), then run
it fresh on seeds it has never seen — e.g. `52,53,54,55,56` — using the
exact same harness. Report both: the seeds-42–51 number (which is
comparable to everything already measured) and the fresh-seed confirmation
number. If they're close, you have a real result. If the fresh-seed number
is much worse, that's the honest finding — report it as such, don't keep
iterating against the fresh seeds either (that just moves the same problem
one level up).

## What to hand back

For whatever you land on: the final design (code diff or description), the
seeds-42–51 TPR/FPR, the fresh-seed TPR/FPR, and — if you went with option
B — an explicit note that it's a data-generation change and a one-line
description of exactly what changed in the generator. No need to write
this up formally; a plain summary is enough for the author to bring back
here.
