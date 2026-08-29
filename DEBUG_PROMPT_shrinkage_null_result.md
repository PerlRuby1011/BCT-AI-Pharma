# Debug prompt — investigate the OPTION5 shrinkage null result

**Read-only investigation. Do NOT modify any code yet.** Analyze, instrument
with print/debug statements if needed (temporary, don't commit), and report
findings back. We'll decide on a fix together after seeing your diagnosis.

## Context

This repo implements a Blockchain+AI pharmaceutical supply-chain simulation.
`evaluation/live_inference.py` computes a provenance trust score (S1) per
product from its custody chain, used alongside AI-classifier signals in a
composite Product Trust Score (PTS) that triggers quarantine below a
threshold.

Relevant functions:
- `compute_org_reputation()` (line ~97) — derives per-organization trust
  from burn-in-period counterfeit rates. Has an optional `shrinkage: bool`
  parameter (added recently) that's supposed to apply empirical-Bayes
  shrinkage to the per-org rate estimate before computing trust, pulling
  noisy/low-volume organizations' rates toward the network base rate.
- `compute_provenance_scores()` (line ~140) — takes the trust dict and
  computes S1 per product as the PRODUCT of trust scores along that
  product's custody chain (computed in log space).

## What happened

`run_fpr_diagnostic.py` established that the published ARM4 configuration
(clustered/source-concentrated counterfeit injection) has a real, stable
~12.34% false-positive rate on clean transactions, averaged over 10 seeds.

We hypothesized the FPR is driven by noisy per-organization rate estimates
under `compute_org_reputation()`, and implemented `shrinkage=True` as a fix:
empirical-Bayes shrinkage of `org_rate` toward `base_rate`, weighted by each
organization's burn-in transaction count, with the shrinkage strength `k`
set automatically to the median per-org transaction count (no manual
tuning).

**Result (`run_shrinkage_experiment.py`, full 10-seed run):** shrinkage had
**zero measurable effect** — baseline and shrunk TPR/FPR were byte-identical
to 4 decimal places on every single seed, for both ARM1 (uniform injection)
and ARM4 (clustered injection). Not "small effect" — literally no change at
all.

## Leading hypothesis (unconfirmed — needs your investigation)

`compute_org_reputation()` clips the excess-over-base-rate term to
`PROVENANCE_EXCESS_CAP = 3.0` BEFORE the trust penalty is applied:

```python
excess = ((org_rate - base_rate) / base_rate).clip(lower=0.0, upper=cap)
trust = (1.0 - penalty * excess).to_dict()
```

Hypothesis: under clustered injection, the "hot" organizations' raw
`org_rate` is so far above `base_rate` that BOTH the unshrunk excess AND the
shrunk excess already exceed `cap=3.0` and get clipped to the same value —
so shrinkage changes the pre-clip number but never changes the post-clip
number that actually matters.

## What we need from you

1. **Confirm or refute the cap-saturation hypothesis with actual numbers.**
   Instrument `compute_org_reputation()` (temporarily, don't commit) to log,
   per organization, per a representative seed/arm (e.g. ARM4, seed 42):
   `n_org`, raw `org_rate`, `base_rate`, raw `excess` (pre-clip), shrunk
   `org_rate`, shrunk `excess` (pre-clip), and whether each was clipped.
   We want to see: for organizations whose clipped trust ended up identical
   before/after shrinkage, was it because both excess values exceeded the
   cap, or is there a different explanation (e.g. a bug in the shrinkage
   formula itself, an indexing/alignment issue, `k` being computed as ~0 or
   as something degenerate)?

2. **Check the shrinkage implementation for bugs independent of the cap
   hypothesis.** Specifically look at:
   - `n_org, org_rate = n_org.align(org_rate, join="right")` — could this
     silently produce NaN or misaligned values for some organizations, and
     could that fail silently in a way that reproduces the raw rate?
   - Whether `k = float(n_org.median())` could come out as a degenerate
     value (e.g., equal to `n_org` for most/all orgs, making the shrinkage
     formula collapse toward a no-op regardless of the cap).
   - Any other reason the shrunk and unshrunk `org_rate` Series could end up
     numerically identical before the clip is even applied.

3. **If the cap-saturation hypothesis is confirmed**, report:
   - Roughly what fraction of "hot" organizations have excess exceeding the
     cap, both before and after shrinkage, under ARM4/clustered injection.
   - How far past the cap the raw excess typically is (e.g., is it just
     barely over 3.0, or wildly over it — this matters for whether a larger
     cap value would plausibly help).
   - Whether `compute_provenance_scores()`'s multiplicative (product-based)
     chain aggregation means a single low-trust hop could dominate a
     product's S1 regardless of cap value — i.e., is the cap even the right
     lever, or does the multiplicative structure itself make any per-org
     trust design vulnerable to this same failure mode?

4. **Do not change any code.** Report findings as a structured summary:
   what you found, with the actual numbers/evidence, and your own read on
   whether the leading hypothesis holds up. If you think there's a better
   explanation than cap saturation, say so and back it with the same kind
   of evidence.

## Constraints

- This is for a research paper going through a pre-registration discipline
  (see `preregistration/OPTION5_provenance_shrinkage_PREREGISTRATION.txt`
  for the full context of what's already been tried and why). Whatever you
  find will inform a NEW pre-registered fix, written and approved before any
  code change — so your job right now is diagnosis, not a patch.
- Needs `torch`/`tensorflow` to actually run (`pip install -r
  requirements.txt`); if you can reason about the pandas/numpy logic
  statically without running it, that's useful too, but actual printed
  numbers from a real run are much more convincing than a static read.
