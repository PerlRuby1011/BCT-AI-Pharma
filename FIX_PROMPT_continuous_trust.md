# Prompt — fifth attempt: make organization trust continuous, not binary

Same discipline as the last four: freeze the design and its justification
in writing before measuring anything (`OPTION10_continuous_trust_PRECOMMITMENT.txt`,
same style as OPTION8/9), hold out fresh seeds never used in this line
(**42–66 are all consumed — use 67–71 as the next clean held-out block**),
report honestly regardless of outcome.

## Why "just remove the lower clip" is not enough — check this before designing

The four prior nulls converged on: trust only takes two values (1.0 for 11
organizations, 0.7 for one), because `compute_org_reputation`'s
`excess = clip((org_rate - base_rate)/base_rate, lower=0.0, upper=cap)`
floors every organization at or below the base rate to `excess=0`, hence
`trust=1.0` for all of them identically.

But look at WHY those 11 organizations all clip to zero: they don't have
slightly-below-base rates that get rounded up — **their raw burn-in
counterfeit rate is exactly `0.000000` for all 11, identically**, because
none of them had a single counterfeit transaction in burn-in. Removing the
lower clip on the RAW (unshrunk) rate does nothing, because
`(0.0 - base_rate)/base_rate` is the same negative number for all 11 —
they'd still be perfectly tied, just at a different (negative, unclipped)
value instead of zero. Un-clipping alone doesn't manufacture information
that was never there.

**What actually contains differentiating information is burn-in
transaction *volume* per organization**, combined with OPTION5's
shrinkage: `shrunk_rate = (n_org*0 + k*base_rate)/(n_org+k)`, which is
DIFFERENT for each organization because their `n_org` (burn-in transaction
counts) differ, even though their raw rate is identically zero. OPTION5's
own diagnosis already showed this shrinkage computes real, substantial,
differentiated pre-clip values (mean |Δrate|=0.0092) — the reason it had
zero effect wasn't that shrinkage failed, it's that the SAME lower clip
this prompt is about erased the differentiation immediately afterward
(shrunk_rate is still below base_rate for all 11, so excess is still
negative, still clipped to 0).

**Therefore: this experiment must combine un-clipping with shrinkage, not
test un-clipping alone on raw rates.** Testing un-clipping alone would be a
foregone, uninformative null — don't spend seed budget confirming it;
state this reasoning in the pre-commitment instead of running it.

## What to freeze

1. **Formula:** `excess = (shrunk_rate - base_rate) / base_rate`, clipped
   ONLY on the upper end (`cap=3.0`, unchanged) — no lower clip, or a
   symmetric lower bound at `-cap` if you want organizations far below
   base rate to get a bounded trust bonus rather than an unbounded one.
   State which you're using and why before testing.
2. **Trust can now exceed 1.0** for organizations shrunk-rate below base
   rate (`trust = 1 - penalty*excess`, negative excess → trust > 1). Decide
   in advance whether to cap trust at some symmetric bound (e.g.
   `[1-penalty*cap, 1+penalty*cap]`) for interpretability, and whether to
   clip final chain-level S1 to `[0,1]` only AFTER aggregation (so grading
   isn't destroyed per-hop before it can accumulate). State the choice
   before measuring anything.
3. **Conditions to test**, same C1/C2 structure as OPTION9:
   - C1: continuous trust (shrinkage + unclipped excess) alone, baseline
     generator.
   - C2: continuous trust combined with OPTION8's injected shortcut-routing
     signal + hop-deficit feature.
   Freeze both before running either.
4. **Metric, stated first as always:** matched-TPR FPR at the paper's
   operating point (~92%) as primary; AUC for context only — you have four
   prior data points showing AUC alone would mislead here.

## Set honest expectations in the pre-commitment before running

State explicitly, before any result: even if this makes trust continuous,
it remains **organization-level** information — the hot organization's 88%
legitimate traffic still shares a hop with its 12% counterfeit traffic, and
no amount of continuous grading among the OTHER 11 organizations changes
that. The plausible best case is a graded discount for products with
longer "safe" stretches of their chain outside the hot organization, which
might shave some FPR at the margin; it is not expected to approach the
0.64 AUC ceiling already established for the ambiguous (touches-hot-org)
population. If the result beats that expectation, treat that as
noteworthy and double-check the held-out block before believing it.

## What to hand back

Both conditions' results on design seeds and the 67–71 held-out block, the
frozen pre-commitment, and an honest verdict. If this is a fifth null,
report it as plainly as the other four — five independent, honestly-run
mechanisms failing at the operating point, all traced to a common,
now-fully-understood cause, is a complete and rare kind of result. If it
helps, hold to the same rule as every prior attempt: don't report a
design-seed number before the held-out confirmation lands.
