# Prompt — test a non-multiplicative PTS aggregation (4th attempt)

Three prior attempts (shrinkage, geometric-mean chain normalization,
injected product-level shortcut signal) all improved a ranking statistic
(AUC) but made matched-TPR FPR *worse* at the paper's registered operating
point. All three kept the same underlying structure: S1 as some function of
a *product* (raw product, normalized product) of per-hop trust scores. This
attempt changes that structure itself.

Same discipline as the last three: freeze the design and its justification
before measuring anything, hold out fresh seeds never used in this line of
experiments, report honestly regardless of outcome. Seeds 42–61 are all
consumed (design + two held-out blocks). **Use 62–66 as the next clean
held-out block; use whatever seeds you like below 62 for iteration.**

## Why this might be different in kind, not just degree

A raw or normalized *product* is AND-composition: the chain's score is only
as good as its worst hop, and every additional hop compounds the effect
(this is why chain length ran backwards in OPTION7, and why OPTION8's
injected per-product signal still got swamped — a single multiplicative
penalty term dominates the whole chain regardless of what else is true
about the product). Reputation and trust-propagation systems outside this
project commonly use weighted-average or evidence-accumulation composition
instead of strict AND-composition specifically to avoid one bad component
overwhelming otherwise-strong evidence (this is a real, citable design
choice in trust-model literature, not invented for this experiment — if
you want a citation anchor, EigenTrust-style and Dempster-Shafer-style
trust aggregation are both weighted/evidence-combining rather than
multiplicative-AND). Switching S1 from a product to a weighted mean of
per-hop trust scores is therefore a genuinely different security semantics,
not a parameter tweak on the same idea — which is also exactly why it needs
its own justification, separate from "does it fix the FPR."

## What to freeze, in writing, before any measurement

1. **The new aggregation formula.** Simplest defensible choice: unweighted
   arithmetic mean of per-hop trust scores,
   `S1 = mean(trust_1, ..., trust_n)`. If you want to weight hops (e.g. by
   recency, or by organization type), state the weighting rule and its
   justification NOW, before testing — don't add a weighting scheme after
   seeing whether the unweighted version helps.
2. **Whether to test with or without OPTION8's injected shortcut-routing
   signal.** Recommended: test BOTH, as two separate frozen conditions,
   because they isolate different questions:
   - Condition 1 (`aggregation_only`): new aggregation, baseline generator
     (`shortcut_routing=False`) — does changing composition alone help,
     using only information already available today?
   - Condition 2 (`aggregation_plus_signal`): new aggregation COMBINED with
     OPTION8's `shortcut_routing=True` + hop-deficit feature — does a
     non-multiplicative aggregation let the genuine per-product signal
     from OPTION8 finally show through, where the multiplicative
     aggregation drowned it?
   Freeze both before running either.
3. **The comparison metric.** Same as before: AUC for context, matched-TPR
   FPR at the paper's operating point (~92%) as the number that matters.
   State this before running, not after — you already know from three
   prior attempts that AUC alone would be misleading here, so there's no
   excuse to lead with it this time.

Write the above into a short pre-commitment file (same style as
`preregistration/OPTION8_product_level_signal_PRECOMMITMENT.txt`) before
touching code. Call it `OPTION9_nonmultiplicative_aggregation_PRECOMMITMENT.txt`.

## Implementation

`compute_provenance_scores()` in `evaluation/live_inference.py` currently
computes S1 via a log-sum-then-exp product. Add a new `composition`
parameter (`"product"` default, unchanged; add `"mean"` or your frozen
name) that instead averages trust scores per chain — plain arithmetic mean,
not log-space, since you're no longer composing multiplicatively. Keep the
existing `aggregation="geometric_mean"` chain-length-normalization option
available to combine with if useful, but don't conflate the two — be clear
in your report which specific formula produced which number.

## What to hand back

Both conditions' results (AUC and matched-TPR FPR) on the design seeds and
on the 62–66 held-out block, the frozen pre-commitment file, and — same as
every prior attempt — an honest verdict either way. If this is a fourth
null, say so plainly and note explicitly that four independent mechanisms
(estimation, chain-length aggregation, injected data signal, and now
composition itself) have all failed at the operating point, which would be
strong evidence the constraint is even more fundamental than "the
aggregation function" — possibly inherent to scoring counterfeit risk from
custody-chain membership at all, at this problem's actual class balance and
clustering structure. If it works, the same held-out-before-reporting rule
applies as always: don't let a promising design-seed number get reported
before the 62–66 confirmation lands.
