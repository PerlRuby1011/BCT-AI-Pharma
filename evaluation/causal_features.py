"""Round-2 review: decision-time (causal) PTS features.

The first live-inference scripts gave every row of a product the provenance
of its COMPLETE evaluation-period custody chain, so a later handler changed an
earlier decision. Here every feature of a row uses only that product's events
with (timestamp, transaction_id) <= the row's own -- including burn-in events
(custody history is carried forward). Appending later events therefore cannot
change any earlier feature (prefix invariance; see tests/test_causal_features.py).

Assumption stated in the paper: organisation trust is estimated from burn-in
counterfeit outcomes, i.e. historical cases are adjudicated before the
evaluation period starts.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def order(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic chronological order (ties broken by transaction id)."""
    return df.sort_values(["timestamp", "transaction_id"], kind="mergesort")


def causal_features(history: pd.DataFrame, trust: Dict[str, float]) -> pd.DataFrame:
    """Per-row features computed from the product's history up to that row.

    Args:
        history: All custody rows (burn-in + evaluation) of the products of
            interest; needs ``timestamp``, ``transaction_id``, ``product_id``,
            ``from_org``.
        trust: Organisation -> trust in (0, 1], from burn-in only.

    Returns:
        DataFrame indexed like ``history`` with
        ``s1``            product of hop trusts so far (provenance, Eq. S1),
        ``n_events``      custody events observed so far (incl. this one),
        ``elapsed_days``  days since the product's FIRST OBSERVED custody event
                          (not true product age -- manufacture date is not simulated),
        ``low_trust_seen`` any handler so far with trust < 1 (regulatory input).
    """
    h = order(history)
    default = float(np.mean(list(trust.values()))) if trust else 1.0
    hop_trust = h["from_org"].map(lambda o: trust.get(o, default)).astype(float)
    s1 =np.exp(np.log(hop_trust.clip(lower=1e-9)).groupby(h["product_id"], sort=False).cumsum())
    n_events = h.groupby("product_id", sort=False).cumcount() + 1
    first_ts = pd.to_datetime(h["timestamp"]).groupby(h["product_id"], sort=False).cummin()
    elapsed = (pd.to_datetime(h["timestamp"]) - first_ts).dt.total_seconds() / 86400.0
    low = (hop_trust < 1.0).astype(int).groupby(h["product_id"], sort=False).cummax().astype(bool)
    out = pd.DataFrame({"s1": s1.clip(0.0, 1.0), "n_events": n_events,
                        "elapsed_days": elapsed, "low_trust_seen": low}, index=h.index)
    return out.reindex(history.index)


def prior_flag_counts(scored: pd.DataFrame, flagged: np.ndarray) -> np.ndarray:
    """S6 input: model-flagged events of the same product strictly before each row.

    ``scored`` must contain EVERY monitored event of each product from the
    monitoring start (product-level sampling), so no unscored event is
    silently treated as unflagged.
    """
    s = order(scored.assign(_f=np.asarray(flagged, dtype=int)))
    prior = s.groupby("product_id", sort=False)["_f"].cumsum() - s["_f"]
    return prior.reindex(scored.index).to_numpy()
