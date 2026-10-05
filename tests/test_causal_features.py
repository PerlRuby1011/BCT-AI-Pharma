"""Prefix invariance: appending FUTURE events never changes earlier features."""
import numpy as np
import pandas as pd

from evaluation.causal_features import causal_features, prior_flag_counts


def _rows(spec):
    return pd.DataFrame(
        [{"transaction_id": f"tx{i}", "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=d),
          "product_id": p, "from_org": o} for i, (d, p, o) in enumerate(spec)])


TRUST = {"good": 1.0, "bad": 0.7, "ok": 0.9}


def test_reviewer_counterexample_first_event_unchanged_by_later_handler():
    one = _rows([(0, "P", "good")])
    two = _rows([(0, "P", "good"), (5, "P", "bad")])
    assert causal_features(one, TRUST)["s1"].iloc[0] == 1.0
    assert causal_features(two, TRUST)["s1"].iloc[0] == 1.0      # was 0.7 before the fix
    assert abs(causal_features(two, TRUST)["s1"].iloc[1] - 0.7) < 1e-12


def test_prefix_invariance_random_histories():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 40))
        spec = [(float(rng.uniform(0, 100)), f"P{rng.integers(0, 4)}",
                 rng.choice(list(TRUST) + ["unknown"])) for _ in range(n)]
        full = _rows(spec).sort_values("timestamp").reset_index(drop=True)
        k = int(rng.integers(1, n))
        f_full = causal_features(full, TRUST).iloc[:k]
        f_pref = causal_features(full.iloc[:k], TRUST)
        pd.testing.assert_frame_equal(f_full, f_pref)


def test_regulatory_and_counts_are_cumulative():
    f = causal_features(_rows([(0, "P", "good"), (1, "P", "bad"), (2, "P", "good")]), TRUST)
    assert list(f["low_trust_seen"]) == [False, True, True]
    assert list(f["n_events"]) == [1, 2, 3]
    assert list(f["elapsed_days"]) == [0.0, 1.0, 2.0]


def test_prior_flags_strictly_before_and_prefix_invariant():
    rows = _rows([(0, "P", "good"), (1, "P", "good"), (2, "P", "good"), (3, "Q", "good")])
    flags = np.array([1, 0, 1, 1])
    assert list(prior_flag_counts(rows, flags)) == [0, 1, 1, 0]
    assert list(prior_flag_counts(rows.iloc[:2], flags[:2])) == [0, 1]
