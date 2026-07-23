"""Injects labeled anomalies into synthetic blockchain transaction data.

Reproduces the paper's 5% anomaly injection scheme: counterfeit products,
temperature excursions, custody breaks, tampered packages, and unauthorized
transfers, in the paper's exact proportions (scaled to whatever total
transaction count is supplied).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

PAPER_ANOMALY_COUNTS: Dict[str, int] = {
    "counterfeit_product": 23000,
    "temperature_excursion": 46000,
    "custody_break": 34500,
    "tampered_package": 11500,
    "unauthorized_transfer": 5750,
}
PAPER_TOTAL_TRANSACTIONS = 2_300_000


def _anomaly_proportions(anomaly_counts: Dict[str, int]) -> Dict[str, float]:
    """Convert absolute paper-scale anomaly counts into proportions of the total.

    Args:
        anomaly_counts: Mapping of anomaly type -> absolute count at paper scale.

    Returns:
        Mapping of anomaly type -> proportion of ``PAPER_TOTAL_TRANSACTIONS``.
    """
    return {k: v / PAPER_TOTAL_TRANSACTIONS for k, v in anomaly_counts.items()}


def inject_anomalies(
    transactions: pd.DataFrame,
    anomaly_counts: Dict[str, int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject labeled anomalies into a transaction DataFrame, in-place-safe.

    Args:
        transactions: DataFrame produced by
            :func:`simulation.transaction_generator.generate_transactions`.
        anomaly_counts: Paper-scale absolute anomaly counts (keyed by
            ``counterfeit_products``, ``temperature_excursions``,
            ``custody_breaks``, ``tampered_packages``,
            ``unauthorized_transfers``). Counts are rescaled proportionally
            to the size of ``transactions``. Defaults to the paper's values.
        seed: Random seed for reproducible anomaly placement.

    Returns:
        A copy of ``transactions`` with ``is_anomaly`` and ``anomaly_type``
        columns populated, plus an ``anomaly_severity`` column in [0, 1].
    """
    if anomaly_counts is None:
        anomaly_counts = {
            "counterfeit_products": 23000,
            "temperature_excursions": 46000,
            "custody_breaks": 34500,
            "tampered_packages": 11500,
            "unauthorized_transfers": 5750,
        }

    key_map = {
        "counterfeit_products": "counterfeit_product",
        "temperature_excursions": "temperature_excursion",
        "custody_breaks": "custody_break",
        "tampered_packages": "tampered_package",
        "unauthorized_transfers": "unauthorized_transfer",
    }
    normalized_counts = {key_map.get(k, k): v for k, v in anomaly_counts.items()}
    proportions = _anomaly_proportions(normalized_counts)

    rng = np.random.default_rng(seed)
    df = transactions.copy()
    n = len(df)

    scaled_counts = {k: int(round(p * n)) for k, p in proportions.items()}
    total_anomalies = sum(scaled_counts.values())
    if total_anomalies > n:
        raise ValueError("Requested anomaly counts exceed available transactions")

    all_idx = rng.permutation(n)
    df["is_anomaly"] = False
    df["anomaly_type"] = "none"
    df["anomaly_severity"] = 0.0

    cursor = 0
    for anomaly_type, count in scaled_counts.items():
        idx = all_idx[cursor: cursor + count]
        cursor += count
        df.loc[df.index[idx], "is_anomaly"] = True
        df.loc[df.index[idx], "anomaly_type"] = anomaly_type
        severity = rng.uniform(0.4, 1.0, size=count)
        df.loc[df.index[idx], "anomaly_severity"] = severity

        if anomaly_type == "temperature_excursion":
            excursion = rng.choice([-1, 1], size=count) * rng.uniform(4.0, 12.0, size=count)
            df.loc[df.index[idx], "temperature_c"] = (
                df.loc[df.index[idx], "temperature_c"].to_numpy() + excursion
            )

    return df


def inject_anomalies_from_config(
    transactions: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Convenience wrapper that reads anomaly counts/seed from a config dict.

    Args:
        transactions: DataFrame of generated transactions.
        config: Full project configuration (as loaded from ``config.yaml``).

    Returns:
        Transactions DataFrame with anomalies injected.
    """
    sim_cfg = config["simulation"]
    return inject_anomalies(
        transactions,
        anomaly_counts=sim_cfg["anomaly_counts"],
        seed=config.get("random_seed", 42),
    )
