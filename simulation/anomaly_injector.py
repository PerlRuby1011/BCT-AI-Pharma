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
    cluster_counterfeit_by_manufacturer: bool = False,
    counterfeit_cluster_fraction: float = 0.3,
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
        cluster_counterfeit_by_manufacturer: If ``True``, counterfeit-product
            anomalies are preferentially placed on transactions originating
            from a random subset of manufacturer nodes (``counterfeit_cluster_fraction``
            of all manufacturers), modeling the realistic pattern that
            counterfeiting tends to originate from a small number of
            compromised sources rather than being spread uniformly at
            random across the whole network. Falls back to uniform random
            placement (the historical default) if there are not enough
            candidate transactions from the clustered manufacturers to fill
            the requested count.
        counterfeit_cluster_fraction: Fraction of manufacturer org IDs
            selected as the "compromised" cluster when
            ``cluster_counterfeit_by_manufacturer`` is enabled.

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

    df["is_anomaly"] = False
    df["anomaly_type"] = "none"
    df["anomaly_severity"] = 0.0

    available = np.ones(n, dtype=bool)

    for anomaly_type, count in scaled_counts.items():
        if count == 0:
            continue

        if anomaly_type == "counterfeit_product" and cluster_counterfeit_by_manufacturer:
            idx = _select_clustered_counterfeit_indices(
                df, available, count, counterfeit_cluster_fraction, rng
            )
        else:
            candidate_idx = np.flatnonzero(available)
            chosen = rng.choice(candidate_idx, size=count, replace=False)
            idx = chosen

        available[idx] = False
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


def _select_clustered_counterfeit_indices(
    df: pd.DataFrame,
    available: np.ndarray,
    count: int,
    cluster_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``count`` available row indices for counterfeit injection,
    preferring rows whose ``from_org`` belongs to a randomly selected subset
    of manufacturer org IDs (a "compromised cluster"), topping up with
    uniformly random available rows if the cluster doesn't have enough.

    Args:
        df: Transactions DataFrame (with ``from_org``/``from_org_type`` columns).
        available: Boolean mask of rows not yet claimed by another anomaly type.
        count: Number of indices to select.
        cluster_fraction: Fraction of manufacturer org IDs to treat as compromised.
        rng: NumPy random generator.

    Returns:
        Array of ``count`` row-position indices.
    """
    manufacturer_orgs = df.loc[df["from_org_type"] == "manufacturers", "from_org"].unique()
    if len(manufacturer_orgs) == 0:
        candidate_idx = np.flatnonzero(available)
        return rng.choice(candidate_idx, size=count, replace=False)

    n_cluster = max(1, int(round(len(manufacturer_orgs) * cluster_fraction)))
    clustered_orgs = set(rng.choice(manufacturer_orgs, size=n_cluster, replace=False))

    clustered_mask = available & df["from_org"].isin(clustered_orgs).to_numpy()
    clustered_idx = np.flatnonzero(clustered_mask)

    if len(clustered_idx) >= count:
        return rng.choice(clustered_idx, size=count, replace=False)

    # Not enough candidates in the cluster: take everything available there,
    # then top up with uniformly random remaining available rows.
    remaining_mask = available.copy()
    remaining_mask[clustered_idx] = False
    remaining_idx = np.flatnonzero(remaining_mask)
    top_up = rng.choice(remaining_idx, size=count - len(clustered_idx), replace=False)
    return np.concatenate([clustered_idx, top_up])


def inject_anomalies_from_config(
    transactions: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Convenience wrapper that reads anomaly counts/seed from a config dict.

    Honors the optional ``data_quality`` config section
    (``cluster_counterfeit_by_manufacturer``, ``counterfeit_cluster_fraction``)
    if present; falls back to uniform random placement otherwise.

    Args:
        transactions: DataFrame of generated transactions.
        config: Full project configuration (as loaded from ``config.yaml``).

    Returns:
        Transactions DataFrame with anomalies injected.
    """
    sim_cfg = config["simulation"]
    dq_cfg = config.get("data_quality", {})
    return inject_anomalies(
        transactions,
        anomaly_counts=sim_cfg["anomaly_counts"],
        seed=config.get("random_seed", 42),
        cluster_counterfeit_by_manufacturer=dq_cfg.get(
            "cluster_counterfeit_by_manufacturer", False
        ),
        counterfeit_cluster_fraction=dq_cfg.get("counterfeit_cluster_fraction", 0.3),
    )
