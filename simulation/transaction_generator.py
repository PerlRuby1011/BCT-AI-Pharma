"""Synthetic pharmaceutical blockchain transaction generator.

Generates tabular transaction records that mimic what would be written to a
12-node Hyperledger Fabric network by 5 organization types (manufacturers,
distributors, logistics providers, regulators, pharmacies), matching the
paper's simulation environment. Generation is fully vectorized with NumPy
so that the paper-scale 2.3 million transaction count remains tractable on
a laptop.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

TRANSACTION_TYPES: List[str] = [
    "manufacture",
    "quality_check",
    "ship",
    "receive",
    "custody_transfer",
    "cold_chain_reading",
    "verify",
    "dispense",
]

DRUG_CLASSES: List[str] = ["A", "B", "C"]
DRUG_CLASS_WEIGHTS: List[float] = [0.15, 0.25, 0.60]


def _build_organizations(org_counts: Dict[str, int]) -> pd.DataFrame:
    """Build the roster of participating organizations (network nodes).

    Args:
        org_counts: Mapping of organization type -> count, e.g.
            ``{"manufacturers": 3, "distributors": 3, ...}``.

    Returns:
        DataFrame with one row per organization: ``org_id``, ``org_type``.
    """
    rows = []
    org_id = 0
    for org_type, count in org_counts.items():
        for _ in range(count):
            rows.append({"org_id": f"{org_type[:-1] if org_type.endswith('s') else org_type}_{org_id:03d}",
                         "org_type": org_type})
            org_id += 1
    return pd.DataFrame(rows)


def generate_transactions(
    n_transactions: int,
    org_counts: Dict[str, int],
    seed: int = 42,
    start_timestamp: str = "2025-01-01T00:00:00",
    temperature_mean_c: float = 5.0,
    temperature_std_c: float = 1.2,
    custody_timing_distribution: str = "uniform",
    custody_timing_lambda: float = 0.1,
) -> pd.DataFrame:
    """Generate synthetic pharmaceutical supply-chain blockchain transactions.

    Args:
        n_transactions: Total number of transactions to generate.
        org_counts: Organization-type counts (manufacturers, distributors,
            logistics, regulators, pharmacies).
        seed: Random seed for reproducibility.
        start_timestamp: ISO timestamp used as the simulation epoch.
        temperature_mean_c: Mean of the cold-chain temperature reading
            distribution (degrees Celsius). Realistic pharma cold-chain
            monitoring clusters tightly around a target (e.g. 5.0C for the
            2-8C range); narrowing ``temperature_std_c`` models
            better-calibrated sensors / tighter carrier compliance.
        temperature_std_c: Standard deviation of the cold-chain temperature
            reading distribution.
        custody_timing_distribution: ``"uniform"`` (transaction timestamps
            spread uniformly across the operational window, the historical
            default) or ``"exponential"`` (inter-transaction gaps drawn from
            an exponential distribution, modeling realistic bursty custody
            handoffs -- most transfers happen in quick succession with
            occasional long dwell times -- rather than a uniformly-paced
            stream).
        custody_timing_lambda: Rate parameter (events per second) for the
            exponential inter-arrival distribution when
            ``custody_timing_distribution="exponential"``. Mean gap is
            ``1 / custody_timing_lambda`` seconds.

    Returns:
        DataFrame of ``n_transactions`` rows with columns: ``transaction_id``,
        ``timestamp``, ``product_id``, ``drug_class``, ``from_org``,
        ``from_org_type``, ``to_org``, ``to_org_type``, ``transaction_type``,
        ``temperature_c``, ``quantity``, ``location``, ``is_anomaly``,
        ``anomaly_type``.
    """
    if n_transactions <= 0:
        raise ValueError("n_transactions must be positive")
    if custody_timing_distribution not in ("uniform", "exponential"):
        raise ValueError(
            f"Unknown custody_timing_distribution: {custody_timing_distribution!r}"
        )

    rng = np.random.default_rng(seed)
    orgs = _build_organizations(org_counts)
    n_orgs = len(orgs)
    if n_orgs < 2:
        raise ValueError("Need at least 2 organizations to generate transfers")

    org_ids = orgs["org_id"].to_numpy()
    org_types = orgs["org_type"].to_numpy()

    from_idx = rng.integers(0, n_orgs, size=n_transactions)
    # Ensure to_org differs from from_org (custody must transfer between nodes).
    to_offset = rng.integers(1, n_orgs, size=n_transactions)
    to_idx = (from_idx + to_offset) % n_orgs

    n_products = max(1, n_transactions // 12)
    product_ids = rng.integers(0, n_products, size=n_transactions)

    start = pd.Timestamp(start_timestamp)
    # Spread transactions over a simulated 18-month operational window.
    total_window_seconds = 18 * 30 * 24 * 3600
    if custody_timing_distribution == "exponential":
        # Bursty custody handoffs: cumulative exponential inter-arrival gaps,
        # rescaled to fill the same 18-month window as the uniform case so
        # the two distributions remain comparable in span.
        gaps = rng.exponential(scale=1.0 / custody_timing_lambda, size=n_transactions)
        cumulative = np.cumsum(gaps)
        offsets = (cumulative * (total_window_seconds / cumulative[-1])).astype(np.int64)
    else:
        offsets = np.sort(rng.integers(0, total_window_seconds, size=n_transactions))
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    transaction_types = rng.choice(TRANSACTION_TYPES, size=n_transactions)
    drug_classes = rng.choice(DRUG_CLASSES, size=n_transactions, p=DRUG_CLASS_WEIGHTS)

    # Cold-chain-relevant readings cluster around a 2-8C target range.
    base_temp = rng.normal(loc=temperature_mean_c, scale=temperature_std_c, size=n_transactions)
    ambient_mask = rng.random(n_transactions) < 0.15
    base_temp[ambient_mask] = rng.normal(loc=21.0, scale=2.0, size=ambient_mask.sum())

    quantities = rng.integers(1, 5000, size=n_transactions)
    locations = rng.choice(
        ["US-East", "US-West", "EU-Central", "EU-North", "APAC-East", "APAC-South"],
        size=n_transactions,
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"tx_{i:09d}" for i in range(n_transactions)],
            "timestamp": timestamps,
            "product_id": [f"prod_{p:07d}" for p in product_ids],
            "drug_class": drug_classes,
            "from_org": org_ids[from_idx],
            "from_org_type": org_types[from_idx],
            "to_org": org_ids[to_idx],
            "to_org_type": org_types[to_idx],
            "transaction_type": transaction_types,
            "temperature_c": np.round(base_temp, 2),
            "quantity": quantities,
            "location": locations,
        }
    )
    df["is_anomaly"] = False
    df["anomaly_type"] = "none"
    return df


def generate_transactions_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    """Convenience wrapper that reads generator parameters from a config dict.

    Honors the optional ``data_quality`` config section
    (``temperature_mean_c``, ``temperature_std_c``,
    ``custody_timing_distribution``, ``custody_timing_lambda``) if present;
    falls back to the historical defaults otherwise.

    Args:
        config: Full project configuration (as loaded from ``config.yaml``).

    Returns:
        DataFrame of generated transactions.
    """
    sim_cfg = config["simulation"]
    dq_cfg = config.get("data_quality", {})
    return generate_transactions(
        n_transactions=sim_cfg["n_transactions"],
        org_counts=sim_cfg["organizations"],
        seed=config.get("random_seed", 42),
        temperature_mean_c=dq_cfg.get("temperature_mean_c", 5.0),
        temperature_std_c=dq_cfg.get("temperature_std_c", 1.2),
        custody_timing_distribution=dq_cfg.get("custody_timing_distribution", "uniform"),
        custody_timing_lambda=dq_cfg.get("custody_timing_lambda", 0.1),
    )
