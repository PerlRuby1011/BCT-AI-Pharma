"""Simulated 12-node Hyperledger Fabric network performance model.

A real Fabric deployment is not available in this environment, so this
module statistically simulates endorsement/ordering/commit latency and
transaction success rate under varying offered load (TPS), calibrated to
the empirical benchmarks reported in Table III of the paper. Latencies are
drawn from normal distributions parameterized by the paper's reported
mean/std at each load level; transaction success/failure is drawn
Bernoulli-style from the paper's reported success rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class FabricNetworkConfig:
    """Configuration for the simulated Fabric network.

    Attributes:
        n_nodes: Number of peer nodes in the network.
        n_organizations: Number of participating organizations.
        performance_targets: List of per-TPS-level target latency/success
            dictionaries, as found in ``config.yaml`` under
            ``blockchain.performance_targets``.
    """

    n_nodes: int
    n_organizations: int
    performance_targets: List[Dict[str, Any]]


class FabricNetworkSimulator:
    """Simulates transaction throughput/latency/success behavior of a
    permissioned Hyperledger Fabric network across a 12-node, 5-organization
    topology (Section IV-A, Table III).
    """

    def __init__(self, config: FabricNetworkConfig, seed: int = 42) -> None:
        """Initialize the simulator.

        Args:
            config: Network topology and calibration targets.
            seed: Random seed for reproducible latency/success sampling.
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

    def simulate_load_level(self, tps_target: Dict[str, Any], n_samples: int = 1000) -> pd.DataFrame:
        """Simulate ``n_samples`` transactions at a single offered-load level.

        Args:
            tps_target: One entry from ``performance_targets`` (contains
                ``tps``, ``avg_latency_ms``, ``avg_latency_std_ms``,
                ``p95_latency_ms``, ``p95_latency_std_ms``,
                ``success_rate_pct``).
            n_samples: Number of simulated transactions to draw.

        Returns:
            DataFrame with columns ``tps_target``, ``latency_ms``, ``success``.
        """
        avg_latency = self.rng.normal(
            loc=tps_target["avg_latency_ms"],
            scale=max(tps_target["avg_latency_std_ms"], 1e-6),
            size=n_samples,
        )
        avg_latency = np.clip(avg_latency, a_min=1.0, a_max=None)

        success_prob = tps_target["success_rate_pct"] / 100.0
        success = self.rng.random(n_samples) < success_prob

        return pd.DataFrame(
            {
                "tps_target": tps_target["tps"],
                "latency_ms": avg_latency,
                "success": success,
            }
        )

    def run_benchmark(self, n_samples_per_level: int = 1000) -> pd.DataFrame:
        """Simulate the full Table III benchmark sweep across all TPS levels.

        Args:
            n_samples_per_level: Number of simulated transactions per TPS level.

        Returns:
            Concatenated DataFrame of simulated transactions across all
            configured TPS levels.
        """
        frames = [
            self.simulate_load_level(target, n_samples_per_level)
            for target in self.config.performance_targets
        ]
        return pd.concat(frames, ignore_index=True)

    def summarize_benchmark(self, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a raw benchmark run into per-TPS-level summary statistics.

        Args:
            benchmark_df: Output of :meth:`run_benchmark`.

        Returns:
            DataFrame with one row per TPS level: ``tps_target``,
            ``avg_latency_ms``, ``p95_latency_ms``, ``success_rate_pct``.
        """
        rows = []
        for tps, group in benchmark_df.groupby("tps_target"):
            rows.append(
                {
                    "tps_target": tps,
                    "avg_latency_ms": group["latency_ms"].mean(),
                    "p95_latency_ms": group["latency_ms"].quantile(0.95),
                    "success_rate_pct": group["success"].mean() * 100.0,
                }
            )
        return pd.DataFrame(rows).sort_values("tps_target").reset_index(drop=True)


def build_simulator_from_config(config: Dict[str, Any]) -> FabricNetworkSimulator:
    """Construct a :class:`FabricNetworkSimulator` from the project config.

    Args:
        config: Full project configuration (as loaded from ``config.yaml``).

    Returns:
        A configured :class:`FabricNetworkSimulator` instance.
    """
    bc_cfg = config["blockchain"]
    net_config = FabricNetworkConfig(
        n_nodes=bc_cfg["n_nodes"],
        n_organizations=bc_cfg["n_organizations"],
        performance_targets=bc_cfg["performance_targets"],
    )
    return FabricNetworkSimulator(net_config, seed=config.get("random_seed", 42))
