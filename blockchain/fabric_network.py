"""Simulated 12-node Hyperledger Fabric network performance model.

A real Fabric deployment is not available in this environment, so this
module statistically simulates endorsement/ordering/commit latency and
transaction success rate under varying offered load (TPS), calibrated to
the empirical benchmarks reported in Table III (pure blockchain latency, no
AI inference). Latencies are drawn from normal distributions parameterized
by the paper's reported mean/std at each load level; transaction
success/failure is drawn Bernoulli-style from the paper's reported success
rate.

The paper separately states a mean AI-inference overhead of 82 ms per
transaction -- a distinct comparison from the Table III sweep, which this
module also reproduces via
:meth:`FabricNetworkSimulator.ai_inference_latency_impact` (see that
method's docstring for the paper's exact wording and the caveat on what
this simulation actually produces for that comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

#: Four private channels (paper Section III-A), each scoped to the
#: organization types authorized to transact on it.
CHANNELS: Dict[str, List[str]] = {
    "manufacturing": ["manufacturers", "regulators"],
    "distribution": ["manufacturers", "distributors", "regulators"],
    "logistics": ["distributors", "logistics", "regulators"],
    "dispensing": ["distributors", "pharmacies", "regulators"],
}


class UnauthorizedChannelAccessError(Exception):
    """Raised when an organization attempts a transaction on a channel its
    org type is not authorized for (paper: built-in MSP identity management)."""


@dataclass
class FabricNetworkConfig:
    """Configuration for the simulated Fabric network.

    Attributes:
        n_nodes: Number of peer nodes in the network.
        n_organizations: Number of participating organizations.
        performance_targets: List of per-TPS-level target latency/success
            dictionaries, as found in ``config.yaml`` under
            ``blockchain.performance_targets``.
        consensus: Consensus protocol name (paper: Raft).
        ai_inference_overhead_ms: Mean per-transaction AI-inference latency
            overhead layered on top of blockchain-only latency.
        channels: Mapping of channel name -> list of authorized org types.
    """

    n_nodes: int
    n_organizations: int
    performance_targets: List[Dict[str, Any]]
    consensus: str = "raft"
    ai_inference_overhead_ms: float = 82.0
    channels: Dict[str, List[str]] = field(default_factory=lambda: dict(CHANNELS))


class FabricNetworkSimulator:
    """Simulates transaction throughput/latency/success behavior of a
    permissioned Hyperledger Fabric network across a 12-node, 5-organization
    topology (Section IV-A, Table III), including channel-scoped MSP
    identity checks and the separately-reported AI-inference overhead.
    """

    def __init__(self, config: FabricNetworkConfig, seed: int = 42) -> None:
        """Initialize the simulator.

        Args:
            config: Network topology and calibration targets.
            seed: Random seed for reproducible latency/success sampling.
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.msp_registry: Dict[str, str] = {}
        logger.info(
            "Initialized Fabric network: consensus=%s, n_nodes=%d, channels=%s",
            config.consensus,
            config.n_nodes,
            list(config.channels.keys()),
        )

    def register_organization(self, org_id: str, org_type: str) -> None:
        """Register an organization's identity with the (simulated) MSP.

        Args:
            org_id: Unique organization identifier.
            org_type: Organization type (e.g. ``"manufacturers"``), used to
                authorize channel access.
        """
        self.msp_registry[org_id] = org_type

    def _node_overhead_ms(self) -> float:
        """Simplified per-transaction latency overhead scaling with node count.

        Not a paper-reported formula (the paper's Table III benchmarks were
        measured at a fixed 12-node topology); this makes the simulator
        responsive to a configured ``n_nodes`` other than 12, at a modest
        0.5 ms/node rate consistent with the paper's measured std bands.

        Returns:
            Overhead in milliseconds.
        """
        return self.config.n_nodes * 0.5

    def simulate_load_level(
        self,
        tps_target: Dict[str, Any],
        n_samples: int = 1000,
        include_ai_overhead: bool = True,
    ) -> pd.DataFrame:
        """Simulate ``n_samples`` transactions at a single offered-load level.

        Args:
            tps_target: One entry from ``performance_targets`` (contains
                ``tps``, ``avg_latency_ms``, ``avg_latency_std_ms``,
                ``p95_latency_ms``, ``p95_latency_std_ms``,
                ``success_rate_pct``).
            n_samples: Number of simulated transactions to draw.
            include_ai_overhead: If True, adds
                ``config.ai_inference_overhead_ms`` to each sampled latency,
                reproducing the paper's "BCT-AI Integrated" (blockchain +
                AI inference) condition rather than the blockchain-only
                Table III figures.

        Returns:
            DataFrame with columns ``tps_target``, ``latency_ms``, ``success``.
        """
        avg_latency = self.rng.normal(
            loc=tps_target["avg_latency_ms"],
            scale=max(tps_target["avg_latency_std_ms"], 1e-6),
            size=n_samples,
        )
        avg_latency += self._node_overhead_ms()
        if include_ai_overhead:
            avg_latency += self.config.ai_inference_overhead_ms
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

    def run_benchmark(
        self, n_samples_per_level: int = 1000, include_ai_overhead: bool = False
    ) -> pd.DataFrame:
        """Simulate the full Table III benchmark sweep across all TPS levels.

        Defaults to ``include_ai_overhead=False`` because Table III itself
        (Section V-A, "Blockchain Network Performance") reports
        blockchain-only latency; see :meth:`ai_inference_latency_impact` for
        the paper's separate AI-inclusive comparison.

        Args:
            n_samples_per_level: Number of simulated transactions per TPS level.
            include_ai_overhead: Whether to include the AI-inference overhead.

        Returns:
            Concatenated DataFrame of simulated transactions across all
            configured TPS levels.
        """
        frames = [
            self.simulate_load_level(target, n_samples_per_level, include_ai_overhead)
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

    def ai_inference_latency_impact(self, n_samples: int = 2000, tps: int = 500) -> Dict[str, float]:
        """Reproduce the paper's AI-inference overhead claim (Section V-A):
        "AI inference added 82 ms mean overhead per transaction."

        Simulates the same TPS level with and without the configured
        ``ai_inference_overhead_ms`` term and reports both means plus their
        difference. At the default ``tps=500`` (Table III's lowest level,
        187+/-23 ms blockchain-only), this simulation's own run typically
        lands close to 194 ms blockchain-only vs. 274 ms BCT-AI-integrated
        (an ~81 ms difference) -- consistent with, but not a verbatim
        restatement of, the paper's single published overhead figure.

        Args:
            n_samples: Number of simulated transactions per condition.
            tps: TPS level to sample at (must match one of
                ``config.performance_targets``).

        Returns:
            Dictionary with ``blockchain_only_ms``, ``bct_ai_integrated_ms``,
            and ``overhead_ms``.

        Raises:
            ValueError: If ``tps`` does not match a configured TPS level.
        """
        target = next((t for t in self.config.performance_targets if t["tps"] == tps), None)
        if target is None:
            raise ValueError(f"No performance target configured for tps={tps}")

        blockchain_only = self.simulate_load_level(target, n_samples, include_ai_overhead=False)
        bct_ai = self.simulate_load_level(target, n_samples, include_ai_overhead=True)

        return {
            "blockchain_only_ms": float(blockchain_only["latency_ms"].mean()),
            "bct_ai_integrated_ms": float(bct_ai["latency_ms"].mean()),
            "overhead_ms": float(
                bct_ai["latency_ms"].mean() - blockchain_only["latency_ms"].mean()
            ),
        }

    def simulate_channel_transaction(
        self, channel_name: str, org_id: str, tps: int, n_samples: int = 1
    ) -> pd.DataFrame:
        """Route a transaction through a private channel, enforcing MSP authorization.

        Args:
            channel_name: One of ``config.channels`` (e.g. ``"dispensing"``).
            org_id: Organization submitting the transaction; must have been
                registered via :meth:`register_organization`.
            tps: TPS level to simulate at (must match a configured
                performance target).
            n_samples: Number of simulated transactions to draw.

        Returns:
            DataFrame from :meth:`simulate_load_level` for the matched TPS target.

        Raises:
            ValueError: If ``channel_name`` or ``tps`` is not configured.
            UnauthorizedChannelAccessError: If ``org_id`` is unregistered or
                its organization type is not authorized for ``channel_name``.
        """
        if channel_name not in self.config.channels:
            raise ValueError(
                f"Unknown channel '{channel_name}'; configured channels: "
                f"{list(self.config.channels.keys())}"
            )

        org_type = self.msp_registry.get(org_id)
        if org_type is None:
            raise UnauthorizedChannelAccessError(
                f"Organization '{org_id}' is not registered with the network MSP"
            )
        allowed_types = self.config.channels[channel_name]
        if org_type not in allowed_types:
            raise UnauthorizedChannelAccessError(
                f"Organization '{org_id}' (type={org_type}) is not authorized for "
                f"channel '{channel_name}' (allowed: {allowed_types})"
            )

        target = next((t for t in self.config.performance_targets if t["tps"] == tps), None)
        if target is None:
            raise ValueError(f"No performance target configured for tps={tps}")
        return self.simulate_load_level(target, n_samples)


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
        consensus=bc_cfg.get("consensus", "raft"),
        ai_inference_overhead_ms=bc_cfg.get("ai_inference_overhead_ms", 82.0),
        channels=bc_cfg.get("channels", dict(CHANNELS)),
    )
    return FabricNetworkSimulator(net_config, seed=config.get("random_seed", 42))
