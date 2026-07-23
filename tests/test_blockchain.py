"""Unit tests for blockchain.fabric_network."""
from __future__ import annotations

import pytest

from blockchain.fabric_network import (
    CHANNELS,
    FabricNetworkConfig,
    FabricNetworkSimulator,
    UnauthorizedChannelAccessError,
)

PERFORMANCE_TARGETS = [
    {
        "tps": 500,
        "avg_latency_ms": 187,
        "avg_latency_std_ms": 23,
        "p95_latency_ms": 245,
        "p95_latency_std_ms": 31,
        "success_rate_pct": 100.0,
    }
]


def _build_simulator() -> FabricNetworkSimulator:
    config = FabricNetworkConfig(
        n_nodes=12,
        n_organizations=5,
        performance_targets=PERFORMANCE_TARGETS,
    )
    return FabricNetworkSimulator(config, seed=1)


def test_consensus_is_raft_not_pbft() -> None:
    simulator = _build_simulator()
    assert simulator.config.consensus == "raft"
    assert "byzantine" not in simulator.config.consensus.lower()
    assert "pbft" not in simulator.config.consensus.lower()


def test_all_four_channels_present() -> None:
    assert set(CHANNELS.keys()) == {"manufacturing", "distribution", "logistics", "dispensing"}
    for allowed_types in CHANNELS.values():
        assert "regulators" in allowed_types


def test_ai_inference_overhead_added_correctly() -> None:
    simulator = _build_simulator()
    without_overhead = simulator.simulate_load_level(
        PERFORMANCE_TARGETS[0], n_samples=5000, include_ai_overhead=False
    )
    with_overhead = simulator.simulate_load_level(
        PERFORMANCE_TARGETS[0], n_samples=5000, include_ai_overhead=True
    )
    mean_diff = with_overhead["latency_ms"].mean() - without_overhead["latency_ms"].mean()
    assert mean_diff == pytest.approx(82.0, abs=3.0)


def test_ai_inference_latency_impact_matches_paper_scale() -> None:
    simulator = _build_simulator()
    result = simulator.ai_inference_latency_impact(n_samples=5000, tps=500)
    assert result["overhead_ms"] == pytest.approx(82.0, abs=3.0)
    assert result["bct_ai_integrated_ms"] > result["blockchain_only_ms"]


def test_msp_rejects_unauthorized_org_type() -> None:
    simulator = _build_simulator()
    simulator.register_organization("pharmacy_001", "pharmacies")

    with pytest.raises(UnauthorizedChannelAccessError):
        simulator.simulate_channel_transaction("manufacturing", "pharmacy_001", tps=500)


def test_msp_rejects_unregistered_org() -> None:
    simulator = _build_simulator()
    with pytest.raises(UnauthorizedChannelAccessError):
        simulator.simulate_channel_transaction("dispensing", "unknown_org", tps=500)


def test_msp_allows_authorized_org_type() -> None:
    simulator = _build_simulator()
    simulator.register_organization("mfg_001", "manufacturers")
    result = simulator.simulate_channel_transaction("manufacturing", "mfg_001", tps=500, n_samples=10)
    assert len(result) == 10


def test_node_count_affects_latency() -> None:
    small_config = FabricNetworkConfig(n_nodes=4, n_organizations=5, performance_targets=PERFORMANCE_TARGETS)
    large_config = FabricNetworkConfig(n_nodes=40, n_organizations=5, performance_targets=PERFORMANCE_TARGETS)
    small_sim = FabricNetworkSimulator(small_config, seed=1)
    large_sim = FabricNetworkSimulator(large_config, seed=1)

    small_latency = small_sim.simulate_load_level(
        PERFORMANCE_TARGETS[0], n_samples=5000, include_ai_overhead=False
    )["latency_ms"].mean()
    large_latency = large_sim.simulate_load_level(
        PERFORMANCE_TARGETS[0], n_samples=5000, include_ai_overhead=False
    )["latency_ms"].mean()

    assert large_latency > small_latency
