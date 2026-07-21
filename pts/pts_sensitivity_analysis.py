"""One-factor-at-a-time PTS sensitivity analysis (Section III-C.4).

Varies the provenance-integrity weight (w1) and temperature-compliance
weight (w2) by +/-15% for Class A biologics while holding the other weights
constant (rescaled so total weight always sums correctly), and reports the
resulting change in PTS under a representative moderate-temperature-excursion
scenario.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pts.product_trust_score import ProductState, compute_pts, weights_for_drug_class


def _moderate_excursion_state() -> ProductState:
    """Build a representative product state with a moderate cold-chain excursion.

    Returns:
        A :class:`ProductState` with realistic mid-severity temperature and
        handling deviations, used as the fixed scenario for sensitivity sweeps.
    """
    return ProductState(
        custody_chain_trust_scores=[0.98, 0.95, 0.97],
        temperature_readings_c=[5.0, 5.2, 7.5, 8.1, 7.8, 6.0, 5.1],
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=10.0,
        shock_events_severity=[0.1, 0.05],
        max_allowed_shocks=5.0,
        verification_count=2,
        expected_verifications=3,
        days_since_manufacture=90,
        shelf_life_days=730,
        anomaly_count=1,
        max_anomalies=5,
        regulatory_status="good_standing",
        cnn_authenticity_score=0.97,
        isolation_forest_anomaly_score=0.2,
    )


def vary_weight(
    base_weights: Dict[str, float],
    varied_component: str,
    variation_pct: float,
    n_steps: int,
) -> List[Dict[str, Any]]:
    """Sweep a single weight component by +/-``variation_pct`` around its base value.

    The varied component's weight changes linearly across ``n_steps`` points
    spanning ``[base * (1 - variation_pct), base * (1 + variation_pct)]``;
    all other weights are held at their base values (the PTS formula's
    division by the weight sum keeps the score well-defined regardless of
    whether the total is renormalized).

    Args:
        base_weights: The drug class's base 8-component weight mapping.
        varied_component: Name of the component to vary (must be a key of
            ``base_weights``).
        variation_pct: Fractional variation to apply (e.g. 0.15 for +/-15%).
        n_steps: Number of points to sample across the variation range.

    Returns:
        List of ``{"delta_pct": float, "weights": dict}`` entries.
    """
    base_value = base_weights[varied_component]
    deltas = np.linspace(-variation_pct, variation_pct, n_steps)
    sweep = []
    for delta in deltas:
        new_weights = dict(base_weights)
        new_weights[varied_component] = base_value * (1 + delta)
        sweep.append({"delta_pct": float(delta * 100), "weights": new_weights})
    return sweep


def run_sensitivity_analysis(pts_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Run the full sensitivity analysis for the configured drug class and weights.

    Args:
        pts_config: The ``pts`` section of the project configuration
            (uses ``sensitivity_analysis.drug_class``,
            ``sensitivity_analysis.varied_weights``,
            ``sensitivity_analysis.variation_pct``,
            ``sensitivity_analysis.n_steps``).

    Returns:
        Mapping of varied-weight-component name to a list of
        ``{"delta_pct", "weight_value", "pts", "status"}`` result rows.
    """
    sens_cfg = pts_config["sensitivity_analysis"]
    drug_class = sens_cfg["drug_class"]
    drug_class_config = pts_config["drug_classes"][drug_class]
    base_weights = weights_for_drug_class(drug_class_config)
    state = _moderate_excursion_state()

    component_key_map = {
        "w1_provenance_integrity": "provenance_integrity",
        "w2_temperature_compliance": "temperature_compliance",
    }

    results: Dict[str, List[Dict[str, Any]]] = {}
    for varied_weight_key in sens_cfg["varied_weights"]:
        component = component_key_map[varied_weight_key]
        sweep = vary_weight(
            base_weights, component, sens_cfg["variation_pct"], sens_cfg["n_steps"]
        )
        rows = []
        for point in sweep:
            outcome = compute_pts(state, point["weights"])
            rows.append(
                {
                    "delta_pct": point["delta_pct"],
                    "weight_value": point["weights"][component],
                    "pts": outcome["pts"],
                    "status": outcome["status"],
                }
            )
        results[component] = rows
    return results


def summarize_sensitivity(
    results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """Summarize sensitivity results into the PTS delta caused by a +10% weight increase.

    Args:
        results: Output of :func:`run_sensitivity_analysis`.

    Returns:
        Mapping of component name -> ``{"pts_at_baseline", "pts_delta_per_10pct"}``.
    """
    summary = {}
    for component, rows in results.items():
        baseline = min(rows, key=lambda r: abs(r["delta_pct"]))
        closest_to_10 = min(rows, key=lambda r: abs(r["delta_pct"] - 10.0))
        summary[component] = {
            "pts_at_baseline": baseline["pts"],
            "pts_delta_per_10pct": closest_to_10["pts"] - baseline["pts"],
        }
    return summary
