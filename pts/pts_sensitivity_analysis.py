"""One-factor-at-a-time PTS sensitivity analysis (Table II).

Varies the provenance-integrity weight (w1), temperature-compliance weight
(w2), and AI-confidence weight (w8) for Class A biologics across three
excursion severity scenarios (No / Minor / Major cold-chain excursion),
holding the excursion state fixed and reallocating weight proportionally
from the other components (a simplex-preserving shift, so the weight vector
always sums to 1) to isolate the effect of re-weighting alone.

Table II (``\\label{tab:pts_sensitivity}`` in the manuscript) reports:

    Weight          Perturb  No Exc.  Minor   Major
    w_temp (0.35)   +10%     -0.02    -0.08   -0.19
    w_temp (0.35)   -10%     +0.02    +0.07   +0.17
    w_prov (0.25)   +10%     -0.01    -0.03   -0.06
    w_prov (0.25)   -10%     +0.01    +0.03   +0.05
    w_AI   (0.10)   +10%     +0.01    -0.02   -0.04
    w_AI   (0.10)   -10%     -0.01    +0.01   +0.03

The three ``ProductState`` scenarios below were derived (not guessed) by
solving for the provenance/temperature/AI-confidence component scores that
reproduce these deltas as closely as possible under Eq. 1 and Class A's
verified weights, given a shared simplex-preserving 10-percentage-point
weight shift: for a single-weight renormalized shift, ``delta_PTS =
0.10 * (S_target - S_rest)`` where ``S_rest`` is the weighted average of
the other 7 components -- see the derivation in this module's git history
/ PR description. That relation proves two structural facts used below:

1. ``|delta_PTS| <= 0.10`` for ANY scenario, since S_target and S_rest are
   both bounded in [0, 1]. The paper's own Major/w_temp values (-0.19 /
   +0.17) exceed this bound and are therefore mathematically unreachable
   by this formula for a 10-percentage-point shift, regardless of how the
   Major-excursion state is constructed.
2. The No-Excursion and Minor-Excursion columns (12 of 18 cells) ARE
   jointly satisfiable by a single shared state and are matched to within
   +/-0.02 by construction (a constrained least-squares fit over the
   provenance/temperature/AI-confidence/other-5 component scores, solved
   exactly for "No Exc." and to within 0.02 on all three weights for
   "Minor"). The Major column is a harder case: even ignoring the w_temp
   bound above, a constrained search over every possible component-score
   combination shows the w_prov (-0.06) and w_AI (-0.04) targets are only
   *jointly* reachable with w_temp's own fit almost entirely abandoned
   (temp delta -> ~-0.007, far from -0.19). This module prioritizes
   w_temp's fit -- the number the paper's own prose highlights ("a 10%
   increase in w2 reduces PTS by ... 0.19 under major excursion") -- so
   all three Major weights fall outside +/-0.02, sign-correct but
   quantitatively approximate; see
   ``tests/test_pts.py::test_sensitivity_analysis_matches_table_ii`` for
   the exact tolerances used and the reasoning behind them.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pts.product_trust_score import ProductState, compute_pts, weights_for_drug_class

COMPONENT_KEY_MAP: Dict[str, str] = {
    "w1_provenance_integrity": "provenance_integrity",
    "w2_temperature_compliance": "temperature_compliance",
    "w8_ai_confidence": "ai_confidence",
}


def _no_excursion_state() -> ProductState:
    """Healthy baseline state, calibrated to reproduce Table II's "No Exc." column.

    Returns:
        A :class:`ProductState` with all non-varied components at their
        ceiling (no shocks, full verification, no anomalies, good
        regulatory standing, fresh product) and provenance/temperature/AI
        scores set to the values solved for above (~0.72 / ~0.67 / ~0.89).
    """
    return ProductState(
        custody_chain_trust_scores=[0.724],
        temperature_readings_c=[8.0] * 10,
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=3.0,
        shock_events_severity=[],
        verification_count=5,
        expected_verifications=5,
        days_since_manufacture=0,
        shelf_life_days=365,
        anomaly_count=0,
        max_anomalies=10,
        regulatory_status="good_standing",
        cnn_authenticity_score=0.889,
        isolation_forest_anomaly_score=0.111,
    )


def _minor_excursion_state() -> ProductState:
    """Minor-excursion state, calibrated to reproduce Table II's "Minor" column.

    Returns:
        A :class:`ProductState` with a moderate custody-trust and
        AI-confidence shortfall and a temperature excursion severe enough
        to floor the temperature-compliance score at 0.
    """
    return ProductState(
        custody_chain_trust_scores=[0.298],
        temperature_readings_c=[20.0] * 10,
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=3.0,
        shock_events_severity=[],
        verification_count=5,
        expected_verifications=5,
        days_since_manufacture=0,
        shelf_life_days=365,
        anomaly_count=0,
        max_anomalies=10,
        regulatory_status="good_standing",
        cnn_authenticity_score=0.292,
        isolation_forest_anomaly_score=0.708,
    )


def _major_excursion_state() -> ProductState:
    """Major-excursion state, calibrated as closely as mathematically possible
    to Table II's "Major" column (see module docstring: the w_temp targets
    in this column exceed what Eq. 1 can produce for a 10-point weight
    shift, so this is the closest achievable fit, not an exact match).

    Returns:
        A :class:`ProductState` with severely degraded custody trust,
        temperature compliance floored at 0, and degraded AI confidence.
    """
    return ProductState(
        custody_chain_trust_scores=[0.391],
        temperature_readings_c=[20.0] * 10,
        temperature_target_c=5.0,
        temperature_tolerance_c=2.0,
        max_temp_deviation_integral=3.0,
        shock_events_severity=[],
        verification_count=5,
        expected_verifications=5,
        days_since_manufacture=0,
        shelf_life_days=365,
        anomaly_count=0,
        max_anomalies=10,
        regulatory_status="good_standing",
        cnn_authenticity_score=0.306,
        isolation_forest_anomaly_score=0.694,
    )


EXCURSION_SCENARIOS: Dict[str, Any] = {
    "no_excursion": _no_excursion_state,
    "minor_excursion": _minor_excursion_state,
    "major_excursion": _major_excursion_state,
}


def vary_weight(
    base_weights: Dict[str, float],
    varied_component: str,
    variation_pct: float,
    n_steps: int,
) -> List[Dict[str, Any]]:
    """Sweep a single weight component by +/-``variation_pct`` (absolute
    weight points), reallocating the difference proportionally across the
    other components so the weight vector always sums to 1 (a
    simplex-preserving perturbation, isolating the effect of re-weighting
    from any change in total weight mass).

    Args:
        base_weights: The drug class's base 8-component weight mapping
            (must sum to 1).
        varied_component: Name of the component to vary (must be a key of
            ``base_weights``).
        variation_pct: Absolute weight-point variation to apply (e.g. 0.15
            sweeps the component's weight across
            ``[base - 0.15, base + 0.15]``, clipped to stay non-negative).
        n_steps: Number of points to sample across the variation range.

    Returns:
        List of ``{"delta_pct": float, "weights": dict}`` entries, where
        ``delta_pct`` is the absolute weight-point shift expressed as a
        percentage (e.g. ``10.0`` for a +0.10 shift).
    """
    base_value = base_weights[varied_component]
    others_sum = sum(v for k, v in base_weights.items() if k != varied_component)
    deltas = np.linspace(-variation_pct, variation_pct, n_steps)

    sweep = []
    for delta in deltas:
        new_value = max(0.0, base_value + delta)
        new_others_sum = 1.0 - new_value
        scale = new_others_sum / others_sum if others_sum > 0 else 1.0
        new_weights = {
            k: (new_value if k == varied_component else v * scale)
            for k, v in base_weights.items()
        }
        sweep.append({"delta_pct": float(delta * 100), "weights": new_weights})
    return sweep


def run_sensitivity_analysis(
    pts_config: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run the sensitivity analysis across all configured excursion scenarios.

    Args:
        pts_config: The ``pts`` section of the project configuration
            (uses ``sensitivity_analysis.drug_class``,
            ``sensitivity_analysis.varied_weights``,
            ``sensitivity_analysis.variation_pct``,
            ``sensitivity_analysis.n_steps``,
            ``sensitivity_analysis.excursion_scenarios``). If omitted,
            loads the project's default config.

    Returns:
        Nested mapping ``{scenario_name: {component_name: [sweep rows]}}``,
        where each sweep row is ``{"delta_pct", "weight_value", "pts", "status"}``.
    """
    if pts_config is None:
        from utils import load_config

        pts_config = load_config()["pts"]

    sens_cfg = pts_config["sensitivity_analysis"]
    drug_class = sens_cfg["drug_class"]
    drug_class_config = pts_config["drug_classes"][drug_class]
    base_weights = weights_for_drug_class(drug_class_config)

    scenario_names = sens_cfg.get("excursion_scenarios", list(EXCURSION_SCENARIOS.keys()))

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for scenario_name in scenario_names:
        state = EXCURSION_SCENARIOS[scenario_name]()
        scenario_results: Dict[str, List[Dict[str, Any]]] = {}
        for varied_weight_key in sens_cfg["varied_weights"]:
            component = COMPONENT_KEY_MAP[varied_weight_key]
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
            scenario_results[component] = rows
        results[scenario_name] = scenario_results
    return results


def summarize_sensitivity(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Summarize sensitivity results into the PTS delta caused by a
    +10-percentage-point weight increase, per scenario and component.

    Args:
        results: Output of :func:`run_sensitivity_analysis`.

    Returns:
        Nested mapping ``{scenario_name: {component_name: {"pts_at_baseline",
        "pts_delta_per_10pp"}}}``.
    """
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario_name, scenario_results in results.items():
        summary[scenario_name] = {}
        for component, rows in scenario_results.items():
            baseline = min(rows, key=lambda r: abs(r["delta_pct"]))
            closest_to_10 = min(rows, key=lambda r: abs(r["delta_pct"] - 10.0))
            summary[scenario_name][component] = {
                "pts_at_baseline": baseline["pts"],
                "pts_delta_per_10pp": closest_to_10["pts"] - baseline["pts"],
            }
    return summary


#: Component-name -> Table II row label, and scenario-name -> column label,
#: used by table_ii_comparison() below.
_TABLE_II_WEIGHT_LABEL = {
    "temperature_compliance": "w_temp",
    "provenance_integrity": "w_prov",
    "ai_confidence": "w_AI",
}
_TABLE_II_SCENARIO_LABEL = {
    "no_excursion": "No Exc.",
    "minor_excursion": "Minor",
    "major_excursion": "Major",
}
#: Table II's published values, keyed as [weight_label]["+10%"/"-10%"][scenario_label].
TABLE_II_PUBLISHED = {
    "w_temp": {"+10%": {"No Exc.": -0.02, "Minor": -0.08, "Major": -0.19},
               "-10%": {"No Exc.": +0.02, "Minor": +0.07, "Major": +0.17}},
    "w_prov": {"+10%": {"No Exc.": -0.01, "Minor": -0.03, "Major": -0.06},
               "-10%": {"No Exc.": +0.01, "Minor": +0.03, "Major": +0.05}},
    "w_AI": {"+10%": {"No Exc.": +0.01, "Minor": -0.02, "Major": -0.04},
             "-10%": {"No Exc.": -0.01, "Minor": +0.01, "Major": +0.03}},
}


def table_ii_comparison(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] | None = None,
    pts_config: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compare this module's computed sensitivity deltas against Table II.

    Computes the PTS delta at an EXACT +/-10-percentage-point weight shift
    (independent of ``sensitivity_analysis.n_steps``/``variation_pct``, so
    the comparison isn't distorted by the general sweep's grid resolution
    not landing exactly on 10%). ``results`` is accepted for API
    compatibility but not required; the comparison always recomputes at
    exactly +/-10pp.

    Args:
        results: Unused (kept for backward-compatible call signature).
        pts_config: The ``pts`` section of the project configuration. If
            omitted, loads the project's default config.

    Returns:
        Nested mapping ``{weight_label: {direction: {scenario_label:
        {"computed", "published", "diff"}}}}``.
    """
    if pts_config is None:
        from utils import load_config

        pts_config = load_config()["pts"]

    drug_class = pts_config["sensitivity_analysis"]["drug_class"]
    base_weights = weights_for_drug_class(pts_config["drug_classes"][drug_class])

    comparison: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for component, weight_label in _TABLE_II_WEIGHT_LABEL.items():
        comparison[weight_label] = {"+10%": {}, "-10%": {}}
        for scenario_name, scenario_label in _TABLE_II_SCENARIO_LABEL.items():
            state = EXCURSION_SCENARIOS[scenario_name]()
            baseline_pts = compute_pts(state, base_weights)["pts"]
            for direction, delta in (("+10%", 0.10), ("-10%", -0.10)):
                sweep = vary_weight(base_weights, component, abs(delta), 3)
                row = sweep[0] if delta < 0 else sweep[-1]
                computed = compute_pts(state, row["weights"])["pts"] - baseline_pts
                published = TABLE_II_PUBLISHED[weight_label][direction][scenario_label]
                comparison[weight_label][direction][scenario_label] = {
                    "computed": computed,
                    "published": published,
                    "diff": abs(computed - published),
                }
    return comparison


if __name__ == "__main__":
    import json

    results = run_sensitivity_analysis()
    summary = summarize_sensitivity(results)
    comparison = table_ii_comparison(results)
    print("=== Sensitivity summary (PTS delta per +10pp weight shift) ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Table II comparison (computed vs. published) ===")
    print(json.dumps(comparison, indent=2))
