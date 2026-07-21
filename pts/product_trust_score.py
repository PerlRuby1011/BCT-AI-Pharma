"""Product Trust Score (PTS): mathematical framework (Section III-C).

Implements the weighted composite trust score

    PTS(p, t) = sum_i(w_i * S_i(p, t)) / sum_i(w_i)

over 8 component dimensions, with drug-class-specific weight profiles, and
the paper's alert (< 0.75) and automated-quarantine (< 0.50) thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

COMPONENT_ORDER: List[str] = [
    "provenance_integrity",
    "temperature_compliance",
    "handling_quality",
    "verification_frequency",
    "age_shelf_life",
    "anomaly_history",
    "regulatory_status",
    "ai_confidence",
]


@dataclass
class ProductState:
    """Raw observations needed to compute all 8 PTS component scores.

    Attributes:
        custody_chain_trust_scores: Trust score (0-1) of each node the
            product has passed through, in custody order.
        temperature_readings_c: Time-ordered temperature observations in Celsius.
        temperature_target_c: Target storage temperature for this product.
        temperature_tolerance_c: Allowed deviation (delta_tolerance) before a
            reading counts against temperature compliance.
        max_temp_deviation_integral: Normalizing constant ``M`` for the
            temperature-compliance integral.
        shock_events_severity: Severity (0-1) of each recorded shock/vibration event.
        max_allowed_shocks: Normalizing constant for handling quality.
        verification_count: Number of verification events recorded for the product.
        expected_verifications: Expected number of verifications for full confidence.
        days_since_manufacture: Elapsed time since manufacture, in days.
        shelf_life_days: Product shelf life, in days (used as the half-life).
        anomaly_count: Number of detected anomalies associated with the product.
        max_anomalies: Normalizing constant for anomaly history.
        regulatory_status: One of ``"good_standing"``, ``"under_investigation"``,
            ``"sanctioned"``.
        cnn_authenticity_score: CNN packaging-authenticity score in [0, 1].
        isolation_forest_anomaly_score: Isolation Forest anomaly score in [0, 1]
            (0 = normal, 1 = highly anomalous).
    """

    custody_chain_trust_scores: List[float] = field(default_factory=lambda: [1.0])
    temperature_readings_c: List[float] = field(default_factory=list)
    temperature_target_c: float = 5.0
    temperature_tolerance_c: float = 2.0
    max_temp_deviation_integral: float = 10.0
    shock_events_severity: List[float] = field(default_factory=list)
    max_allowed_shocks: float = 5.0
    verification_count: int = 0
    expected_verifications: int = 3
    days_since_manufacture: float = 0.0
    shelf_life_days: float = 730.0
    anomaly_count: int = 0
    max_anomalies: int = 5
    regulatory_status: str = "good_standing"
    cnn_authenticity_score: float = 1.0
    isolation_forest_anomaly_score: float = 0.0


def score_provenance_integrity(state: ProductState) -> float:
    """Factor 1: provenance integrity, S_prov = product of custody-chain node trust scores.

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1].
    """
    score = 1.0
    for trust in state.custody_chain_trust_scores:
        score *= max(0.0, min(1.0, trust))
    return score


def score_temperature_compliance(state: ProductState) -> float:
    """Factor 2: temperature compliance, based on time-integrated deviation from target.

    Discretizes the paper's continuous integral
    ``S_temp = max(0, 1 - integral(max(0, |T-target| - tolerance) dt) / (M * duration))``
    over the recorded temperature readings (unit time step per reading).

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1]. Returns 1.0 if no readings are available.
    """
    if not state.temperature_readings_c:
        return 1.0
    excursions = [
        max(0.0, abs(t - state.temperature_target_c) - state.temperature_tolerance_c)
        for t in state.temperature_readings_c
    ]
    integral = sum(excursions)
    duration = len(state.temperature_readings_c)
    denom = state.max_temp_deviation_integral * duration
    if denom <= 0:
        return 1.0
    return max(0.0, 1.0 - integral / denom)


def score_handling_quality(state: ProductState) -> float:
    """Factor 3: handling quality, penalized by weighted shock/vibration events.

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1].
    """
    if state.max_allowed_shocks <= 0:
        return 1.0
    penalty = sum(state.shock_events_severity) / state.max_allowed_shocks
    return max(0.0, 1.0 - penalty)


def score_verification_frequency(state: ProductState) -> float:
    """Factor 4: verification frequency relative to the expected count.

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1].
    """
    if state.expected_verifications <= 0:
        return 1.0
    return min(1.0, state.verification_count / state.expected_verifications)


def score_age_shelf_life(state: ProductState) -> float:
    """Factor 5: age/shelf-life score, exponential decay with half-life = shelf life.

    Args:
        state: Product observation state.

    Returns:
        Score in (0, 1].
    """
    if state.shelf_life_days <= 0:
        return 0.0
    decay_rate = math.log(2) / state.shelf_life_days
    elapsed = max(0.0, state.days_since_manufacture)
    return math.exp(-decay_rate * elapsed)


def score_anomaly_history(state: ProductState) -> float:
    """Factor 6: anomaly history, penalized by prior detected anomalies.

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1].
    """
    if state.max_anomalies <= 0:
        return 1.0
    return max(0.0, 1.0 - state.anomaly_count / state.max_anomalies)


def score_regulatory_status(state: ProductState) -> float:
    """Factor 7: regulatory status, a categorical score based on custody-chain standing.

    Args:
        state: Product observation state.

    Returns:
        1.0 if all participants are in good standing, 0.5 if any is under
        investigation, 0.0 if any is sanctioned.

    Raises:
        ValueError: If ``regulatory_status`` is not a recognized value.
    """
    mapping = {"good_standing": 1.0, "under_investigation": 0.5, "sanctioned": 0.0}
    if state.regulatory_status not in mapping:
        raise ValueError(f"Unknown regulatory_status: {state.regulatory_status}")
    return mapping[state.regulatory_status]


def score_ai_confidence(state: ProductState) -> float:
    """Factor 8: AI confidence, average of CNN authenticity and inverse Isolation Forest score.

    Args:
        state: Product observation state.

    Returns:
        Score in [0, 1].
    """
    inverse_isolation = 1.0 - state.isolation_forest_anomaly_score
    return (state.cnn_authenticity_score + inverse_isolation) / 2.0


_COMPONENT_SCORERS = {
    "provenance_integrity": score_provenance_integrity,
    "temperature_compliance": score_temperature_compliance,
    "handling_quality": score_handling_quality,
    "verification_frequency": score_verification_frequency,
    "age_shelf_life": score_age_shelf_life,
    "anomaly_history": score_anomaly_history,
    "regulatory_status": score_regulatory_status,
    "ai_confidence": score_ai_confidence,
}


def compute_component_scores(state: ProductState) -> Dict[str, float]:
    """Compute all 8 PTS component scores for a product state.

    Args:
        state: Product observation state.

    Returns:
        Mapping of component name -> score in [0, 1], in :data:`COMPONENT_ORDER`.
    """
    return {name: _COMPONENT_SCORERS[name](state) for name in COMPONENT_ORDER}


def weights_for_drug_class(drug_class_config: Dict[str, Any]) -> Dict[str, float]:
    """Expand a drug-class config entry into 8 explicit component weights.

    ``w1`` (provenance), ``w2`` (temperature), and ``w8`` (AI confidence) are
    specified explicitly; ``remaining_total`` is split evenly across the
    other 5 components, matching the paper's weight-determination scheme
    (Section III-C.3).

    Args:
        drug_class_config: One entry from ``config.yaml``'s
            ``pts.drug_classes`` mapping.

    Returns:
        Mapping of all 8 component names to their weight.
    """
    remaining_components = [
        "handling_quality",
        "verification_frequency",
        "age_shelf_life",
        "anomaly_history",
        "regulatory_status",
    ]
    remaining_weight_each = drug_class_config["remaining_total"] / len(remaining_components)

    weights = {
        "provenance_integrity": drug_class_config["w1_provenance_integrity"],
        "temperature_compliance": drug_class_config["w2_temperature_compliance"],
        "ai_confidence": drug_class_config["w8_ai_confidence"],
    }
    for component in remaining_components:
        weights[component] = remaining_weight_each
    return weights


def compute_pts(
    state: ProductState, weights: Dict[str, float]
) -> Dict[str, Any]:
    """Compute the Product Trust Score and its component breakdown.

    Args:
        state: Product observation state.
        weights: Mapping of component name -> weight (need not be
            pre-normalized; PTS divides by the weight sum per the paper's formula).

    Returns:
        Dictionary with ``pts`` (float in [0, 1]), ``components`` (per-factor
        scores), ``status`` (``"quarantine"``, ``"alert"``, or ``"active"``
        using the default 0.50/0.75 thresholds).
    """
    components = compute_component_scores(state)
    weight_sum = sum(weights.get(name, 0.0) for name in COMPONENT_ORDER)
    if weight_sum <= 0:
        raise ValueError("Sum of PTS weights must be positive")

    weighted_sum = sum(weights.get(name, 0.0) * components[name] for name in COMPONENT_ORDER)
    pts = weighted_sum / weight_sum

    if pts < 0.50:
        status = "quarantine"
    elif pts < 0.75:
        status = "alert"
    else:
        status = "active"

    return {"pts": pts, "components": components, "status": status}


def compute_pts_for_drug_class(
    state: ProductState, drug_class: str, pts_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Convenience wrapper: compute PTS using a drug class's configured weights.

    Args:
        state: Product observation state.
        drug_class: One of ``"A"``, ``"B"``, ``"C"``.
        pts_config: The ``pts`` section of the project configuration.

    Returns:
        Same structure as :func:`compute_pts`.

    Raises:
        KeyError: If ``drug_class`` is not present in ``pts_config["drug_classes"]``.
    """
    drug_class_config = pts_config["drug_classes"][drug_class]
    weights = weights_for_drug_class(drug_class_config)
    result = compute_pts(state, weights)

    alert_threshold = pts_config.get("alert_threshold", 0.75)
    quarantine_threshold = pts_config.get("quarantine_threshold", 0.50)
    if result["pts"] < quarantine_threshold:
        result["status"] = "quarantine"
    elif result["pts"] < alert_threshold:
        result["status"] = "alert"
    else:
        result["status"] = "active"
    return result
