"""Per-transaction live-inference evaluation (pre-registered redesign).

Implements the protocol frozen in
``Journal Journey/OPTION2_live_inference_PREREGISTRATION.txt`` (approved
2026-07-28). This module is ADDITIVE: it does not modify, wrap, or alter
the behaviour of :mod:`evaluation.run_simulation`, whose published results
remain the paper's primary reported experiment.

The purpose is to make system-level counterfeit detection causally depend
on trained-model quality. The existing per-seed evaluation computes its
PTS inputs from closed-form severity proxies::

    cnn_authenticity = clip(1.0 - 0.30*severity_norm + N(0, 0.06))
    isolation_score  = clip(N(0.38, 0.13))
    custody_trust    = clip(1.0 - 0.90*severity_norm + N(0, 0.05))

so no trained model influences the reported detection rate. Here all three
are replaced:

- **S1 provenance** (protocol 3.1) is the product of organization trust
  scores along each product's realized custody chain, where trust is
  derived from counterfeits attributed to that organization in *prior
  periods only* (temporally causal; see :func:`compute_org_reputation`).
- **CNN authenticity** (protocol 3.2) is a live forward pass over a
  severity-blended packaging tensor generated per transaction.
- **Isolation Forest score** (protocol 3.3) is live inference over a
  per-transaction cold-chain reading sequence.

Anomaly injection, transaction generation, PTS weights and formula, and
both thresholds are untouched (protocol 3.4).
"""
from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_modules.cnn_verification import CLASS_NAMES
from ai_modules.isolation_forest_detector import (
    _apply_anomaly,
    _generate_normal_readings,
    normalized_anomaly_scores,
)
from pts.product_trust_score import ProductState, compute_pts

# --- Frozen protocol constants (pre-registration section 10a) ---------------
PROVENANCE_PENALTY = 0.10
PROVENANCE_EXCESS_CAP = 3.0
BURN_IN_FRACTION = 0.5
CNN_MAX_BLEND = 0.60
IF_SEQUENCE_LENGTH = 20
TENSOR_BATCH_SIZE = 256

# Tamper classes assigned per anomaly type (protocol 3.2). Index 0 is
# "Authentic" in CLASS_NAMES.
_TAMPER_CLASSES: Dict[str, Tuple[int, ...]] = {
    "counterfeit_product": (1, 2),   # Hologram Mismatch / Printing Defects
    "tampered_package": (3, 4),      # Seal Broken / Package Resealed
}


def _tx_seed(transaction_id: str) -> int:
    """Derive a deterministic per-transaction RNG seed from its identifier.

    Protocol 8(b) requires that each transaction's feature tensor be
    reproducible and independent of batching order. CRC32 is used because
    it is stable across platforms and Python versions, unlike ``hash()``.

    Args:
        transaction_id: The transaction's ``transaction_id`` field.

    Returns:
        A non-negative 31-bit integer seed.
    """
    return zlib.crc32(transaction_id.encode("utf-8")) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Protocol 3.1 — provenance from realized custody chains
# ---------------------------------------------------------------------------
def split_timeline(transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the transaction table chronologically into burn-in and evaluation halves.

    Args:
        transactions: Transaction table with a ``timestamp`` column.

    Returns:
        Tuple of ``(burn_in, evaluation)``, ordered by timestamp.
    """
    ordered = transactions.sort_values("timestamp").reset_index(drop=True)
    split = int(len(ordered) * BURN_IN_FRACTION)
    return ordered.iloc[:split], ordered.iloc[split:]


def compute_org_reputation(
    burn_in: pd.DataFrame,
    penalty: float = PROVENANCE_PENALTY,
    cap: float = PROVENANCE_EXCESS_CAP,
    shrinkage: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Derive per-organization trust from prior-period counterfeit history.

    An organization is penalized only for the *excess* of its counterfeit
    rate over the network base rate, so an organization no worse than
    baseline carries no penalty::

        trust = 1 - penalty * clip((org_rate - base_rate) / base_rate, 0, cap)

    Normalizing by the maximum observed rate instead (an earlier
    formulation, rejected by the feasibility spike) makes every
    organization absorb nearly the full penalty under uniform anomaly
    injection, which compounds over a ~12-handoff chain into a uniformly
    low S1 and pushes every product below the alert threshold regardless
    of severity. See pre-registration gate R8.

    Trust never reads the label of any row it will later be applied to:
    only ``burn_in`` transactions contribute, and metrics are computed
    exclusively on the evaluation half.

    ``shrinkage`` (OPTION5 pre-registration, approved 2026-08-24, default
    OFF): when enabled, ``org_rate`` is empirical-Bayes shrunk toward
    ``base_rate`` before the excess/trust calculation, weighted by how much
    burn-in evidence exists for that organization::

        k           = median(n_org)            # across orgs in this burn-in
        shrunk_rate = (n_org * org_rate + k * base_rate) / (n_org + k)

    ``k`` is derived automatically from this call's own burn-in data (never
    tuned against evaluation-half outcomes), so this stays leak-free even
    when reused on already-registered seeds. Responds to the finding that
    the published live-inference arms' false-positive rate is
    provenance-driven, not classifier-driven (``run_fpr_diagnostic.py``,
    2026-08-24: ARM4 92.29% TPR / 12.34% FPR; ARM5 87.53% TPR / 12.33% FPR,
    FPR unmoved by CNN degradation). Default is ``False`` so every existing
    caller, and every previously reported result, is reproduced exactly
    unless a caller opts in.

    Args:
        burn_in: The chronologically earlier half of the transaction table.
        penalty: Maximum trust reduction applied at or above ``cap`` excess.
        cap: Upper bound on normalized excess counterfeit rate.
        shrinkage: If True, apply the OPTION5 empirical-Bayes correction
            described above before computing trust. Default False preserves
            the exact registered/published behaviour.

    Returns:
        Tuple of ``(trust_by_org, base_rate)``.
    """
    is_counterfeit = (burn_in["anomaly_type"] == "counterfeit_product").astype(float)
    base_rate = float(is_counterfeit.mean())
    if base_rate <= 0:
        return {org: 1.0 for org in burn_in["from_org"].unique()}, 0.0

    org_rate = is_counterfeit.groupby(burn_in["from_org"]).mean()
    if shrinkage:
        n_org = burn_in.groupby("from_org").size()
        n_org, org_rate = n_org.align(org_rate, join="right")
        k = float(n_org.median())
        org_rate = (n_org * org_rate + k * base_rate) / (n_org + k)
    excess = ((org_rate - base_rate) / base_rate).clip(lower=0.0, upper=cap)
    trust = (1.0 - penalty * excess).to_dict()
    return trust, base_rate


def compute_provenance_scores(
    evaluation: pd.DataFrame, trust: Dict[str, float],
    aggregation: str = "product",
) -> np.ndarray:
    """Compute S1 by aggregating custody-chain trust scores per product.

    The realized custody chain for a product is the ordered sequence of
    organizations that handled it. Computed in log space to avoid underflow
    on long chains.

    Two aggregations are available:

    ``"product"`` (default, the registered and published behaviour)
        S1 is the raw product of chain trust scores, as
        :func:`pts.product_trust_score.score_provenance_integrity` specifies.

    ``"geometric_mean"`` (OPTION7)
        S1 is the *chain-length-normalized* product,
        ``(prod_i t_i) ** (1 / n_hops)``. This corrects a directional error
        in the raw product: every additional hop multiplies the score down
        again, so the raw product penalizes long chains, but longer chains
        are empirically *less* likely to be counterfeit in this generator
        (measured on seeds 42-44 under clustered injection: mean chain
        length 6.93 for counterfeit rows against 7.67 for clean, giving an
        AUC of 0.42 for chain length as a counterfeit score, i.e. inverted).
        Normalizing removes chain length from the score, leaving it a
        monotone function of the *fraction* of hops that are low-trust
        rather than their count.

    Because trust here is near-binary in practice --- one low-trust
    organization under clustered injection, the rest at 1.0 --- the two
    aggregations reduce to ranking by low-trust hop *count* and low-trust
    hop *fraction* respectively.

    Args:
        evaluation: The chronologically later half of the transaction table.
        trust: Mapping of organization id to trust score.
        aggregation: ``"product"`` (default) or ``"geometric_mean"``.

    Returns:
        Array aligned to ``evaluation`` rows giving each row's product-level S1.

    Raises:
        ValueError: If ``aggregation`` is not a recognized value.
    """
    if aggregation not in ("product", "geometric_mean"):
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    default = float(np.mean(list(trust.values()))) if trust else 1.0
    log_trust = evaluation["from_org"].map(
        lambda org: np.log(max(trust.get(org, default), 1e-9))
    )
    grouped = log_trust.groupby(evaluation["product_id"])
    chain_log_sum = grouped.transform("sum")

    if aggregation == "geometric_mean":
        n_hops = grouped.transform("size")
        chain_log_sum = chain_log_sum / n_hops.clip(lower=1)

    return np.clip(np.exp(chain_log_sum.to_numpy()), 0.0, 1.0)


def compute_hop_deficit_factor(
    burn_in: pd.DataFrame, evaluation: pd.DataFrame, beta: float = 0.30,
) -> np.ndarray:
    """OPTION8: per-product custody-chain shortfall, as a bounded S1 multiplier.

    Counterfeit product entering at the wholesale tier traverses fewer
    upstream handoffs than a legitimate product of the same class, so an
    unusually short custody chain is an observable red flag. This computes,
    per product::

        deficit(p) = median_chain_length(class(p)) - chain_length(p)
        factor(p)  = 1 - beta * clip(deficit / median_chain_length, 0, 1)

    and returns ``factor`` aligned to ``evaluation`` rows, for use as a
    multiplier on S1.

    The class median is estimated on the BURN-IN half only, so no
    evaluation-half data informs the reference. The feature reads
    ``product_id``, ``drug_class`` and custody-row counts --- never
    ``is_anomaly`` or ``anomaly_type``.

    ``beta`` is fixed at 0.30 by pre-commitment (the largest value leaving a
    maximally-shortcut product with a non-zero provenance score), not
    selected against any detection outcome. See
    ``preregistration/OPTION8_product_level_signal_PRECOMMITMENT.txt``.

    Args:
        burn_in: Chronologically earlier half, supplying the class reference.
        evaluation: Chronologically later half, scored.
        beta: Maximum proportional trust reduction at full deficit.

    Returns:
        Array aligned to ``evaluation`` rows, each in ``[1 - beta, 1]``.
    """
    bi_len = burn_in.groupby(["product_id"]).size()
    bi_class = burn_in.groupby("product_id")["drug_class"].first()
    ref = pd.DataFrame({"n": bi_len, "drug_class": bi_class})
    class_median = ref.groupby("drug_class")["n"].median()
    overall_median = float(ref["n"].median()) if len(ref) else 1.0

    chain_len = evaluation.groupby("product_id")["from_org"].transform("size").to_numpy()
    med = evaluation["drug_class"].map(class_median).fillna(overall_median).to_numpy()
    med = np.where(med <= 0, overall_median if overall_median > 0 else 1.0, med)

    deficit = np.clip((med - chain_len) / med, 0.0, 1.0)
    return 1.0 - beta * deficit


# ---------------------------------------------------------------------------
# Protocol 3.2 — live CNN inference on severity-blended tensors
# ---------------------------------------------------------------------------
class _PackagingSignatures:
    """The fixed per-class colour/texture signatures used by the image generator.

    Reproduces the constants in
    :func:`ai_modules.cnn_verification.generate_synthetic_packaging_images`,
    which derives them from a fixed internal seed (0) so that every caller
    agrees on what each class looks like.
    """

    def __init__(self, image_size: int, n_classes: int = len(CLASS_NAMES)) -> None:
        signature_rng = np.random.default_rng(0)
        self.signatures = signature_rng.uniform(0.2, 0.8, size=(n_classes, 3))
        self.frequencies = signature_rng.uniform(2.0, 8.0, size=n_classes)
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, image_size), np.linspace(0, 1, image_size)
        )
        self.textures = [
            0.15 * np.sin(self.frequencies[c] * (grid_x + grid_y) * np.pi)
            for c in range(n_classes)
        ]
        self.image_size = image_size


def _assign_tamper_class(anomaly_type: str, rng: np.random.Generator) -> int:
    """Map an anomaly type to the packaging class its tensor is drawn from."""
    options = _TAMPER_CLASSES.get(anomaly_type)
    return int(rng.choice(options)) if options else 0


def _build_tensor(
    tamper_class: int,
    severity_norm: float,
    signatures: _PackagingSignatures,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one packaging tensor, blended toward Authentic by low severity.

    A low-severity counterfeit is blended up to ``CNN_MAX_BLEND`` toward the
    Authentic signature, so that it is genuinely harder to classify. This is
    what makes the classifier's output a graded function of severity rather
    than a lookup of the assigned class label (pre-registration gate R7).

    Args:
        tamper_class: Index into the packaging class signatures.
        severity_norm: Normalized anomaly severity in [0, 1].
        signatures: Fixed per-class signature bank.
        rng: Per-transaction deterministic generator.

    Returns:
        Float32 array of shape ``(3, size, size)`` with values in [0, 1].
    """
    size = signatures.image_size
    blend = (1.0 - severity_norm) * CNN_MAX_BLEND
    signature = (1 - blend) * signatures.signatures[tamper_class] + blend * signatures.signatures[0]
    texture = (1 - blend) * signatures.textures[tamper_class] + blend * signatures.textures[0]
    base = signature[:, None, None] * np.ones((3, size, size))
    noise = rng.normal(0, 0.08, size=(3, size, size))
    return np.clip(base + texture[None, :, :] + noise, 0.0, 1.0).astype(np.float32)


def compute_cnn_authenticity(
    model: Any,
    rows: pd.DataFrame,
    severity_norm: np.ndarray,
    image_size: int,
    batch_size: int = TENSOR_BATCH_SIZE,
) -> np.ndarray:
    """Score every row through the trained CNN, streaming in batches.

    Materializing one 64x64x3 float32 tensor per anomaly row costs ~594 MB
    per seed and ~5.9 GB across ten seeds, so tensors are generated and
    discarded batch by batch (pre-registration section 8, constraint (a)).

    Args:
        model: Trained packaging-verification CNN (on CPU).
        rows: Anomaly-labelled transactions to score.
        severity_norm: Normalized severities aligned to ``rows``.
        image_size: Runtime tensor edge length.
        batch_size: Rows generated and scored per batch.

    Returns:
        Array of P(Authentic) values aligned to ``rows``.
    """
    import torch

    signatures = _PackagingSignatures(image_size)
    anomaly_types = rows["anomaly_type"].to_numpy()
    transaction_ids = rows["transaction_id"].to_numpy()
    n = len(rows)
    scores = np.empty(n, dtype=np.float64)

    model.eval()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = np.empty((stop - start, 3, image_size, image_size), dtype=np.float32)
        for offset, i in enumerate(range(start, stop)):
            rng = np.random.default_rng(_tx_seed(transaction_ids[i]))
            tamper_class = _assign_tamper_class(anomaly_types[i], rng)
            batch[offset] = _build_tensor(
                tamper_class, float(severity_norm[i]), signatures, rng
            )
        with torch.no_grad():
            _, logits = model(torch.from_numpy(batch))
            probabilities = torch.softmax(logits, dim=1)[:, 0].numpy()
        scores[start:stop] = probabilities
    return scores


# ---------------------------------------------------------------------------
# Protocol 3.3 — live Isolation Forest inference
# ---------------------------------------------------------------------------
def compute_isolation_scores(
    model: Any,
    rows: pd.DataFrame,
    calibration: Tuple[float, float],
    sequence_length: int = IF_SEQUENCE_LENGTH,
    batch_size: int = 2048,
) -> np.ndarray:
    """Score a per-transaction cold-chain reading sequence through the forest.

    Each transaction is given its own reading sequence, generated from its
    ``transaction_id`` seed. Rows labelled ``temperature_excursion`` have the
    corresponding anomaly signature applied. The per-transaction score is the
    mean normalized score over the sequence, using the model's fixed
    calibration bounds so scores stay comparable across batches.

    Args:
        model: Trained Isolation Forest.
        rows: Anomaly-labelled transactions to score.
        calibration: ``(raw_low, raw_high)`` bounds from the trained model.
        sequence_length: Readings generated per transaction.
        batch_size: Transactions per scoring batch.

    Returns:
        Array of normalized anomaly scores in [0, 1] aligned to ``rows``.
    """
    anomaly_types = rows["anomaly_type"].to_numpy()
    transaction_ids = rows["transaction_id"].to_numpy()
    n = len(rows)
    scores = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        blocks = []
        for i in range(start, stop):
            rng = np.random.default_rng(_tx_seed(transaction_ids[i]) ^ 0x5EED)
            block = _generate_normal_readings(sequence_length, rng)
            if anomaly_types[i] == "temperature_excursion":
                block = _apply_anomaly(block, "Gradual Drift", rng)
            blocks.append(block)
        stacked = np.concatenate(blocks, axis=0)
        normalized = normalized_anomaly_scores(model, stacked, calibration)
        scores[start:stop] = normalized.reshape(stop - start, sequence_length).mean(axis=1)
    return scores


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------
def _pts_metrics(
    s1: np.ndarray,
    cnn_authenticity: np.ndarray,
    isolation_score: np.ndarray,
    drug_classes: np.ndarray,
    pts_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Compute BCT-AI and AI-Only detection/recall from live PTS inputs.

    Uses the published PTS weights, formula, and thresholds unchanged
    (protocol 3.4). BCT-AI weights provenance and AI confidence by the
    drug class's configured w1/w8; AI-Only places all weight on AI
    confidence, exactly as the existing evaluation does.
    """
    n = len(s1)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0, "bct_ai_detection": nan, "ai_only_detection": nan,
            "bct_ai_recall": nan, "ai_only_recall": nan,
        }

    ai_only_weights = {"ai_confidence": 1.0}
    bct_ai_pts = np.empty(n)
    ai_only_pts = np.empty(n)

    for i in range(n):
        state = ProductState(
            custody_chain_trust_scores=[float(s1[i])],
            temperature_readings_c=[],
            cnn_authenticity_score=float(cnn_authenticity[i]),
            isolation_forest_anomaly_score=float(isolation_score[i]),
        )
        drug_class = drug_classes[i] if drug_classes[i] in ("A", "B", "C") else "C"
        class_cfg = pts_cfg["drug_classes"][drug_class]
        bct_ai_weights = {
            "provenance_integrity": class_cfg["w1_provenance_integrity"],
            "ai_confidence": class_cfg["w8_ai_confidence"],
        }
        bct_ai_pts[i] = compute_pts(state, bct_ai_weights)["pts"]
        ai_only_pts[i] = compute_pts(state, ai_only_weights)["pts"]

    alert = pts_cfg["alert_threshold"]
    quarantine = pts_cfg["quarantine_threshold"]
    return {
        "n": n,
        "bct_ai_detection": float(np.mean(bct_ai_pts < alert)),
        "ai_only_detection": float(np.mean(ai_only_pts < alert)),
        "bct_ai_recall": float(np.mean(bct_ai_pts < quarantine)),
        "ai_only_recall": float(np.mean(ai_only_pts < quarantine)),
    }


def run_live_inference_seed(
    config: Dict[str, Any],
    seed: int,
    transactions: pd.DataFrame,
    cnn_model: Any,
    if_model: Any,
    if_calibration: Tuple[float, float],
) -> Dict[str, Any]:
    """Evaluate one seed under the live-inference protocol.

    Args:
        config: Full project configuration.
        seed: Seed identifying this run (recorded, not re-drawn here).
        transactions: Generated and anomaly-injected transaction table.
        cnn_model: Trained packaging-verification CNN.
        if_model: Trained Isolation Forest.
        if_calibration: Fixed score-normalization bounds for ``if_model``.

    Returns:
        Per-seed metrics, plus diagnostic distributions required by gates
        R7 and R8.
    """
    pts_cfg = config["pts"]
    image_size = config["cnn"]["runtime_image_size"]

    burn_in, evaluation = split_timeline(transactions)
    trust, base_rate = compute_org_reputation(burn_in)
    s1_all = compute_provenance_scores(evaluation, trust)

    anomaly_mask = evaluation["is_anomaly"].to_numpy()
    anomaly_rows = evaluation.loc[anomaly_mask].copy()
    s1_anomaly = s1_all[anomaly_mask]

    severity = anomaly_rows["anomaly_severity"].to_numpy()
    severity_norm = np.clip((severity - 0.4) / 0.6, 0.0, 1.0)

    cnn_scores = compute_cnn_authenticity(
        cnn_model, anomaly_rows, severity_norm, image_size
    )
    isolation_scores = compute_isolation_scores(if_model, anomaly_rows, if_calibration)

    counterfeit_mask = (anomaly_rows["anomaly_type"] == "counterfeit_product").to_numpy()
    drug_classes = anomaly_rows["drug_class"].to_numpy()

    counterfeit_stats = _pts_metrics(
        s1_anomaly[counterfeit_mask], cnn_scores[counterfeit_mask],
        isolation_scores[counterfeit_mask], drug_classes[counterfeit_mask], pts_cfg,
    )
    recall_stats = _pts_metrics(
        s1_anomaly, cnn_scores, isolation_scores, drug_classes, pts_cfg
    )

    cnn_counterfeit = cnn_scores[counterfeit_mask]
    degenerate_mass = (
        float(np.mean((cnn_counterfeit < 0.05) | (cnn_counterfeit > 0.95)))
        if len(cnn_counterfeit) else float("nan")
    )
    severity_corr = (
        float(np.corrcoef(cnn_counterfeit, severity_norm[counterfeit_mask])[0, 1])
        if len(cnn_counterfeit) > 1 else float("nan")
    )

    return {
        "seed": seed,
        "n_evaluation_transactions": int(len(evaluation)),
        "n_counterfeit": counterfeit_stats["n"],
        "n_anomalies": recall_stats["n"],
        "bct_ai_counterfeit_detection": counterfeit_stats["bct_ai_detection"],
        "ai_only_counterfeit_detection": counterfeit_stats["ai_only_detection"],
        "bct_ai_recall_efficiency": recall_stats["bct_ai_recall"],
        "ai_only_recall_efficiency": recall_stats["ai_only_recall"],
        "diagnostics": {
            "base_counterfeit_rate": base_rate,
            "org_trust_min": float(min(trust.values())) if trust else float("nan"),
            "org_trust_max": float(max(trust.values())) if trust else float("nan"),
            "org_trust_spread": (
                float(max(trust.values()) - min(trust.values())) if trust else float("nan")
            ),
            "s1_mean_counterfeit": float(s1_anomaly[counterfeit_mask].mean()),
            "s1_mean_other": float(s1_anomaly[~counterfeit_mask].mean()),
            "cnn_mean_counterfeit": float(cnn_counterfeit.mean()) if len(cnn_counterfeit) else float("nan"),
            "cnn_degenerate_mass": degenerate_mass,
            "cnn_severity_corr": severity_corr,
            "isolation_mean": float(isolation_scores.mean()),
        },
    }
