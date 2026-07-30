"""PTS calibration diagnostic: do Fixes A/B/C raise counterfeit detection?

Motivation
----------
The full 10-seed orchestrator run measured a BCT-AI counterfeit detection
rate of 71.4% +/- 1.4%, below the AI-Only Table VI baseline of 78.2%. Three
calibration changes were proposed:

  Fix A -- raise Class A's AI-confidence weight w8 from 0.10 to 0.20, with
           the residual 0.80 redistributed proportionally across w1, w2 and
           `remaining_total` so the 8 weights still sum to exactly 1.0.
  Fix B -- severity-graded quarantine override for Class A: force quarantine
           when the raw CNN authenticity score < 0.30 AND the
           temperature-compliance score S2 < 0.20, regardless of composite PTS.
  Fix C -- lower Class A's quarantine threshold from 0.50 to 0.45.

This script runs each fix independently, in pairs, and combined, over a
reduced diagnostic scale (5 seeds x 50,000 transactions), and reports both
headline metrics so the effect of each fix is attributable:

  detection rate = fraction of counterfeit-labeled products with PTS below
                   the ALERT threshold (0.75)
  recall rate    = fraction of counterfeit-labeled products quarantined
                   (PTS below the class quarantine threshold, or forced by
                   the Fix B override)

Two variants beyond the specified three are included:

  Fix C-up -- raise Class A's quarantine threshold to 0.55 instead of
              lowering it. The 0.518 excursion finding motivating Fix C is a
              case that FAILED to quarantine at 0.50; catching it requires a
              threshold above 0.518, so lowering to 0.45 moves away from the
              finding. C-up is measured alongside C so the direction of the
              effect is on the record rather than assumed.
  Fix B-obs -- Fix B with the CNN/temperature gates set to values reachable
              by the counterfeit-detection product states, used only to
              confirm that Fix B's null result is a coverage artifact and not
              a wiring bug.

No seed is excluded, no anomaly injection is modified, and every variant is
run over the identical seed set.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_simulation import run_single_simulation_run  # noqa: E402
from pts.product_trust_score import validate_drug_class_weights  # noqa: E402
from utils import load_config  # noqa: E402

N_SEEDS = 5
TRANSACTIONS_PER_RUN = 50_000
BASE_SEED = 42
AI_ONLY_BASELINE_PCT = 78.2


def apply_fix_a(config: Dict[str, Any], new_w8: float = 0.20) -> Dict[str, Any]:
    """Raise Class A's w8 to ``new_w8``, rescaling the residual proportionally.

    The three other weight groups (w1, w2, remaining_total) currently sum to
    ``1 - w8_old``; each is multiplied by ``(1 - new_w8) / (1 - w8_old)`` so
    their relative proportions are preserved and the total stays exactly 1.0.

    Args:
        config: Full project configuration (mutated in place).
        new_w8: New AI-confidence weight for Class A.

    Returns:
        The same config object, for chaining.
    """
    class_a = config["pts"]["drug_classes"]["A"]
    old_w8 = class_a["w8_ai_confidence"]
    scale = (1.0 - new_w8) / (1.0 - old_w8)
    for key in ("w1_provenance_integrity", "w2_temperature_compliance", "remaining_total"):
        class_a[key] = class_a[key] * scale
    class_a["w8_ai_confidence"] = new_w8
    validate_drug_class_weights(class_a)
    return config


def apply_fix_b(config: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Enable the Class A severity-graded quarantine override.

    Args:
        config: Full project configuration (mutated in place).
        **overrides: Optional gate overrides, e.g. ``cnn_authenticity_max``.

    Returns:
        The same config object, for chaining.
    """
    override_cfg = config["pts"].setdefault("quarantine_override", {})
    override_cfg["enabled"] = True
    override_cfg.setdefault("drug_classes", ["A"])
    override_cfg.setdefault("cnn_authenticity_max", 0.30)
    override_cfg.setdefault("temperature_compliance_max", 0.20)
    override_cfg.update(overrides)
    return config


def apply_fix_c(config: Dict[str, Any], threshold: float = 0.45) -> Dict[str, Any]:
    """Set a Class A-specific quarantine threshold.

    Args:
        config: Full project configuration (mutated in place).
        threshold: New Class A quarantine threshold (must stay below the
            alert threshold; enforced downstream by
            :func:`pts.product_trust_score.quarantine_threshold_for_drug_class`).

    Returns:
        The same config object, for chaining.
    """
    config["pts"]["drug_classes"]["A"]["quarantine_threshold"] = threshold
    return config


def build_variants(base_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Construct every calibration variant to be measured.

    Args:
        base_config: The unmodified project configuration.

    Returns:
        Mapping of variant label -> its own deep-copied configuration.
    """

    def fresh() -> Dict[str, Any]:
        cfg = copy.deepcopy(base_config)
        cfg["simulation"]["transactions_per_run"] = TRANSACTIONS_PER_RUN
        return cfg

    variants: Dict[str, Dict[str, Any]] = {}
    variants["baseline"] = fresh()
    variants["A (w8 0.10->0.20)"] = apply_fix_a(fresh())
    variants["B (quarantine override)"] = apply_fix_b(fresh())
    variants["C (quar. thr 0.50->0.45)"] = apply_fix_c(fresh())
    variants["C-up (quar. thr 0.50->0.55)"] = apply_fix_c(fresh(), threshold=0.55)
    variants["A+B"] = apply_fix_b(apply_fix_a(fresh()))
    variants["A+C"] = apply_fix_c(apply_fix_a(fresh()))
    variants["B+C"] = apply_fix_c(apply_fix_b(fresh()))
    variants["A+B+C (combined)"] = apply_fix_c(apply_fix_b(apply_fix_a(fresh())))
    variants["A+B+C-up"] = apply_fix_c(
        apply_fix_b(apply_fix_a(fresh())), threshold=0.55
    )
    # Coverage probe: gates set to values the counterfeit-detection product
    # states can actually reach, isolating "override never fires" (coverage)
    # from "override is not wired in" (bug).
    variants["B-obs (reachable gates)"] = apply_fix_b(
        fresh(), cnn_authenticity_max=0.90, temperature_compliance_max=1.01
    )
    return variants


def measure_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run all diagnostic seeds for one variant and aggregate its metrics.

    Args:
        config: A variant configuration.

    Returns:
        Dictionary of mean/std detection and recall rates plus per-seed values.
    """
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    runs = [run_single_simulation_run(config, seed) for seed in seeds]

    detection = np.array([r["bct_ai_counterfeit_detection"] for r in runs])
    recall = np.array([r["bct_ai_recall_efficiency"] for r in runs])
    ai_only = np.array([r["ai_only_counterfeit_detection"] for r in runs])

    return {
        "seeds": seeds,
        "n_counterfeit": int(runs[0]["n_counterfeit"]),
        "detection_mean": float(detection.mean()),
        "detection_std": float(detection.std(ddof=1)),
        "recall_mean": float(recall.mean()),
        "recall_std": float(recall.std(ddof=1)),
        "ai_only_detection_mean": float(ai_only.mean()),
        "per_seed_detection": detection.tolist(),
        "per_seed_recall": recall.tolist(),
    }


def main() -> None:
    """Run the diagnostic across all variants and print a comparison table."""
    base_config = load_config()
    for drug_class, class_cfg in base_config["pts"]["drug_classes"].items():
        validate_drug_class_weights(class_cfg)
        print(f"Baseline weight check: class {drug_class} sums to 1.0 OK")

    variants = build_variants(base_config)
    results: Dict[str, Dict[str, Any]] = {}

    for label, config in variants.items():
        print(f"\n=== {label} ===", flush=True)
        results[label] = measure_variant(config)
        r = results[label]
        print(
            f"  detection = {r['detection_mean'] * 100:.2f}% +/- "
            f"{r['detection_std'] * 100:.2f}%   "
            f"recall/quarantine = {r['recall_mean'] * 100:.2f}% +/- "
            f"{r['recall_std'] * 100:.2f}%",
            flush=True,
        )

    baseline_det = results["baseline"]["detection_mean"]
    baseline_rec = results["baseline"]["recall_mean"]

    header = (
        f"\n\n{'Variant':<30} {'Detection':>12} {'d vs base':>11} "
        f"{'Quarantine':>12} {'d vs base':>11} {'>78.2%?':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        det_pct = r["detection_mean"] * 100
        print(
            f"{label:<30} "
            f"{det_pct:>10.2f}% "
            f"{(det_pct - baseline_det * 100):>+10.2f} "
            f"{r['recall_mean'] * 100:>11.2f}% "
            f"{(r['recall_mean'] - baseline_rec) * 100:>+10.2f} "
            f"{('YES' if det_pct > AI_ONLY_BASELINE_PCT else 'no'):>9}"
        )

    out_path = REPO_ROOT / "results" / "pts_calibration_diagnostic.json"
    payload = {
        "n_seeds": N_SEEDS,
        "transactions_per_run": TRANSACTIONS_PER_RUN,
        "base_seed": BASE_SEED,
        "ai_only_baseline_pct": AI_ONLY_BASELINE_PCT,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
