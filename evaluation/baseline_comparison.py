"""Recall-localization estimate (calculated from assumptions, not observed).

Release 1.1.0: the conference-reported "Table VI baseline comparison"
(96.8 / 78.2 / 41.7 / 23.4% and the associated times) and the helper that
generated jittered per-run samples around it were moved to
``historical/bcca2026_reference.py``. Those values are BCCA 2026
conference-reported figures read from configuration, not results of this
repository, and the current evaluation no longer imports or reports them.

What remains here is the recall-localization estimate, which is
*calculated* from an assumed sustained throughput and the configured AI
inference overhead. It is not a timing observation.
"""
from __future__ import annotations

from typing import Any, Dict

RECALL_LOCALIZATION_BASIS = (
    "calculated: n_locations x (1/tps + ai_inference_overhead_ms/1000); "
    "tps is assumed, not measured on a deployed network; not an observed time"
)


def compute_recall_localization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate end-to-end recall-localization time across affected locations.

    ``per_location_seconds = 1/tps + ai_overhead_ms/1000``, with ``tps`` and
    ``n_locations`` from ``recall_localization_model`` and the overhead from
    ``blockchain.ai_inference_overhead_ms``.

    Returns:
        ``n_locations``, ``per_location_seconds``, ``bct_ai_time_minutes`` and
        ``basis`` (a statement that the value is calculated, not observed).
        Conference-reported comparison figures are available only from
        ``historical.bcca2026_reference.reference_recall_localization``.
    """
    model = config["recall_localization_model"]
    n_locations = model["n_locations"]
    tps = model["tps"]
    ai_overhead_ms = config["blockchain"]["ai_inference_overhead_ms"]

    per_location_seconds = (1.0 / tps) + (ai_overhead_ms / 1000.0)
    total_minutes = (n_locations * per_location_seconds) / 60.0

    return {
        "n_locations": n_locations,
        "per_location_seconds": round(per_location_seconds, 4),
        "bct_ai_time_minutes": round(total_minutes, 1),
        "basis": RECALL_LOCALIZATION_BASIS,
    }
