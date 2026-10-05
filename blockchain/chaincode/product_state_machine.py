"""Unified product enforcement state machine (SC-5 / SC-7 / release).

Added 2026-10 in response to review: the original SC-7 logic set a product's
status from the latest Product Trust Score alone, so a quarantined product
returned to ``OK`` as soon as a later score rose above 0.75, and SC-5 (cold
chain) quarantined a *shipment* record that SC-7 never consulted. This
module makes the rules explicit:

States (most to least severe):
    DISPOSED   terminal; no further transitions
    QUARANTINE one or more open quarantine reasons
    ALERT      no open quarantine, but a current alert condition
    ACTIVE     nothing open

Quarantine reasons (a set; a product stays quarantined while any is open):
    PTS_BELOW_QUARANTINE   SC-7: PTS < tau_q
    COLD_CHAIN_CRITICAL    SC-5: Isolation Forest score > theta_c
    MANUAL_HOLD            an authorised party placed a hold

Alert conditions (recomputed on every event; never clear a quarantine):
    PTS_BELOW_ALERT        tau_q <= PTS < tau_a
    COLD_CHAIN_WARNING     theta_w < IF score <= theta_c
    TELEMETRY_GAP          no cold-chain reading for longer than max_gap

Rules:
    1. Quarantine reasons are sticky. A later good PTS or cold-chain reading
       is recorded but never closes an open reason.
    2. Only an authorised role (default: QA / regulatory) may release, and
       must give a reason code. Release is refused while the latest PTS is
       still below tau_q or the latest cold-chain score is still critical.
    3. Disposal is allowed only from QUARANTINE and is irreversible.
    4. Missing telemetry raises an alert; it never raises trust.
    5. Every transition is appended to ``history`` with its cause.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ACTIVE, ALERT, QUARANTINE, DISPOSED = "ACTIVE", "ALERT", "QUARANTINE", "DISPOSED"
PTS_Q, COLD_Q, MANUAL_Q = "PTS_BELOW_QUARANTINE", "COLD_CHAIN_CRITICAL", "MANUAL_HOLD"
PTS_A, COLD_A, GAP_A = "PTS_BELOW_ALERT", "COLD_CHAIN_WARNING", "TELEMETRY_GAP"
CNN_Q, CNN_A = "PACKAGING_SUSPECT", "PACKAGING_LOW_CONFIDENCE"

DEFAULT_RELEASE_ROLES = ("QA", "REGULATOR")


class TransitionError(Exception):
    """Raised when an event is not permitted from the current state."""


class ProductStateMachine:
    """Explicit enforcement state for one product."""

    def __init__(self, product_id: str, tau_q: float = 0.50, tau_a: float = 0.75,
                 theta_w: float = 0.70, theta_c: float = 0.90,
                 max_gap_minutes: float = 60.0,
                 release_roles: tuple = DEFAULT_RELEASE_ROLES,
                 cnn_q: float = 0.50, cnn_a: float = 0.75,
                 monitoring_start: Optional[float] = None) -> None:
        self.product_id = product_id
        # Round-2 review: a telemetry gap must be detectable even when the
        # FIRST reading never arrives, so monitoring has an explicit start.
        self.monitoring_start = monitoring_start
        self.cnn_q, self.cnn_a = cnn_q, cnn_a
        self.latest_cnn: Optional[float] = None
        self.tau_q, self.tau_a = tau_q, tau_a
        self.theta_w, self.theta_c = theta_w, theta_c
        self.max_gap_minutes = max_gap_minutes
        self.release_roles = tuple(release_roles)
        self.quarantine_reasons: set = set()
        self.alert_conditions: set = set()
        self.disposed = False
        self.latest_pts: Optional[float] = None
        self.latest_if: Optional[float] = None
        self.last_reading_t: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        if self.disposed:
            return DISPOSED
        if self.quarantine_reasons:
            return QUARANTINE
        if self.alert_conditions:
            return ALERT
        return ACTIVE

    def _log(self, t: float, event: str, before: str, detail: Dict[str, Any]) -> None:
        self.history.append({"t": t, "event": event, "from": before, "to": self.state,
                             "quarantine_reasons": sorted(self.quarantine_reasons),
                             "alert_conditions": sorted(self.alert_conditions), **detail})

    def _require_not_disposed(self, event: str) -> None:
        if self.disposed:
            raise TransitionError(f"{self.product_id}: {event} after disposal")

    # ----------------------------------------------------------------- events
    def pts_update(self, t: float, pts: float) -> str:
        """SC-7: record a new PTS. Can open PTS_Q; never closes any quarantine."""
        self._require_not_disposed("pts_update")
        before = self.state
        self.latest_pts = pts
        if pts < self.tau_q:
            self.quarantine_reasons.add(PTS_Q)
        self.alert_conditions.discard(PTS_A)
        if self.tau_q <= pts < self.tau_a:
            self.alert_conditions.add(PTS_A)
        self._log(t, "pts_update", before, {"pts": pts})
        return self.state

    def cold_chain_reading(self, t: float, if_score: float) -> str:
        """SC-5: record a cold-chain anomaly score. Can open COLD_Q; never closes it."""
        self._require_not_disposed("cold_chain_reading")
        before = self.state
        self.latest_if = if_score
        self.last_reading_t = t
        self.alert_conditions.discard(GAP_A)
        if if_score > self.theta_c:
            self.quarantine_reasons.add(COLD_Q)
        self.alert_conditions.discard(COLD_A)
        if self.theta_w < if_score <= self.theta_c:
            self.alert_conditions.add(COLD_A)
        self._log(t, "cold_chain_reading", before, {"if_score": if_score})
        return self.state

    def cnn_reading(self, t: float, p_authentic: float) -> str:
        """SC-4 (added 2026-10, OPTION11): packaging classifier output.

        Opens CNN_Q when P(authentic) < cnn_q (default 0.50, the PTS
        quarantine threshold reused); alert when < cnn_a (default 0.75).
        Like the other reasons, a later authentic-looking scan never closes it.
        """
        self._require_not_disposed("cnn_reading")
        before = self.state
        self.latest_cnn = p_authentic
        if p_authentic < self.cnn_q:
            self.quarantine_reasons.add(CNN_Q)
        self.alert_conditions.discard(CNN_A)
        if self.cnn_q <= p_authentic < self.cnn_a:
            self.alert_conditions.add(CNN_A)
        self._log(t, "cnn_reading", before, {"p_authentic": p_authentic})
        return self.state

    def start_monitoring(self, t: float) -> None:
        """Set the time from which a missing first reading counts as a gap."""
        self.monitoring_start = t

    def check_telemetry(self, t: float) -> str:
        """Raise TELEMETRY_GAP if no reading within max_gap_minutes.

        The reference is the last reading, or -- if none has arrived yet --
        the monitoring start time, so an absent first reading is detected.
        With neither set, no gap can be judged and nothing changes.
        """
        self._require_not_disposed("check_telemetry")
        before = self.state
        ref = self.last_reading_t if self.last_reading_t is not None else self.monitoring_start
        if ref is not None and t - ref > self.max_gap_minutes:
            self.alert_conditions.add(GAP_A)
        self._log(t, "check_telemetry", before, {})
        return self.state

    # ------------------------------------------------------------ persistence
    _SCALARS = ("product_id", "tau_q", "tau_a", "theta_w", "theta_c", "max_gap_minutes",
                "cnn_q", "cnn_a", "monitoring_start", "disposed", "latest_pts", "latest_if",
                "latest_cnn", "last_reading_t")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot for world state (sets become sorted lists)."""
        d = {k: getattr(self, k) for k in self._SCALARS}
        d.update(release_roles=list(self.release_roles),
                 quarantine_reasons=sorted(self.quarantine_reasons),
                 alert_conditions=sorted(self.alert_conditions),
                 history=[dict(h) for h in self.history])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductStateMachine":
        """Restore a machine exactly as saved by :meth:`to_dict`."""
        m = cls(d["product_id"], tau_q=d["tau_q"], tau_a=d["tau_a"], theta_w=d["theta_w"],
                theta_c=d["theta_c"], max_gap_minutes=d["max_gap_minutes"],
                release_roles=tuple(d["release_roles"]), cnn_q=d["cnn_q"], cnn_a=d["cnn_a"],
                monitoring_start=d["monitoring_start"])
        for k in ("disposed", "latest_pts", "latest_if", "latest_cnn", "last_reading_t"):
            setattr(m, k, d[k])
        m.quarantine_reasons = set(d["quarantine_reasons"])
        m.alert_conditions = set(d["alert_conditions"])
        m.history = [dict(h) for h in d["history"]]
        return m

    def hold(self, t: float, role: str, reason_code: str) -> str:
        """An authorised party places a manual quarantine hold."""
        self._require_not_disposed("hold")
        if role not in self.release_roles:
            raise TransitionError(f"{role} may not place a hold")
        before = self.state
        self.quarantine_reasons.add(MANUAL_Q)
        self._log(t, "hold", before, {"role": role, "reason_code": reason_code})
        return self.state

    def release(self, t: float, role: str, reason_code: str) -> str:
        """Authorised release of all open quarantine reasons (rule 2)."""
        self._require_not_disposed("release")
        if self.state != QUARANTINE:
            raise TransitionError(f"{self.product_id}: release from {self.state}")
        if role not in self.release_roles:
            raise TransitionError(f"{role} is not authorised to release")
        if not reason_code:
            raise TransitionError("release requires a reason code")
        if self.latest_pts is not None and self.latest_pts < self.tau_q:
            raise TransitionError("release refused: latest PTS still below quarantine threshold")
        if self.latest_if is not None and self.latest_if > self.theta_c:
            raise TransitionError("release refused: latest cold-chain score still critical")
        if self.latest_cnn is not None and self.latest_cnn < self.cnn_q:
            raise TransitionError("release refused: latest packaging scan still suspect")
        before = self.state
        released = sorted(self.quarantine_reasons)
        self.quarantine_reasons.clear()
        self._log(t, "release", before, {"role": role, "reason_code": reason_code,
                                         "released_reasons": released})
        return self.state

    def dispose(self, t: float, role: str, reason_code: str) -> str:
        """Irreversible disposal, only from QUARANTINE (rule 3)."""
        self._require_not_disposed("dispose")
        if self.state != QUARANTINE:
            raise TransitionError(f"{self.product_id}: dispose from {self.state}")
        if role not in self.release_roles:
            raise TransitionError(f"{role} is not authorised to dispose")
        before = self.state
        self.disposed = True
        self._log(t, "dispose", before, {"role": role, "reason_code": reason_code})
        return self.state


# ---------------------------------------------------------------- shared store
# All contracts that enforce on a product (SC-4 packaging, SC-5 cold chain,
# SC-7 trust score) load and save ONE machine per product under this key in a
# shared world state, so their decisions cannot diverge.
def machine_key(product_id: str) -> str:
    return f"psm_{product_id}"


def load_machine(contract: Any, product_id: str, **defaults: Any) -> ProductStateMachine:
    """Load the product's machine from the contract's world state, or create one."""
    saved = contract.get_state(machine_key(product_id))
    return ProductStateMachine.from_dict(saved) if saved else ProductStateMachine(product_id, **defaults)


def save_machine(contract: Any, machine: ProductStateMachine) -> None:
    contract.put_state(machine_key(machine.product_id), machine.to_dict())


STATUS_LABEL = {ACTIVE: "OK", ALERT: "ALERT", QUARANTINE: "QUARANTINE", DISPOSED: "DISPOSED"}


def sync_product_record(contract: Any, machine: ProductStateMachine) -> Optional[Dict[str, Any]]:
    """Mirror the machine's status onto the product record, if one is registered."""
    key = f"product_{machine.product_id}"
    record = contract.get_state(key)
    if record is None:
        return None
    record["status"] = STATUS_LABEL[machine.state]
    record["quarantine_reasons"] = sorted(machine.quarantine_reasons)
    record["alert_conditions"] = sorted(machine.alert_conditions)
    contract.put_state(key, record)
    return record
