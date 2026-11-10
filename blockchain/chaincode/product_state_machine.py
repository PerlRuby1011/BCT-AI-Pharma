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

Alert conditions (updated by relevant evidence and explicit freshness checks; never clear a quarantine):
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

Time convention and late events (added 2026-10-06, 7th internal review):
    * Every evidence event and every release carries an explicit EVENT time
      ``t`` -- when the observation was made, in minutes since the simulation
      epoch -- supplied by the caller. Contracts no longer substitute the
      history length. ``history`` also records ``seq``, the arrival order,
      so event time and arrival order can be told apart.
    * A reading is LATE if its event time is earlier than the stored latest
      reading of the same type. A late reading never replaces newer stored
      evidence, never changes value-based alert conditions and never counts
      toward release; receipt time can still update the telemetry-gap alert. If it is adverse it still opens (or extends) the matching
      hold: adverse evidence is never discarded.
    * Readings of one type with equal event times are concurrent; the more
      adverse value is kept.
    * Release of an evidence-based reason requires a stored reading of that
      type that is acceptable and strictly newer than the LATEST adverse
      reading of that type (not merely newer than the first one), and the
      release time may not precede that evidence.
    * Shipment records carry no independent hold. Their status is a view of
      this machine (see :func:`shipment_view`), so a product release cannot
      clear an independent shipment hold, and the two cannot contradict.

    * Future-dated evidence: if the caller supplies the receipt time
      ``received_t`` (minutes), evidence whose event time exceeds it by more
      than ``max_future_skew_minutes`` (default 5) is rejected with
      TransitionError and not recorded. Without ``received_t`` no check is
      possible; a future-dated reading is then stored as latest, and
      ``release`` still cannot use it before its own event time.
    * Freshness uses the greatest known receipt/check time, persisted as
      telemetry_clock_t. Receipt times on evidence calls advance that clock;
      explicit check_telemetry calls also advance it. Newer-but-stale cold
      readings do not clear a gap. Once established, event times on calls
      without receipt times cannot move the clock backwards. Before any
      receipt/check time is known, cold readings are assessed at their event
      time only; callers must supply receipt times or schedule explicit checks
      to detect delivery delay or silence. No background scheduler is provided.
    * Late adverse evidence arriving AFTER a release reopens the matching
      hold (conservative). It can be cleared by a new authorised release once
      the stored reading of that type is acceptable and newer than the late
      adverse reading -- including stored evidence that predates its arrival.
      ``seq`` in ``history`` records the arrival order.

Units: the machine's time unit is MINUTES. Callers holding hours must
convert at the interface with :func:`hours_to_machine_time` and convert
reported delays back with :func:`machine_time_to_hours`. The archived
round-3 lifecycle replay (commit 859e1e9) passed hours directly; it never
called ``check_telemetry`` or ``release`` (0 attempts in 20 runs), the only
places where the unit matters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ACTIVE, ALERT, QUARANTINE, DISPOSED = "ACTIVE", "ALERT", "QUARANTINE", "DISPOSED"
PTS_Q, COLD_Q, MANUAL_Q = "PTS_BELOW_QUARANTINE", "COLD_CHAIN_CRITICAL", "MANUAL_HOLD"
PTS_A, COLD_A, GAP_A = "PTS_BELOW_ALERT", "COLD_CHAIN_WARNING", "TELEMETRY_GAP"
CNN_Q, CNN_A = "PACKAGING_SUSPECT", "PACKAGING_LOW_CONFIDENCE"

DEFAULT_RELEASE_ROLES = ("QA", "REGULATOR")

MINUTES_PER_HOUR = 60.0


def hours_to_machine_time(hours: float) -> float:
    """Convert an event time in hours to the machine's unit (minutes)."""
    return float(hours) * MINUTES_PER_HOUR


def machine_time_to_hours(minutes: float) -> float:
    """Convert a machine time or delay (minutes) back to hours for reporting."""
    return float(minutes) / MINUTES_PER_HOUR



class TransitionError(Exception):
    """Raised when an event is not permitted from the current state."""


class ProductStateMachine:
    """Explicit enforcement state for one product."""

    def __init__(self, product_id: str, tau_q: float = 0.50, tau_a: float = 0.75,
                 theta_w: float = 0.70, theta_c: float = 0.90,
                 max_gap_minutes: float = 60.0,
                 release_roles: tuple = DEFAULT_RELEASE_ROLES,
                 cnn_q: float = 0.50, cnn_a: float = 0.75,
                 monitoring_start: Optional[float] = None,
                 max_future_skew_minutes: float = 5.0) -> None:
        self.product_id = product_id
        self.max_future_skew_minutes = max_future_skew_minutes
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
        self.telemetry_clock_t: Optional[float] = None
        # Round-3 release policy: when each hold reason opened, and when each
        # evidence type was last observed.
        self.reason_opened: Dict[str, float] = {}
        self.latest_pts_t: Optional[float] = None
        self.latest_if_t: Optional[float] = None
        self.latest_cnn_t: Optional[float] = None
        # 7th review: latest adverse event time per reason, shipments whose
        # readings opened/extended the current cold-chain hold, and the
        # shipment that supplied the latest stored cold-chain reading.
        self.reason_last_adverse: Dict[str, float] = {}
        self.cold_sources: List[str] = []
        self.latest_if_source: Optional[str] = None
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

    def _check_not_future(self, t: float, received_t: Optional[float], event: str) -> None:
        if received_t is not None and t > received_t + self.max_future_skew_minutes:
            raise TransitionError(
                f"{self.product_id}: {event} rejected: event time {t} is more than "
                f"{self.max_future_skew_minutes} min after receipt time {received_t}")

    @staticmethod
    def _late(t: float, stored_t: Optional[float]) -> bool:
        return stored_t is not None and t < stored_t

    def _log(self, t: float, event: str, before: str, detail: Dict[str, Any]) -> None:
        self.history.append({"t": t, "seq": len(self.history), "event": event, "from": before, "to": self.state,
                             "quarantine_reasons": sorted(self.quarantine_reasons),
                             "alert_conditions": sorted(self.alert_conditions), **detail})

    def _refresh_gap(self, t: float, received_t: Optional[float] = None,
                     cold_observation: bool = False, explicit: bool = False) -> None:
        """Evaluate freshness without replacing the latest evidence or closing holds.

        Explicit receipt/check times establish a monotone known clock. Without
        one, the archived synchronous replay retains event-time-only semantics.
        """
        now = received_t if received_t is not None else t
        if received_t is not None or explicit or self.telemetry_clock_t is not None:
            self.telemetry_clock_t = max(now, self.telemetry_clock_t) if self.telemetry_clock_t is not None else now
            now = self.telemetry_clock_t
        elif not cold_observation:
            return
        ref = self.last_reading_t if self.last_reading_t is not None else self.monitoring_start
        if ref is None:
            return
        if now - ref > self.max_gap_minutes:
            self.alert_conditions.add(GAP_A)
        else:
            self.alert_conditions.discard(GAP_A)

    def _require_not_disposed(self, event: str) -> None:
        if self.disposed:
            raise TransitionError(f"{self.product_id}: {event} after disposal")

    # ----------------------------------------------------------------- events
    def pts_update(self, t: float, pts: float, received_t: Optional[float] = None) -> str:
        """SC-7: record a new PTS. Can open PTS_Q; never closes any quarantine."""
        self._require_not_disposed("pts_update")
        self._check_not_future(t, received_t, "pts_update")
        before = self.state
        late = self._late(t, self.latest_pts_t)
        if not late:
            if t == self.latest_pts_t:
                pts_kept = min(pts, self.latest_pts)          # concurrent: keep more adverse
            else:
                pts_kept = pts
            self.latest_pts, self.latest_pts_t = pts_kept, t
        if pts < self.tau_q:
            self._open(PTS_Q, t)
        if not late:
            self.alert_conditions.discard(PTS_A)
            if self.tau_q <= self.latest_pts < self.tau_a:
                self.alert_conditions.add(PTS_A)
        self._refresh_gap(t, received_t)
        self._log(t, "pts_update", before, {"pts": pts, "late": late, "received_t": received_t})
        return self.state

    def cold_chain_reading(self, t: float, if_score: float, source: Optional[str] = None,
                           received_t: Optional[float] = None) -> str:
        """SC-5: record a cold-chain anomaly score. Can open COLD_Q; never closes it."""
        self._require_not_disposed("cold_chain_reading")
        self._check_not_future(t, received_t, "cold_chain_reading")
        before = self.state
        late = self._late(t, self.latest_if_t)
        if not late:
            if t == self.latest_if_t and if_score < self.latest_if:
                pass                                        # concurrent: keep more adverse
            else:
                self.latest_if, self.latest_if_source = if_score, source
            self.latest_if_t = t
            self.last_reading_t = t
        if if_score > self.theta_c:
            self._open(COLD_Q, t)
            if source is not None and source not in self.cold_sources:
                self.cold_sources.append(source)
        if not late:
            self.alert_conditions.discard(COLD_A)
            if self.theta_w < self.latest_if <= self.theta_c:
                self.alert_conditions.add(COLD_A)
        self._refresh_gap(t, received_t, cold_observation=True)
        self._log(t, "cold_chain_reading", before,
                  {"if_score": if_score, "late": late, "source": source, "received_t": received_t})
        return self.state

    def cnn_reading(self, t: float, p_authentic: float, received_t: Optional[float] = None) -> str:
        """SC-4 (added 2026-10, OPTION11): packaging classifier output.

        Opens CNN_Q when P(authentic) < cnn_q (default 0.50, the PTS
        quarantine threshold reused); alert when < cnn_a (default 0.75).
        Like the other reasons, a later authentic-looking scan never closes it.
        """
        self._require_not_disposed("cnn_reading")
        self._check_not_future(t, received_t, "cnn_reading")
        before = self.state
        late = self._late(t, self.latest_cnn_t)
        if not late:
            if t == self.latest_cnn_t:
                p_kept = min(p_authentic, self.latest_cnn)  # concurrent: keep more adverse
            else:
                p_kept = p_authentic
            self.latest_cnn, self.latest_cnn_t = p_kept, t
        if p_authentic < self.cnn_q:
            self._open(CNN_Q, t)
        if not late:
            self.alert_conditions.discard(CNN_A)
            if self.cnn_q <= self.latest_cnn < self.cnn_a:
                self.alert_conditions.add(CNN_A)
        self._refresh_gap(t, received_t)
        self._log(t, "cnn_reading", before, {"p_authentic": p_authentic, "late": late, "received_t": received_t})
        return self.state

    def _open(self, reason: str, t: float) -> None:
        if reason not in self.quarantine_reasons:
            self.reason_opened[reason] = t
        prev = self.reason_last_adverse.get(reason)
        self.reason_last_adverse[reason] = t if prev is None else max(prev, t)
        self.quarantine_reasons.add(reason)

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
        if self.last_reading_t is not None and t < self.last_reading_t:
            raise TransitionError("check_telemetry: check time precedes the last reading")
        self._refresh_gap(t, explicit=True)
        self._log(t, "check_telemetry", before, {})
        return self.state

    # ------------------------------------------------------------ persistence
    _SCALARS = ("product_id", "tau_q", "tau_a", "theta_w", "theta_c", "max_gap_minutes",
                "cnn_q", "cnn_a", "monitoring_start", "disposed", "latest_pts", "latest_if",
                "latest_cnn", "last_reading_t", "latest_pts_t", "latest_if_t", "latest_cnn_t",
                "latest_if_source", "max_future_skew_minutes", "telemetry_clock_t")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot for world state (sets become sorted lists)."""
        d = {k: getattr(self, k) for k in self._SCALARS}
        d.update(release_roles=list(self.release_roles),
                 quarantine_reasons=sorted(self.quarantine_reasons),
                 alert_conditions=sorted(self.alert_conditions),
                 reason_opened=dict(self.reason_opened),
                 reason_last_adverse=dict(self.reason_last_adverse),
                 cold_sources=list(self.cold_sources),
                 history=[dict(h) for h in self.history])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductStateMachine":
        """Restore a machine exactly as saved by :meth:`to_dict`."""
        m = cls(d["product_id"], tau_q=d["tau_q"], tau_a=d["tau_a"], theta_w=d["theta_w"],
                theta_c=d["theta_c"], max_gap_minutes=d["max_gap_minutes"],
                release_roles=tuple(d["release_roles"]), cnn_q=d["cnn_q"], cnn_a=d["cnn_a"],
                monitoring_start=d["monitoring_start"],
                max_future_skew_minutes=d.get("max_future_skew_minutes", 5.0))
        for k in ("disposed", "latest_pts", "latest_if", "latest_cnn", "last_reading_t",
                  "latest_pts_t", "latest_if_t", "latest_cnn_t", "latest_if_source", "telemetry_clock_t"):
            setattr(m, k, d.get(k))
        m.reason_opened = dict(d.get("reason_opened", {}))
        m.reason_last_adverse = dict(d.get("reason_last_adverse", m.reason_opened))
        m.cold_sources = list(d.get("cold_sources", []))
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
        self._open(MANUAL_Q, t)
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
        # Round-3: every open evidence-based reason needs a reading of that type
        # observed AFTER the hold opened, within its acceptable range; missing
        # evidence for an open reason blocks release. Manual holds need role +
        # reason code only.
        need = {PTS_Q: ("PTS", self.latest_pts, self.latest_pts_t, lambda v: v >= self.tau_q),
                COLD_Q: ("cold-chain", self.latest_if, self.latest_if_t, lambda v: v <= self.theta_c),
                CNN_Q: ("packaging", self.latest_cnn, self.latest_cnn_t, lambda v: v >= self.cnn_q)}
        for reason in self.quarantine_reasons:
            if reason not in need:
                continue
            name, val, vt, ok = need[reason]
            last_bad = self.reason_last_adverse.get(reason, self.reason_opened.get(reason))
            if val is None or vt is None or (last_bad is not None and not vt > last_bad):
                raise TransitionError(
                    f"release refused: no {name} evidence newer than the latest adverse reading")
            if t < vt:
                raise TransitionError("release refused: release time precedes the evidence it uses")
            if not ok(val):
                raise TransitionError(f"release refused: latest {name} evidence still unacceptable")
        if self.latest_pts is not None and self.latest_pts < self.tau_q:
            raise TransitionError("release refused: latest PTS still below quarantine threshold")
        if self.latest_if is not None and self.latest_if > self.theta_c:
            raise TransitionError("release refused: latest cold-chain score still critical")
        if self.latest_cnn is not None and self.latest_cnn < self.cnn_q:
            raise TransitionError("release refused: latest packaging scan still suspect")
        before = self.state
        released = sorted(self.quarantine_reasons)
        self.quarantine_reasons.clear(); self.reason_opened.clear()
        self.reason_last_adverse.clear(); self.cold_sources = []
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


def shipment_view(machine: ProductStateMachine, shipment_id: str) -> str:
    """Cold-chain status of one shipment, derived from the product machine.

    ``quarantined``: the product's cold-chain hold is open and this shipment
    supplied adverse evidence for it; ``warning``: a cold-chain warning is
    current and this shipment supplied the latest stored reading; otherwise
    ``in_transit``. Holds for other reasons (packaging, PTS, manual) show in
    the shipment record's ``product_status``, not as a shipment quarantine.
    """
    if COLD_Q in machine.quarantine_reasons and shipment_id in machine.cold_sources:
        return "quarantined"
    if COLD_A in machine.alert_conditions and machine.latest_if_source == shipment_id:
        return "warning"
    return "in_transit"


def sync_shipment_records(contract: Any, machine: ProductStateMachine) -> int:
    """Re-derive the status of every shipment record carrying this product."""
    n = 0
    for key, rec in contract.query_by_prefix("shipment_").items():
        if rec.get("product_id") != machine.product_id:
            continue
        rec["status"] = shipment_view(machine, rec["shipment_id"])
        rec["product_status"] = machine.state
        contract.put_state(key, rec)
        n += 1
    return n
