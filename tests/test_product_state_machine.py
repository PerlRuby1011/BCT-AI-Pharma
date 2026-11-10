"""Transition tests for the unified enforcement state machine (review item 2)."""
import pytest

from blockchain.chaincode.product_state_machine import (
    ACTIVE, ALERT, COLD_Q, DISPOSED, GAP_A, PTS_Q, QUARANTINE,
    ProductStateMachine, TransitionError,
)


def sm():
    return ProductStateMachine("P1")


def test_score_recovery_does_not_clear_quarantine():
    m = sm()
    assert m.pts_update(0, 0.40) == QUARANTINE
    assert m.pts_update(1, 0.95) == QUARANTINE          # the reviewer's case
    assert PTS_Q in m.quarantine_reasons


def test_severe_excursion_quarantines_even_when_pts_stays_above_threshold():
    m = sm()
    m.pts_update(0, 0.518)                               # paper's 30 h / 17 C case
    assert m.state == ALERT
    assert m.cold_chain_reading(1, 0.97) == QUARANTINE   # SC-5 fires independently
    m.cold_chain_reading(2, 0.10)
    m.pts_update(3, 0.90)
    assert m.state == QUARANTINE and COLD_Q in m.quarantine_reasons


def test_overlapping_triggers_keep_both_reasons_and_release_clears_both():
    m = sm()
    m.pts_update(0, 0.30)
    m.cold_chain_reading(1, 0.95)
    assert m.quarantine_reasons == {PTS_Q, COLD_Q}
    m.pts_update(2, 0.90)
    m.cold_chain_reading(3, 0.20)
    assert m.release(4, "QA", "lab retest passed") == ACTIVE


def test_release_refused_while_evidence_still_bad():
    m = sm()
    m.pts_update(0, 0.30)
    with pytest.raises(TransitionError):
        m.release(1, "QA", "premature")
    m.pts_update(2, 0.80)
    m.cold_chain_reading(3, 0.95)
    with pytest.raises(TransitionError):
        m.release(4, "QA", "cold chain still critical")


def test_unauthorised_release_rejected():
    m = sm()
    m.pts_update(0, 0.30)
    m.pts_update(1, 0.90)
    for role in ("DISTRIBUTOR", "MANUFACTURER", "PHARMACY", ""):
        with pytest.raises(TransitionError):
            m.release(2, role, "x")
    with pytest.raises(TransitionError):
        m.release(2, "QA", "")                           # reason code required
    assert m.state == QUARANTINE


def test_release_returns_to_alert_if_alert_condition_open():
    m = sm()
    m.pts_update(0, 0.30)
    m.pts_update(1, 0.60)
    assert m.release(2, "REGULATOR", "investigation closed") == ALERT


def test_missing_telemetry_alerts_and_never_raises_trust():
    m = sm()
    m.pts_update(0, 0.90)
    m.cold_chain_reading(0, 0.10)
    assert m.check_telemetry(30) == ACTIVE
    assert m.check_telemetry(120) == ALERT and GAP_A in m.alert_conditions
    m.cold_chain_reading(121, 0.10)
    assert m.state == ACTIVE


def test_gap_does_not_mask_quarantine():
    m = sm()
    m.cold_chain_reading(0, 0.95)
    assert m.check_telemetry(500) == QUARANTINE


def test_disposal_is_terminal_and_only_from_quarantine():
    m = sm()
    with pytest.raises(TransitionError):
        m.dispose(0, "QA", "x")                          # not quarantined
    m.pts_update(1, 0.20)
    assert m.dispose(2, "QA", "destroyed") == DISPOSED
    for call in (lambda: m.pts_update(3, 0.9), lambda: m.cold_chain_reading(3, 0.1),
                 lambda: m.release(3, "QA", "x"), lambda: m.hold(3, "QA", "x")):
        with pytest.raises(TransitionError):
            call()


def test_manual_hold_survives_good_scores():
    m = sm()
    m.pts_update(0, 0.95)
    m.hold(1, "REGULATOR", "FDA request")
    m.pts_update(2, 0.99)
    assert m.state == QUARANTINE
    assert m.release(3, "REGULATOR", "request closed") == ACTIVE


def test_history_records_every_transition_with_cause():
    m = sm()
    m.pts_update(0, 0.40)
    m.pts_update(1, 0.90)
    m.release(2, "QA", "retest")
    events = [(h["event"], h["from"], h["to"]) for h in m.history]
    assert events == [("pts_update", ACTIVE, QUARANTINE),
                      ("pts_update", QUARANTINE, QUARANTINE),
                      ("release", QUARANTINE, ACTIVE)]
    assert m.history[-1]["released_reasons"] == [PTS_Q]


def test_contract_keeps_quarantine_until_authorised_release():
    from blockchain.chaincode.counterfeit_detection import CounterfeitDetectionContract
    c = CounterfeitDetectionContract()
    c.invoke("register_product", {"product_id": "X", "drug_class": "A", "manufacturer": "M"})
    assert c.invoke("update_trust_score", {"product_id": "X", "trust_score": 0.40, "t": 1.0})["status"] == "QUARANTINE"
    assert c.invoke("update_trust_score", {"product_id": "X", "trust_score": 0.95, "t": 2.0})["status"] == "QUARANTINE"
    with pytest.raises(TransitionError):
        c.invoke("release_quarantine", {"product_id": "X", "role": "DISTRIBUTOR", "reason_code": "x", "t": 3.0})
    out = c.invoke("release_quarantine", {"product_id": "X", "role": "QA", "reason_code": "retest", "t": 3.0})
    assert out["status"] == "OK"


def test_sc4_packaging_quarantine_independent_of_pts():
    from blockchain.chaincode.product_state_machine import CNN_Q
    m = sm()
    m.pts_update(0, 0.92)                                # full PTS looks healthy
    assert m.cnn_reading(1, 0.30) == QUARANTINE          # SC-4 fires anyway
    m.cnn_reading(2, 0.99)
    assert m.state == QUARANTINE and CNN_Q in m.quarantine_reasons
    assert m.release(3, "QA", "lab authenticated") == ACTIVE


def test_sc4_release_refused_while_scan_suspect():
    m = sm()
    m.cnn_reading(0, 0.30)
    with pytest.raises(TransitionError):
        m.release(1, "QA", "premature")
    assert m.cnn_reading(2, 0.60) == QUARANTINE          # alert band, quarantine still open


def test_threshold_boundaries():
    m = sm()
    assert m.pts_update(0, 0.50) == ALERT                # tau_q itself is not quarantine
    assert m.pts_update(1, 0.75) == ACTIVE               # tau_a itself is not alert
    assert m.cold_chain_reading(2, 0.70) == ACTIVE       # theta_w is strict
    assert m.cold_chain_reading(3, 0.90) == ALERT        # theta_c is strict


# ---- Round-3 release policy: evidence for each open reason must be re-observed
def test_release_needs_new_evidence_for_each_open_reason():
    m = sm()
    m.cnn_reading(0, 0.30)                       # packaging hold, PTS/IF never observed
    with pytest.raises(TransitionError):
        m.release(1, "QA", "no new scan yet")    # was allowed before the fix
    m.cnn_reading(2, 0.95)
    assert m.release(3, "QA", "rescan authentic") == ACTIVE


def test_absent_unrelated_evidence_does_not_block_release():
    m = sm()
    m.cnn_reading(0, 0.30)
    m.cnn_reading(1, 0.90)                       # no PTS, no IF ever: not open reasons
    assert m.release(2, "QA", "rescan") == ACTIVE


def test_reading_at_same_time_as_hold_does_not_count():
    m = sm()
    m.pts_update(5, 0.30)
    with pytest.raises(TransitionError):
        m.release(6, "QA", "x")                  # only the triggering reading exists
    m.pts_update(7, 0.90)
    assert m.release(8, "QA", "recovered") == ACTIVE


def test_overlapping_reasons_each_need_evidence():
    m = sm()
    m.pts_update(0, 0.30); m.cold_chain_reading(1, 0.95)
    m.pts_update(2, 0.90)                        # PTS recovered, cold chain not re-read
    with pytest.raises(TransitionError):
        m.release(3, "QA", "x")
    m.cold_chain_reading(4, 0.10)
    assert m.release(5, "QA", "both cleared") == ACTIVE
