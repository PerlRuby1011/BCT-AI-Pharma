"""Round-2 review: SC-4 / SC-5 / SC-7 enforce through the PUBLIC contract entry
points on ONE persisted product state machine (shared world state)."""
import pytest

from blockchain.chaincode.cold_chain_monitor import ColdChainMonitorContract
from blockchain.chaincode.counterfeit_detection import CounterfeitDetectionContract
from blockchain.chaincode.product_state_machine import (
    ProductStateMachine, TransitionError, load_machine, machine_key,
)


def channel():
    ws = {}
    cf = CounterfeitDetectionContract(world_state=ws)
    cc = ColdChainMonitorContract(world_state=ws)
    cf.invoke("register_product", {"product_id": "P", "drug_class": "A", "manufacturer": "M"})
    cc.invoke("register_shipment", {"shipment_id": "S", "product_id": "P",
                                    "target_temp_c": 5.0, "t": 0.0})
    return ws, cf, cc


def status(cf):
    return cf.invoke("get_product_status", {"product_id": "P"})["status"]


def test_sc4_packaging_scan_quarantines_product():          # reviewer counterexample 1
    _, cf, _ = channel()
    out = cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.30,
                                            "tamper_class": "Broken Seal", "t": 1.0})
    assert out["status"] == "QUARANTINE" and "PACKAGING_SUSPECT" in out["quarantine_reasons"]


def test_sc5_critical_reading_quarantines_product_not_just_shipment():  # counterexample 2
    _, cf, cc = channel()
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0,
                                        "anomaly_score": 0.97, "t": 10.0})
    assert cc.invoke("get_shipment_status", {"shipment_id": "S"})["status"] == "quarantined"
    assert status(cf) == "QUARANTINE"


def test_overlapping_holds_from_all_three_contracts_and_release():
    _, cf, cc = channel()
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.40, "t": 1.0})
    cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.20,
                                      "tamper_class": "Hologram Mismatch", "t": 2.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 16.0,
                                        "anomaly_score": 0.95, "t": 5.0})
    rec = cf.invoke("get_product_status", {"product_id": "P"})
    assert set(rec["quarantine_reasons"]) == {"PTS_BELOW_QUARANTINE", "PACKAGING_SUSPECT",
                                              "COLD_CHAIN_CRITICAL"}
    # evidence still bad -> release refused
    with pytest.raises(TransitionError):
        cf.invoke("release_quarantine", {"product_id": "P", "role": "QA", "reason_code": "x",
                                         "t": 6.0})
    # recovery on every channel does NOT clear it
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.95, "t": 18.0})
    cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.99,
                                      "tamper_class": "Authentic", "t": 19.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                        "anomaly_score": 0.10, "t": 20.0})
    assert status(cf) == "QUARANTINE"
    with pytest.raises(TransitionError):
        cf.invoke("release_quarantine", {"product_id": "P", "role": "DISTRIBUTOR", "reason_code": "x",
                                         "t": 21.0})
    assert cf.invoke("release_quarantine", {"product_id": "P", "role": "QA",
                                            "reason_code": "lab retest", "t": 21.0})["status"] == "OK"


def test_missing_first_reading_raises_gap_alert():          # reviewer missing-data case
    _, cf, cc = channel()                                   # monitoring starts at t=0
    out = cc.invoke("check_telemetry", {"shipment_id": "S", "t": 120.0})
    assert "TELEMETRY_GAP" in out["alert_conditions"]
    assert status(cf) == "ALERT"


def test_standalone_machine_missing_first_reading():
    m = ProductStateMachine("X", monitoring_start=0.0)
    assert m.check_telemetry(120.0) == "ALERT"
    assert ProductStateMachine("Y").check_telemetry(120.0) == "ACTIVE"   # no start: cannot judge


def test_state_restored_from_world_state_by_a_fresh_contract():
    ws, cf, _ = channel()
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.30, "t": 1.0})
    cf2 = CounterfeitDetectionContract(world_state=ws)       # e.g. peer restart
    cf2.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.95, "t": 2.0})
    assert cf2.invoke("get_product_status", {"product_id": "P"})["status"] == "QUARANTINE"
    m = ProductStateMachine.from_dict(ws[machine_key("P")])
    assert m.to_dict() == ws[machine_key("P")]
    assert [h["event"] for h in m.history][-2:] == ["pts_update", "pts_update"]


def test_private_world_states_do_not_share_enforcement():
    cf = CounterfeitDetectionContract()
    cc = ColdChainMonitorContract()
    cf.invoke("register_product", {"product_id": "P", "drug_class": "A", "manufacturer": "M"})
    cc.invoke("register_shipment", {"shipment_id": "S", "product_id": "P", "target_temp_c": 5.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0, "anomaly_score": 0.97,
                                        "t": 1.0})
    # documented behaviour: integration requires a shared world state
    assert cf.invoke("get_product_status", {"product_id": "P"})["status"] == "OK"
    assert load_machine(cc, "P").state == "QUARANTINE"


# ---------------------------------------------------------------------------
# 7th internal review (2026-10-06): reproduced counterexamples, kept as
# regression tests. Time convention: explicit event time t (minutes) on every
# public call; late readings never replace newer evidence or support release.

def test_regression_release_resyncs_shipment_status():
    """Counterexample 1: product released to OK while shipment stayed 'quarantined'."""
    _, cf, cc = channel()
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0,
                                        "anomaly_score": 0.95, "t": 10.0})
    assert cc.invoke("get_shipment_status", {"shipment_id": "S"})["status"] == "quarantined"
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                        "anomaly_score": 0.20, "t": 20.0})
    rec = cf.invoke("release_quarantine", {"product_id": "P", "role": "QA",
                                           "reason_code": "lab retest", "t": 21.0})
    assert rec["status"] == "OK"
    ship = cc.invoke("get_shipment_status", {"shipment_id": "S"})
    assert ship["status"] == "in_transit" and ship["product_status"] == "ACTIVE"
    assert [a["level"] for a in ship["alerts"]] == ["critical"]      # log preserved


def test_regression_stale_reading_cannot_support_release():
    """Counterexample 2: critical at 10 and 30, older acceptable reading at 20, release at 31."""
    m = ProductStateMachine("X")
    m.cold_chain_reading(10, 0.95)
    m.cold_chain_reading(30, 0.95)
    m.cold_chain_reading(20, 0.20)                       # arrives late
    assert m.latest_if == 0.95 and m.latest_if_t == 30   # not replaced
    assert m.history[-1]["late"] is True
    with pytest.raises(TransitionError):
        m.release(31, "QA", "ok")
    m.cold_chain_reading(40, 0.20)                       # genuinely newer evidence
    assert m.release(41, "QA", "ok") == "ACTIVE"


def test_regression_same_counterexample_through_public_contract():
    _, cf, cc = channel()
    for t, s in ((10.0, 0.95), (30.0, 0.95), (20.0, 0.20)):
        cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                            "anomaly_score": s, "t": t})
    with pytest.raises(TransitionError):
        cf.invoke("release_quarantine", {"product_id": "P", "role": "QA",
                                         "reason_code": "x", "t": 31.0})
    assert status(cf) == "QUARANTINE"


def test_regression_transition_times_are_event_times_across_contracts():
    """Reviewer saw transition times 10, 20, 2 (history length mixed with event time)."""
    ws, cf, cc = channel()
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0,
                                        "anomaly_score": 0.95, "t": 10.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                        "anomaly_score": 0.20, "t": 20.0})
    cf.invoke("release_quarantine", {"product_id": "P", "role": "QA", "reason_code": "x",
                                     "t": 21.0})
    m = load_machine(cf, "P")
    assert [h["t"] for h in m.history] == [10.0, 20.0, 21.0]
    assert [h["seq"] for h in m.history] == [0, 1, 2]


def test_late_adverse_reading_still_opens_a_hold():
    m = ProductStateMachine("X")
    m.cnn_reading(30, 0.95)
    assert m.cnn_reading(10, 0.20) == "QUARANTINE"       # late but adverse: not dropped
    assert m.latest_cnn == 0.95                         # newer evidence kept
    assert m.release(31, "QA", "ok") == "ACTIVE"        # newer acceptable evidence exists


def test_concurrent_readings_keep_the_more_adverse_value():
    m = ProductStateMachine("X")
    m.pts_update(5, 0.60); m.pts_update(5, 0.90)
    assert m.latest_pts == 0.60 and m.state == "ALERT"


def test_release_time_may_not_precede_its_evidence():
    m = ProductStateMachine("X")
    m.pts_update(10, 0.30); m.pts_update(20, 0.90)
    with pytest.raises(TransitionError):
        m.release(15, "QA", "ok")


def test_other_hold_reason_is_not_shown_as_shipment_quarantine_nor_cleared():
    """Independent reasons vs contradiction: a packaging hold is visible on the
    shipment as product_status, not as a cold-chain shipment quarantine, and a
    cold-chain recovery does not clear it."""
    _, cf, cc = channel()
    cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.20,
                                      "tamper_class": "Broken Seal", "t": 5.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                        "anomaly_score": 0.10, "t": 6.0})
    ship = cc.invoke("get_shipment_status", {"shipment_id": "S"})
    assert ship["status"] == "in_transit" and ship["product_status"] == "QUARANTINE"
    assert status(cf) == "QUARANTINE"


def test_hold_from_one_shipment_is_not_attributed_to_another():
    ws, cf, cc = channel()
    cc.invoke("register_shipment", {"shipment_id": "S2", "product_id": "P",
                                    "target_temp_c": 5.0, "t": 50.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0,
                                        "anomaly_score": 0.95, "t": 10.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S2", "temperature_c": 5.0,
                                        "anomaly_score": 0.10, "t": 60.0})
    assert cc.invoke("get_shipment_status", {"shipment_id": "S"})["status"] == "quarantined"
    assert cc.invoke("get_shipment_status", {"shipment_id": "S2"})["status"] == "in_transit"
    assert status(cf) == "QUARANTINE"


def test_contract_calls_require_explicit_event_time():
    _, cf, cc = channel()
    with pytest.raises(TypeError):
        cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.9})
    with pytest.raises(TypeError):
        cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                            "anomaly_score": 0.1})
