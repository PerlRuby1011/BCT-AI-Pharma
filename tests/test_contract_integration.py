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
                                            "tamper_class": "Broken Seal"})
    assert out["status"] == "QUARANTINE" and "PACKAGING_SUSPECT" in out["quarantine_reasons"]


def test_sc5_critical_reading_quarantines_product_not_just_shipment():  # counterexample 2
    _, cf, cc = channel()
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0,
                                        "anomaly_score": 0.97, "t": 10.0})
    assert cc.invoke("get_shipment_status", {"shipment_id": "S"})["status"] == "quarantined"
    assert status(cf) == "QUARANTINE"


def test_overlapping_holds_from_all_three_contracts_and_release():
    _, cf, cc = channel()
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.40})
    cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.20,
                                      "tamper_class": "Hologram Mismatch"})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 16.0,
                                        "anomaly_score": 0.95, "t": 5.0})
    rec = cf.invoke("get_product_status", {"product_id": "P"})
    assert set(rec["quarantine_reasons"]) == {"PTS_BELOW_QUARANTINE", "PACKAGING_SUSPECT",
                                              "COLD_CHAIN_CRITICAL"}
    # evidence still bad -> release refused
    with pytest.raises(TransitionError):
        cf.invoke("release_quarantine", {"product_id": "P", "role": "QA", "reason_code": "x"})
    # recovery on every channel does NOT clear it
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.95})
    cf.invoke("record_verification", {"product_id": "P", "cnn_authenticity_score": 0.99,
                                      "tamper_class": "Authentic"})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 5.0,
                                        "anomaly_score": 0.10, "t": 20.0})
    assert status(cf) == "QUARANTINE"
    with pytest.raises(TransitionError):
        cf.invoke("release_quarantine", {"product_id": "P", "role": "DISTRIBUTOR", "reason_code": "x"})
    assert cf.invoke("release_quarantine", {"product_id": "P", "role": "QA",
                                            "reason_code": "lab retest"})["status"] == "OK"


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
    cf.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.30})
    cf2 = CounterfeitDetectionContract(world_state=ws)       # e.g. peer restart
    cf2.invoke("update_trust_score", {"product_id": "P", "trust_score": 0.95})
    assert cf2.invoke("get_product_status", {"product_id": "P"})["status"] == "QUARANTINE"
    m = ProductStateMachine.from_dict(ws[machine_key("P")])
    assert m.to_dict() == ws[machine_key("P")]
    assert [h["event"] for h in m.history][-2:] == ["pts_update", "pts_update"]


def test_private_world_states_do_not_share_enforcement():
    cf = CounterfeitDetectionContract()
    cc = ColdChainMonitorContract()
    cf.invoke("register_product", {"product_id": "P", "drug_class": "A", "manufacturer": "M"})
    cc.invoke("register_shipment", {"shipment_id": "S", "product_id": "P", "target_temp_c": 5.0})
    cc.invoke("record_sensor_reading", {"shipment_id": "S", "temperature_c": 17.0, "anomaly_score": 0.97})
    # documented behaviour: integration requires a shared world state
    assert cf.invoke("get_product_status", {"product_id": "P"})["status"] == "OK"
    assert load_machine(cc, "P").state == "QUARANTINE"
