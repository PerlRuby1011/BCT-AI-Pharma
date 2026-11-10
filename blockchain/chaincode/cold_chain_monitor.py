"""Cold chain monitoring smart contract.

Records Isolation Forest anomaly scores for temperature sensor readings and
applies the paper's stated thresholds ("score > 0.7 warning, > 0.9 critical
quarantine"): ``score > 0.7`` raises a warning to the logistics provider,
``score > 0.9`` triggers an automated critical quarantine.
"""
from __future__ import annotations

from typing import Any, Dict, List

from blockchain.chaincode.product_state_machine import (
    load_machine, save_machine, sync_product_record, sync_shipment_records,
)
from blockchain.chaincode.smart_contract_base import SmartContractBase

WARNING_THRESHOLD = 0.7
CRITICAL_THRESHOLD = 0.9


class ColdChainMonitorContract(SmartContractBase):
    """Chaincode contract for cold-chain anomaly alerting and quarantine."""

    def __init__(
        self,
        warning_threshold: float = WARNING_THRESHOLD,
        critical_threshold: float = CRITICAL_THRESHOLD,
        world_state: Dict[str, Any] | None = None,
        max_gap_minutes: float = 60.0,
    ) -> None:
        """Initialize the contract with configurable anomaly-score thresholds.

        Args:
            warning_threshold: Isolation Forest anomaly score above which a
                warning alert is issued.
            critical_threshold: Anomaly score above which the shipment is
                automatically quarantined.
            world_state: Optional world state shared with the counterfeit
                contract. Each reading updates the shipment record AND the
                shipped product's enforcement machine (mapping: one shipment
                carries one product_id, set at registration).
            max_gap_minutes: Telemetry gap that raises an alert.
        """
        super().__init__(contract_name="ColdChainMonitorContract", world_state=world_state)
        self.max_gap_minutes = max_gap_minutes
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def invoke(self, function: str, args: Dict[str, Any]) -> Any:
        """Dispatch to the named contract function.

        Args:
            function: One of ``register_shipment``, ``record_sensor_reading``,
                ``get_shipment_status``, ``get_alerts``.
            args: Function-specific keyword arguments.

        Returns:
            The result of the dispatched function.

        Raises:
            ValueError: If ``function`` is not a recognized contract method.
        """
        dispatch = {
            "register_shipment": self.register_shipment,
            "record_sensor_reading": self.record_sensor_reading,
            "get_shipment_status": self.get_shipment_status,
            "get_alerts": self.get_alerts,
            "check_telemetry": self.check_telemetry,
        }
        if function not in dispatch:
            raise ValueError(f"Unknown chaincode function: {function}")
        result = dispatch[function](**args)
        self._record_transaction(function, args, result)
        return result

    def _product_machine(self, product_id: str):
        return load_machine(self, product_id, theta_w=self.warning_threshold,
                            theta_c=self.critical_threshold, max_gap_minutes=self.max_gap_minutes)

    def check_telemetry(self, shipment_id: str, t: float) -> Dict[str, Any]:
        """Raise a telemetry-gap alert on the shipped product if readings are overdue.

        Counts from the last reading, or from monitoring start if no reading
        has ever arrived (round-2 review: absent first reading).
        """
        record = self.get_shipment_status(shipment_id)
        machine = self._product_machine(record["product_id"])
        machine.check_telemetry(t)
        save_machine(self, machine)
        sync_product_record(self, machine)
        sync_shipment_records(self, machine)
        return {"shipment_id": shipment_id, "product_status": machine.state,
                "alert_conditions": sorted(machine.alert_conditions)}

    def register_shipment(self, shipment_id: str, product_id: str, target_temp_c: float,
                          t: float = 0.0) -> Dict[str, Any]:
        """Register a new cold-chain shipment.

        Args:
            shipment_id: Unique shipment identifier.
            product_id: Product being shipped.
            target_temp_c: Target storage temperature in Celsius.

        Returns:
            The newly created shipment state record.
        """
        record = {
            "shipment_id": shipment_id,
            "product_id": product_id,
            "target_temp_c": target_temp_c,
            "status": "in_transit",
            "readings": [],
            "alerts": [],
        }
        self.put_state(f"shipment_{shipment_id}", record)
        machine = self._product_machine(product_id)
        if machine.monitoring_start is None:
            machine.start_monitoring(t)
        save_machine(self, machine)
        return record

    def record_sensor_reading(
        self, shipment_id: str, temperature_c: float, anomaly_score: float,
        t: float, received_t: float | None = None,
    ) -> Dict[str, Any]:
        """Record a sensor reading and its Isolation Forest anomaly score.

        Applies the two-tier alert policy: a warning is logged above
        ``warning_threshold`` and the shipment is quarantined above
        ``critical_threshold``.

        Args:
            shipment_id: Shipment the reading belongs to.
            temperature_c: Measured temperature in Celsius.
            anomaly_score: Isolation Forest anomaly score in [0, 1].
            t: Event time of the reading (minutes since the simulation
                epoch); required. Late readings are handled by the product
                machine's late-event policy.

        Returns:
            The updated shipment state record.

        Raises:
            KeyError: If the shipment has not been registered.
        """
        key = f"shipment_{shipment_id}"
        record = self.get_state(key)
        if record is None:
            raise KeyError(f"Shipment {shipment_id} is not registered")
        # Reject future-dated evidence BEFORE touching the shipment log.
        self._product_machine(record["product_id"])._check_not_future(
            t, received_t, "cold_chain_reading")

        record["readings"].append(
            {"t": t, "temperature_c": temperature_c, "anomaly_score": anomaly_score}
        )
        # The alert list is an append-only log; the shipment's STATUS is a
        # view of the product machine (7th review: no divergent shipment hold).
        if anomaly_score > self.critical_threshold:
            record["alerts"].append({"t": t, "level": "critical", "anomaly_score": anomaly_score,
                                     "action": "auto_quarantine"})
        elif anomaly_score > self.warning_threshold:
            record["alerts"].append({"t": t, "level": "warning", "anomaly_score": anomaly_score,
                                     "action": "notify_logistics"})
        self.put_state(key, record)
        machine = self._product_machine(record["product_id"])
        machine.cold_chain_reading(t, anomaly_score, source=shipment_id, received_t=received_t)
        save_machine(self, machine)
        sync_product_record(self, machine)
        sync_shipment_records(self, machine)
        return self.get_state(key)

    def get_shipment_status(self, shipment_id: str) -> Dict[str, Any]:
        """Query the current on-chain state of a shipment.

        Args:
            shipment_id: Shipment to look up.

        Returns:
            The shipment's state record.

        Raises:
            KeyError: If the shipment has not been registered.
        """
        record = self.get_state(f"shipment_{shipment_id}")
        if record is None:
            raise KeyError(f"Shipment {shipment_id} is not registered")
        return record

    def get_alerts(self, level: str | None = None) -> List[Dict[str, Any]]:
        """Return all alerts across all shipments, optionally filtered by level.

        Args:
            level: If provided, only return alerts of this level
                (``"warning"`` or ``"critical"``).

        Returns:
            List of alert dictionaries, each tagged with its ``shipment_id``.
        """
        alerts: List[Dict[str, Any]] = []
        for key, record in self.query_by_prefix("shipment_").items():
            for alert in record.get("alerts", []):
                if level is None or alert["level"] == level:
                    alerts.append({**alert, "shipment_id": record["shipment_id"]})
        return alerts
