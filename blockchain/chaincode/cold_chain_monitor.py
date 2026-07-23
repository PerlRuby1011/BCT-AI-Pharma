"""Cold chain monitoring smart contract.

Records Isolation Forest anomaly scores for temperature sensor readings and
applies the paper's stated thresholds ("score > 0.7 warning, > 0.9 critical
quarantine"): ``score > 0.7`` raises a warning to the logistics provider,
``score > 0.9`` triggers an automated critical quarantine.
"""
from __future__ import annotations

from typing import Any, Dict, List

from blockchain.chaincode.smart_contract_base import SmartContractBase

WARNING_THRESHOLD = 0.7
CRITICAL_THRESHOLD = 0.9


class ColdChainMonitorContract(SmartContractBase):
    """Chaincode contract for cold-chain anomaly alerting and quarantine."""

    def __init__(
        self,
        warning_threshold: float = WARNING_THRESHOLD,
        critical_threshold: float = CRITICAL_THRESHOLD,
    ) -> None:
        """Initialize the contract with configurable anomaly-score thresholds.

        Args:
            warning_threshold: Isolation Forest anomaly score above which a
                warning alert is issued.
            critical_threshold: Anomaly score above which the shipment is
                automatically quarantined.
        """
        super().__init__(contract_name="ColdChainMonitorContract")
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
        }
        if function not in dispatch:
            raise ValueError(f"Unknown chaincode function: {function}")
        result = dispatch[function](**args)
        self._record_transaction(function, args, result)
        return result

    def register_shipment(self, shipment_id: str, product_id: str, target_temp_c: float) -> Dict[str, Any]:
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
        return record

    def record_sensor_reading(
        self, shipment_id: str, temperature_c: float, anomaly_score: float
    ) -> Dict[str, Any]:
        """Record a sensor reading and its Isolation Forest anomaly score.

        Applies the two-tier alert policy: a warning is logged above
        ``warning_threshold`` and the shipment is quarantined above
        ``critical_threshold``.

        Args:
            shipment_id: Shipment the reading belongs to.
            temperature_c: Measured temperature in Celsius.
            anomaly_score: Isolation Forest anomaly score in [0, 1].

        Returns:
            The updated shipment state record.

        Raises:
            KeyError: If the shipment has not been registered.
        """
        key = f"shipment_{shipment_id}"
        record = self.get_state(key)
        if record is None:
            raise KeyError(f"Shipment {shipment_id} is not registered")

        record["readings"].append(
            {"temperature_c": temperature_c, "anomaly_score": anomaly_score}
        )

        if anomaly_score > self.critical_threshold:
            record["status"] = "quarantined"
            record["alerts"].append(
                {"level": "critical", "anomaly_score": anomaly_score, "action": "auto_quarantine"}
            )
        elif anomaly_score > self.warning_threshold:
            if record["status"] != "quarantined":
                record["status"] = "warning"
            record["alerts"].append(
                {"level": "warning", "anomaly_score": anomaly_score, "action": "notify_logistics"}
            )

        self.put_state(key, record)
        return record

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
