"""Counterfeit detection smart contract.

Records CNN packaging-verification results and Product Trust Score (PTS)
updates on-chain, and automatically triggers product quarantine when
``PTS < 0.50`` -- the paper states "values below 0.50 initiate automatic
quarantine via smart contract execution."
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Set

from blockchain.chaincode.product_state_machine import (
    ProductStateMachine, load_machine, save_machine, sync_shipment_records,
)
from blockchain.chaincode.smart_contract_base import SmartContractBase
from utils import get_logger

logger = get_logger(__name__)

QUARANTINE_THRESHOLD = 0.50
ALERT_THRESHOLD = 0.75
_STATUS_LABEL = {"ACTIVE": "OK", "ALERT": "ALERT", "QUARANTINE": "QUARANTINE",
                 "DISPOSED": "DISPOSED"}


class CounterfeitDetectionContract(SmartContractBase):
    """Chaincode contract for counterfeit verification and trust-score gating."""

    def __init__(
        self,
        quarantine_threshold: float = QUARANTINE_THRESHOLD,
        alert_threshold: float = ALERT_THRESHOLD,
        world_state: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the contract with configurable PTS thresholds.

        Args:
            quarantine_threshold: PTS value below which a product is
                automatically quarantined.
            alert_threshold: PTS value below which a quality alert is raised.
            world_state: Optional world state shared with the cold-chain
                contract, so SC-4, SC-5 and SC-7 act on one product machine.
        """
        super().__init__(contract_name="CounterfeitDetectionContract", world_state=world_state)
        self.quarantine_threshold = quarantine_threshold
        self.alert_threshold = alert_threshold

    def invoke(self, function: str, args: Dict[str, Any]) -> Any:
        """Dispatch to the named contract function.

        Args:
            function: One of ``register_product``, ``record_verification``,
                ``update_trust_score``, ``get_product_status``.
            args: Function-specific keyword arguments.

        Returns:
            The result of the dispatched function.

        Raises:
            ValueError: If ``function`` is not a recognized contract method.
        """
        dispatch = {
            "register_product": self.register_product,
            "record_verification": self.record_verification,
            "update_trust_score": self.update_trust_score,
            "get_product_status": self.get_product_status,
            "release_quarantine": self.release_quarantine,
        }
        if function not in dispatch:
            raise ValueError(f"Unknown chaincode function: {function}")
        result = dispatch[function](**args)
        self._record_transaction(function, args, result)
        return result

    def register_product(self, product_id: str, drug_class: str, manufacturer: str) -> Dict[str, Any]:
        """Register a new product on the ledger.

        Args:
            product_id: Unique product identifier.
            drug_class: Drug classification (``A``, ``B``, or ``C``).
            manufacturer: Manufacturer org ID.

        Returns:
            The newly created product state record.
        """
        record = {
            "product_id": product_id,
            "drug_class": drug_class,
            "manufacturer": manufacturer,
            "status": "OK",
            "trust_score": 1.0,
            "verification_count": 0,
            "cnn_authenticity_scores": [],
        }
        self.put_state(f"product_{product_id}", record)
        return record

    def record_verification(
        self, product_id: str, cnn_authenticity_score: float, tamper_class: str, t: float,
        received_t: float | None = None,
    ) -> Dict[str, Any]:
        """Record a CNN packaging-verification result on-chain.

        Args:
            product_id: Product being verified.
            cnn_authenticity_score: CNN authenticity score in [0, 1].
            tamper_class: Predicted tamper class label.

        Returns:
            The updated product state record.

        Raises:
            KeyError: If the product has not been registered.
        """
        key = f"product_{product_id}"
        record = self.get_state(key)
        if record is None:
            raise KeyError(f"Product {product_id} is not registered")
        record["verification_count"] += 1
        record["cnn_authenticity_scores"].append(cnn_authenticity_score)
        record["last_tamper_class"] = tamper_class
        # Round-2 review: SC-4 now enforces through the shared product machine.
        machine = self._machine(product_id)
        machine.cnn_reading(t, cnn_authenticity_score, received_t=received_t)
        self._save(machine, record)
        return record

    def _save(self, machine: ProductStateMachine, record: Dict[str, Any]) -> None:
        """Persist the machine and mirror its status onto the product record."""
        save_machine(self, machine)
        record["status"] = _STATUS_LABEL[machine.state]
        record["quarantine_reasons"] = sorted(machine.quarantine_reasons)
        record["alert_conditions"] = sorted(machine.alert_conditions)
        record["status_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.put_state(f"product_{machine.product_id}", record)
        sync_shipment_records(self, machine)

    def update_trust_score(self, product_id: str, trust_score: float, t: float,
                           received_t: float | None = None) -> Dict[str, Any]:
        """Update a product's PTS and apply automated quarantine/alert logic.

        Implements the paper's rule that "values below 0.50 initiate
        automatic quarantine via smart contract execution": products with
        ``trust_score < quarantine_threshold`` are quarantined,
        ``quarantine_threshold <= trust_score < alert_threshold`` raises a
        quality alert, otherwise the product remains clear.

        Args:
            product_id: Product whose trust score is being updated.
            trust_score: New Product Trust Score in [0, 1].

        Returns:
            The updated product state record, with ``status`` set to one of
            ``"QUARANTINE"``, ``"ALERT"``, or ``"OK"``, and a
            ``status_updated_at`` timestamp.

        Raises:
            KeyError: If the product has not been registered.
        """
        key = f"product_{product_id}"
        record = self.get_state(key)
        if record is None:
            raise KeyError(f"Product {product_id} is not registered")

        # 2026-10: status now comes from the unified state machine, so an
        # open quarantine is NOT cleared by a later, higher PTS (it needs an
        # authorised release). For a product's first update the outcome is
        # identical to the original three-way threshold rule.
        machine = self._machine(product_id)
        machine.pts_update(t, trust_score, received_t=received_t)
        record["trust_score"] = trust_score
        self._save(machine, record)
        if record["status"] == "QUARANTINE":
            logger.warning("Product %s QUARANTINE (reasons %s), PTS=%.4f",
                           product_id, record["quarantine_reasons"], trust_score)
        elif record["status"] == "ALERT":
            logger.info("Product %s flagged ALERT: PTS=%.4f", product_id, trust_score)
        return record

    def _machine(self, product_id: str) -> ProductStateMachine:
        """Load the product's enforcement machine from (shared) world state."""
        return load_machine(self, product_id, tau_q=self.quarantine_threshold,
                            tau_a=self.alert_threshold)

    def release_quarantine(self, product_id: str, role: str, reason_code: str,
                           t: float) -> Dict[str, Any]:
        """Authorised release of an open quarantine (QA / regulator only)."""
        key = f"product_{product_id}"
        record = self.get_state(key)
        if record is None:
            raise KeyError(f"Product {product_id} is not registered")
        machine = self._machine(product_id)
        machine.release(t, role, reason_code)
        self._save(machine, record)
        return record

    def get_product_status(self, product_id: str) -> Dict[str, Any]:
        """Query the current on-chain state of a product.

        Args:
            product_id: Product to look up.

        Returns:
            The product's state record.

        Raises:
            KeyError: If the product has not been registered.
        """
        record = self.get_state(f"product_{product_id}")
        if record is None:
            raise KeyError(f"Product {product_id} is not registered")
        return record

    def get_quarantine_count(self) -> int:
        """Count distinct products quarantined at least once this session.

        Scans the ledger's ``update_trust_score`` invocation history (not
        just current world state) so a product that was later re-verified
        and cleared still counts as having been quarantined this session.

        Returns:
            Number of distinct product IDs quarantined this session.
        """
        quarantined_ids: Set[str] = {
            tx.args["product_id"]
            for tx in self.get_ledger_history("update_trust_score")
            if isinstance(tx.result, dict) and tx.result.get("status") == "QUARANTINE"
        }
        return len(quarantined_ids)

    def get_alert_count(self) -> int:
        """Count distinct products flagged for alert (not quarantine) this session.

        Returns:
            Number of distinct product IDs whose most severe status this
            session was ``"ALERT"`` (excludes products that also hit
            ``"QUARANTINE"`` at some point, to avoid double-counting).
        """
        alerted_ids: Set[str] = {
            tx.args["product_id"]
            for tx in self.get_ledger_history("update_trust_score")
            if isinstance(tx.result, dict) and tx.result.get("status") == "ALERT"
        }
        return len(alerted_ids - {
            tx.args["product_id"]
            for tx in self.get_ledger_history("update_trust_score")
            if isinstance(tx.result, dict) and tx.result.get("status") == "QUARANTINE"
        })
