"""Counterfeit detection smart contract.

Records CNN packaging-verification results and Product Trust Score (PTS)
updates on-chain, and automatically triggers product quarantine when
``PTS < 0.50`` (Section III-C of the paper).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Set

from blockchain.chaincode.smart_contract_base import SmartContractBase
from utils import get_logger

logger = get_logger(__name__)

QUARANTINE_THRESHOLD = 0.50
ALERT_THRESHOLD = 0.75


class CounterfeitDetectionContract(SmartContractBase):
    """Chaincode contract for counterfeit verification and trust-score gating."""

    def __init__(
        self,
        quarantine_threshold: float = QUARANTINE_THRESHOLD,
        alert_threshold: float = ALERT_THRESHOLD,
    ) -> None:
        """Initialize the contract with configurable PTS thresholds.

        Args:
            quarantine_threshold: PTS value below which a product is
                automatically quarantined.
            alert_threshold: PTS value below which a quality alert is raised.
        """
        super().__init__(contract_name="CounterfeitDetectionContract")
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
        self, product_id: str, cnn_authenticity_score: float, tamper_class: str
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
        self.put_state(key, record)
        return record

    def update_trust_score(self, product_id: str, trust_score: float) -> Dict[str, Any]:
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

        record["trust_score"] = trust_score
        if trust_score < self.quarantine_threshold:
            record["status"] = "QUARANTINE"
            logger.warning(
                "Product %s QUARANTINED: PTS=%.4f < threshold %.2f",
                product_id,
                trust_score,
                self.quarantine_threshold,
            )
        elif trust_score < self.alert_threshold:
            record["status"] = "ALERT"
            logger.info(
                "Product %s flagged ALERT: PTS=%.4f < threshold %.2f",
                product_id,
                trust_score,
                self.alert_threshold,
            )
        else:
            record["status"] = "OK"

        record["status_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.put_state(key, record)
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
