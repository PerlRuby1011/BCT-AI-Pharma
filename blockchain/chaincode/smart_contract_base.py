"""Abstract base class for simulated Hyperledger Fabric chaincode contracts.

A real Fabric chaincode is invoked/queried through the peer's gRPC shim
against a versioned world state (typically backed by CouchDB/LevelDB). This
module simulates that programming model in pure Python with an in-memory
key-value world state, so the counterfeit-detection and cold-chain-monitor
contracts in this package can be unit tested without a live Fabric network.
"""
from __future__ import annotations

import copy
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Transaction:
    """A record of a single chaincode invocation, mirroring a Fabric transaction.

    Attributes:
        tx_id: Unique transaction identifier.
        timestamp: UTC invocation time.
        function: Chaincode function name that was invoked.
        args: Arguments passed to the function.
        result: Return value of the invocation.
    """

    tx_id: str
    timestamp: datetime
    function: str
    args: Dict[str, Any]
    result: Any


class SmartContractBase(ABC):
    """Base class simulating a Hyperledger Fabric chaincode's world state and ledger.

    Subclasses implement domain logic (e.g. counterfeit detection, cold chain
    monitoring) on top of the ``put_state``/``get_state`` primitives and the
    append-only ``ledger`` of invocations, mirroring the Fabric programming
    model closely enough to demonstrate smart-contract-driven automation
    (e.g. automatic quarantine) without requiring a live network.
    """

    def __init__(self, contract_name: str) -> None:
        """Initialize an empty world state and ledger for this contract.

        Args:
            contract_name: Human-readable name of the chaincode contract.
        """
        self.contract_name = contract_name
        self._world_state: Dict[str, Any] = {}
        self.ledger: List[Transaction] = []

    def put_state(self, key: str, value: Any) -> None:
        """Write a value into the world state.

        Args:
            key: World-state key.
            value: JSON-serializable value to store.
        """
        self._world_state[key] = copy.deepcopy(value)

    def get_state(self, key: str) -> Optional[Any]:
        """Read a value from the world state.

        Args:
            key: World-state key.

        Returns:
            The stored value, or ``None`` if the key does not exist.
        """
        return copy.deepcopy(self._world_state.get(key))

    def delete_state(self, key: str) -> None:
        """Delete a key from the world state, if present.

        Args:
            key: World-state key to remove.
        """
        self._world_state.pop(key, None)

    def query_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Return all world-state entries whose key starts with ``prefix``.

        Args:
            prefix: Key prefix to match (e.g. ``"product_"``).

        Returns:
            Mapping of matching keys to their stored values.
        """
        return {
            k: copy.deepcopy(v)
            for k, v in self._world_state.items()
            if k.startswith(prefix)
        }

    def _record_transaction(self, function: str, args: Dict[str, Any], result: Any) -> Transaction:
        """Append an invocation record to the ledger.

        Args:
            function: Name of the invoked chaincode function.
            args: Arguments supplied to the function.
            result: Return value produced by the function.

        Returns:
            The recorded :class:`Transaction`.
        """
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            function=function,
            args=args,
            result=result,
        )
        self.ledger.append(tx)
        return tx

    @abstractmethod
    def invoke(self, function: str, args: Dict[str, Any]) -> Any:
        """Invoke a state-changing chaincode function.

        Args:
            function: Name of the function to invoke.
            args: Function arguments.

        Returns:
            The function's return value.
        """
        raise NotImplementedError

    def get_ledger_history(self, function: Optional[str] = None) -> List[Transaction]:
        """Return the recorded transaction history, optionally filtered by function.

        Args:
            function: If provided, only return transactions for this function.

        Returns:
            List of recorded :class:`Transaction` objects in invocation order.
        """
        if function is None:
            return list(self.ledger)
        return [tx for tx in self.ledger if tx.function == function]
