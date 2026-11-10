"""Round-3 lifecycle invariants (see preregistration/ROUND3_lifecycle_PROTOCOL.txt)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.lifecycle_generator import PERMITTED, TRANSFER_TYPES


def lifecycle_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per invariant violation: product_id, transaction_id, rule."""
    d = df.sort_values(["product_id", "event_index"], kind="mergesort")
    pid = d.product_id.to_numpy(); tid = d.transaction_id.to_numpy()
    tt = d.transaction_type.to_numpy(); fr = d.from_org.to_numpy(); to = d.to_org.to_numpy()
    ft = d.from_org_type.to_numpy(); tty = d.to_org_type.to_numpy()
    ts = d.timestamp.to_numpy(); dc = d.drug_class.to_numpy()
    is_tr = np.isin(tt, TRANSFER_TYPES)
    out = []
    cust = None
    for k in range(len(d)):
        new = k == 0 or pid[k] != pid[k - 1]
        p = pid[k]
        if new:
            if tt[k] != "manufacture":
                out.append((p, tid[k], "first_not_manufacture"))
            cust = to[k]
            continue
        if dc[k] != dc[k - 1]:
            out.append((p, tid[k], "drug_class_changes"))
        if not ts[k] > ts[k - 1]:
            out.append((p, tid[k], "timestamps_not_increasing"))
        if is_tr[k]:
            if fr[k] != cust:
                out.append((p, tid[k], "transfer_not_from_custodian"))
            if (ft[k], tty[k]) not in PERMITTED:
                out.append((p, tid[k], "transfer_type_not_permitted"))
            cust = to[k]
        elif not (fr[k] == to[k] == cust):
            out.append((p, tid[k], "observation_not_at_custodian"))
    return pd.DataFrame(out, columns=["product_id", "transaction_id", "rule"])
