"""Round-3: coherent product-lifecycle generator with lifecycle-aware anomalies.

Semantics are fixed in preregistration/ROUND3_lifecycle_PROTOCOL.txt (written before this
code). Unlike simulation/transaction_generator.py (kept unchanged for the archived runs),
each product keeps one drug class and manufacturer, and its events form one connected
custody chain: manufacturer -> logistics -> distributor [-> distributor] -> logistics ->
pharmacy, with observations recorded at the current custodian.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulation.transaction_generator import DRUG_CLASSES, DRUG_CLASS_WEIGHTS, _build_organizations

TRANSFER_TYPES = ("ship", "receive", "custody_transfer")
OBSERVATION_TYPES = ("quality_check", "cold_chain_reading", "verify")
PERMITTED = {("manufacturers", "logistics"), ("logistics", "distributors"),
             ("distributors", "distributors"), ("distributors", "logistics"),
             ("logistics", "pharmacies")}
WINDOW_S = 18 * 30 * 24 * 3600
MEAN_GAP_S = 2 * 24 * 3600

PAPER_COUNTS = {"counterfeit_product": 23000, "temperature_excursion": 46000,
                "custody_break": 34500, "tampered_package": 11500,
                "unauthorized_transfer": 5750}
PAPER_TOTAL = 2_300_000
KEY_MAP = {"counterfeit_products": "counterfeit_product",
           "temperature_excursions": "temperature_excursion",
           "custody_breaks": "custody_break", "tampered_packages": "tampered_package",
           "unauthorized_transfers": "unauthorized_transfer"}


def _lifecycle(rng, orgs: Dict[str, np.ndarray]) -> List[Tuple[str, str, str]]:
    """One product's (transaction_type, from_org, to_org) sequence."""
    pick = lambda t: str(rng.choice(orgs[t]))
    m, l1, d1, l2, p = pick("manufacturers"), pick("logistics"), pick("distributors"), pick("logistics"), pick("pharmacies")
    ev = [("manufacture", m, m), ("quality_check", m, m), ("ship", m, l1)]
    ev += [("cold_chain_reading", l1, l1)] * (1 + int(rng.random() < 0.5))
    ev += [("receive", l1, d1), ("verify", d1, d1)]
    cur = d1
    if rng.random() < 0.2 and len(orgs["distributors"]) > 1:
        d2 = str(rng.choice([d for d in orgs["distributors"] if d != d1]))
        ev += [("custody_transfer", d1, d2), ("verify", d2, d2)]
        cur = d2
    ev += [("ship", cur, l2)]
    ev += [("cold_chain_reading", l2, l2)] * (1 + int(rng.random() < 0.5))
    ev += [("receive", l2, p), ("verify", p, p), ("dispense", p, p)]
    return ev


def generate_lifecycles(n_transactions: int, org_counts: Dict[str, int], seed: int = 42,
                        temperature_mean_c: float = 5.0, temperature_std_c: float = 1.2) -> pd.DataFrame:
    """Generate about ``n_transactions`` events grouped into coherent product lifecycles."""
    rng = np.random.default_rng(seed)
    od = _build_organizations(org_counts)
    orgs = {t: od.loc[od.org_type == t, "org_id"].to_numpy() for t in od.org_type.unique()}
    org_type = dict(zip(od.org_id, od.org_type))
    start = pd.Timestamp("2025-01-01T00:00:00")
    rows, pid = [], 0
    while len(rows) < n_transactions:
        ev = _lifecycle(rng, orgs)
        gaps = rng.exponential(MEAN_GAP_S, size=len(ev)).astype(np.int64) + 60
        offs = np.cumsum(gaps) - gaps[0]
        t0 = int(rng.integers(0, max(1, WINDOW_S - int(offs[-1]))))
        dc = str(rng.choice(DRUG_CLASSES, p=DRUG_CLASS_WEIGHTS))
        for i, ((tt, f, to), o) in enumerate(zip(ev, offs)):
            rows.append((f"prod_{pid:07d}", i, t0 + int(o), dc, f, to, tt))
        pid += 1
    rows = rows[:n_transactions]
    df = pd.DataFrame(rows, columns=["product_id", "event_index", "offset_s", "drug_class",
                                     "from_org", "to_org", "transaction_type"])
    n = len(df)
    df["timestamp"] = start + pd.to_timedelta(df.pop("offset_s"), unit="s")
    df["from_org_type"] = df.from_org.map(org_type)
    df["to_org_type"] = df.to_org.map(org_type)
    df["is_transfer"] = df.transaction_type.isin(TRANSFER_TYPES)
    temp = rng.normal(temperature_mean_c, temperature_std_c, size=n)
    amb = rng.random(n) < 0.15
    temp[amb] = rng.normal(21.0, 2.0, size=amb.sum())
    df["temperature_c"] = np.round(temp, 2)
    df["quantity"] = rng.integers(1, 5000, size=n)
    df["location"] = rng.choice(["US-East", "US-West", "EU-Central", "EU-North", "APAC-East", "APAC-South"], size=n)
    df = df.sort_values(["timestamp", "product_id", "event_index"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"tx_{i:09d}" for i in range(n)])
    df["is_anomaly"] = False
    df["anomaly_type"] = "none"
    df["anomaly_severity"] = 0.0
    df["counterfeit_onset"] = False
    df.attrs["org_type"] = org_type
    df.attrs["orgs"] = {k: list(v) for k, v in orgs.items()}
    return df


def inject_lifecycle_anomalies(df: pd.DataFrame, anomaly_counts: Dict[str, int] | None = None,
                               seed: int = 42, clustered: bool = False,
                               cluster_fraction: float = 0.3) -> pd.DataFrame:
    """Inject anomalies with the lifecycle semantics of the round-3 protocol."""
    rng = np.random.default_rng(seed + 7919)
    attrs = dict(df.attrs)
    org_type = attrs["org_type"]; orgs = attrs["orgs"]
    counts = {KEY_MAP.get(k, k): v for k, v in (anomaly_counts or PAPER_COUNTS).items()}
    n = len(df)
    want = {k: int(round(v / PAPER_TOTAL * n)) for k, v in counts.items()}
    # work on plain lists for speed
    frm = df.from_org.tolist(); to = df.to_org.tolist(); ftyp = df.from_org_type.tolist()
    ttyp = df.to_org_type.tolist(); ttype = df.transaction_type.tolist()
    is_tr = df.is_transfer.tolist(); temp = df.temperature_c.to_numpy().copy()
    an = [False] * n; kind = ["none"] * n; sevs = [0.0] * n; onset = [False] * n
    order = np.lexsort((df.event_index.to_numpy(), df.product_id.to_numpy()))
    by_prod: Dict[str, List[int]] = {}
    pids = df.product_id.to_numpy()
    for i in order:
        by_prod.setdefault(pids[i], []).append(int(i))
    plist = list(by_prod)
    used = set()

    def label(i, k, sv):
        an[i] = True; kind[i] = k; sevs[i] = sv

    prods = [p for p in plist if len(by_prod[p]) >= 4]
    compromised = []
    if clustered:
        k = max(1, int(round(cluster_fraction * len(orgs["manufacturers"]))))
        compromised = sorted(rng.choice(orgs["manufacturers"], size=k, replace=False).tolist())
        cs = set(compromised)
        prods = [p for p in prods if frm[by_prod[p][0]] in cs]
    for p in rng.choice(prods, size=min(want.get("counterfeit_product", 0), len(prods)), replace=False):
        ix = by_prod[p]
        tr = [j for j, i in enumerate(ix) if is_tr[i]]
        j = tr[0] if clustered else int(rng.choice(tr))
        sv = float(rng.uniform(0.4, 1.0))
        for i in ix[j:]:
            label(i, "counterfeit_product", sv)
        onset[ix[j]] = True; used.add(p)
    free = [p for p in plist if p not in used]
    for p in rng.choice(free, size=min(want.get("tampered_package", 0), len(free)), replace=False):
        ix = by_prod[p]
        insp = [j for j, i in enumerate(ix) if ttype[i] in ("verify", "quality_check")]
        if not insp: continue
        j = int(rng.choice(insp)); sv = float(rng.uniform(0.4, 1.0))
        for i in ix[j:]:
            label(i, "tampered_package", sv)
        used.add(p)
    free = [p for p in plist if p not in used]
    for p in rng.choice(free, size=min(want.get("custody_break", 0), len(free)), replace=False):
        ix = by_prod[p]
        i = int(rng.choice([i for i in ix if is_tr[i]])); f = frm[i]
        alt = [o for o in orgs[org_type[f]] if o != f]
        if not alt: continue
        frm[i] = str(rng.choice(alt))
        label(i, "custody_break", float(rng.uniform(0.4, 1.0))); used.add(p)
    free = [p for p in plist if p not in used]
    for p in rng.choice(free, size=min(want.get("unauthorized_transfer", 0), len(free)), replace=False):
        ix = by_prod[p]
        tr = [j for j, i in enumerate(ix) if is_tr[i]]
        if len(tr) < 2: continue
        j = int(rng.choice(tr[:-1])); i = ix[j]; ft = ftyp[i]
        bad_types = [t for t in ("manufacturers", "distributors", "logistics", "pharmacies")
                     if (ft, t) not in PERMITTED and orgs.get(t)]
        u = str(rng.choice(orgs[str(rng.choice(bad_types))]))
        to[i] = u; ttyp[i] = org_type[u]
        for i2 in ix[j + 1:]:
            if is_tr[i2]:
                frm[i2] = u; ftyp[i2] = org_type[u]; break
            frm[i2] = to[i2] = u; ftyp[i2] = ttyp[i2] = org_type[u]
        label(i, "unauthorized_transfer", float(rng.uniform(0.4, 1.0))); used.add(p)
    cands = [i for i in range(n) if ttype[i] == "cold_chain_reading" and not an[i]]
    sel = rng.choice(cands, size=min(want.get("temperature_excursion", 0), len(cands)), replace=False)
    for i in sel:
        label(int(i), "temperature_excursion", float(rng.uniform(0.4, 1.0)))
        temp[i] += float(rng.choice([-1, 1]) * rng.uniform(4, 12))
    out = df.copy()
    out["from_org"] = frm; out["to_org"] = to; out["from_org_type"] = ftyp; out["to_org_type"] = ttyp
    out["temperature_c"] = temp; out["is_anomaly"] = an; out["anomaly_type"] = kind
    out["anomaly_severity"] = sevs; out["counterfeit_onset"] = onset
    out.attrs = attrs; out.attrs["compromised_manufacturers"] = compromised
    return out


def build_lifecycle_transactions(config, seed: int, clustered: bool) -> pd.DataFrame:
    sim = config["simulation"]; dq = config.get("data_quality", {})
    df = generate_lifecycles(sim["transactions_per_run"], sim["organizations"], seed=seed,
                             temperature_mean_c=dq.get("temperature_mean_c", 5.0),
                             temperature_std_c=dq.get("temperature_std_c", 1.2))
    return inject_lifecycle_anomalies(df, sim["anomaly_counts"], seed=seed, clustered=clustered,
                                      cluster_fraction=dq.get("counterfeit_cluster_fraction", 0.3))
