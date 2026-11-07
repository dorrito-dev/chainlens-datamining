"""Stage 4: pattern detection.

NetworkX graph (R1/R3/R4), DBSCAN (outlier flag), Apriori (R5, trained on the
train split only), and Louvain (display-only communities).

Order matters: DBSCAN and PageRank run before Apriori because R5 uses
`dbscan_noise` and `high_pagerank` as items.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent

APRIORI_ITEMS = [
    "high_fanout", "high_tx_count", "low_value", "high_value", "many_round",
    "is_contract", "recent_account", "high_pagerank", "dbscan_noise",
]


def load_detection_config(path: str | Path = "config/detection.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_sanctions(path: str | Path = "config/sanctions.json") -> set[str]:
    """Load the frozen sanctions snapshot, independent of train/test labels.

    [P6][P8] The seed set must survive label re-splitting. A missing or empty
    file returns an empty set so callers can fall back to explicit labels.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return set()
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw = payload.get("addresses", []) if isinstance(payload, dict) else payload
    return {
        str(address).strip().lower()
        for address in raw
        if isinstance(address, str) and address.strip().lower().startswith("0x")
    }


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph(con, addresses: list[str]) -> nx.DiGraph:
    edges = con.execute("""
        SELECT from_address, to_address,
               COUNT(*)::BIGINT AS tx_count,
               SUM(value_eth) AS total_value_eth,
               MIN(block_timestamp) AS first_tx_ts,
               MAX(block_timestamp) AS last_tx_ts
        FROM transactions
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    graph = nx.DiGraph()
    graph.add_nodes_from(addresses)
    for row in edges.itertuples(index=False):
        if row.from_address not in graph:
            graph.add_node(row.from_address)
        if row.to_address not in graph:
            graph.add_node(row.to_address)
        graph.add_edge(
            row.from_address,
            row.to_address,
            tx_count=int(row.tx_count),
            total_value_eth=float(row.total_value_eth),
            first_tx_ts=pd.Timestamp(row.first_tx_ts),
            last_tx_ts=pd.Timestamp(row.last_tx_ts),
        )
    return graph


def cycle_summary(flags: dict[tuple[str, str], dict[str, Any]]) -> dict[str, int]:
    cycles = 0
    addresses = 0
    for (_, signal), payload in flags.items():
        if signal != "cycle_member":
            continue
        addresses += 1
        cycles += int(payload.get("evidence", {}).get("cycle_count", 0))
    return {"addresses": addresses, "cycles": cycles}


# ── R1 ────────────────────────────────────────────────────────────────────────

def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Rotation-invariant cycle key; a 2-cycle has one representation."""
    if len(cycle) <= 2:
        return tuple(sorted(cycle))
    rotations = [cycle[i:] + cycle[:i] for i in range(len(cycle))]
    return min(rotations)


def detect_cycles(
    graph: nx.DiGraph,
    *,
    min_length: int = 2,
    max_length: int = 5,
    value_tolerance: float = 0.02,
    window_hours: float = 24,
    per_scc_cap: int = 1000,
    enum_cap: int = 5000,
    max_scc_size: int = 5000,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    qualifying: list[dict[str, Any]] = []
    skipped_sccs = 0
    sccs = [set(c) for c in nx.strongly_connected_components(graph) if len(c) >= min_length]
    sccs.sort(key=lambda c: sorted(c))

    for scc in sccs:
        if len(scc) > max_scc_size:
            skipped_sccs += 1
            continue
        sub = graph.subgraph(scc)
        found: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        try:
            for cycle in islice(nx.simple_cycles(sub, length_bound=max_length), enum_cap):
                if not (min_length <= len(cycle) <= max_length):
                    continue
                canonical = _canonical_cycle(tuple(cycle))
                if canonical in seen:
                    continue
                seen.add(canonical)
                found.append(tuple(cycle))
        except nx.NetworkXError:
            continue
        found.sort(key=lambda c: (len(c), tuple(sorted(c)), c))

        for cycle in found[:per_scc_cap]:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            values = [float(graph.edges[f, t].get("total_value_eth", 0.0)) for f, t in edges]
            if not values or min(values) <= 0:
                continue
            ratio = max(values) / min(values)
            if ratio > 1 + value_tolerance:
                continue
            firsts = [pd.Timestamp(graph.edges[f, t]["first_tx_ts"]) for f, t in edges]
            lasts = [pd.Timestamp(graph.edges[f, t]["last_tx_ts"]) for f, t in edges]
            if any(pd.isna(t) for t in firsts + lasts):
                continue
            span_h = (max(lasts) - min(firsts)).total_seconds() / 3600
            if span_h > window_hours:
                continue
            qualifying.append({
                "cycle_path": sorted(cycle),
                "cycle_length": len(cycle),
                "edge_values": values,
                "max_min_ratio": round(ratio, 6),
                "value_tolerance": value_tolerance,
                "time_span_hours": round(span_h, 6),
                "scc_size": len(scc),
            })

    by_addr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cycle in qualifying:
        for addr in cycle["cycle_path"]:
            by_addr[addr].append(cycle)

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for addr in sorted(by_addr):
        cycles = by_addr[addr]
        flags[(addr, "cycle_member")] = {
            "signal_value": int(cycles[0]["cycle_length"]),
            "evidence": {"cycle_count": len(cycles), "cycles": cycles[:20]},
        }
    return flags, {
        "qualifying_cycles": len(qualifying),
        "flagged_addresses": len(flags),
        "skipped_sccs": skipped_sccs,
    }


# ── R2 ────────────────────────────────────────────────────────────────────────

def _edge_timestamp(edge: dict[str, Any], key: str) -> pd.Timestamp:
    value = edge.get(key)
    return pd.Timestamp(value) if value is not None else pd.NaT


def _temporal_gather_evidence(
    graph: nx.DiGraph,
    address: str,
    *,
    window_hours: float,
    min_gather_fraction: float,
    min_gather_intermediaries: int,
    min_collectors: int,
) -> dict[str, Any]:
    """Check the gather half of Starnini's scatter-gather motif.

    A scatter edge address -> intermediary must start before an onward
    intermediary -> collector edge, and the gather must happen inside the
    configured window. This turns a plain fan-out into an ordered smurf motif.
    """
    scatter: list[tuple[str, pd.Timestamp]] = []
    for intermediary in graph.successors(address):
        start = _edge_timestamp(graph.edges[address, intermediary], "first_tx_ts")
        if not pd.isna(start):
            scatter.append((intermediary, start))

    valid_intermediaries: set[str] = set()
    collectors: set[str] = set()
    leads: list[float] = []
    pairs = 0
    for intermediary, scatter_start in scatter:
        if intermediary not in graph:
            continue
        for collector in graph.successors(intermediary):
            if collector in (address, intermediary):
                continue
            gather_start = _edge_timestamp(graph.edges[intermediary, collector], "first_tx_ts")
            if pd.isna(gather_start) or gather_start < scatter_start:
                continue
            lead_hours = (gather_start - scatter_start).total_seconds() / 3600
            if lead_hours > window_hours:
                continue
            valid_intermediaries.add(intermediary)
            collectors.add(collector)
            leads.append(lead_hours)
            pairs += 1

    required = max(
        min_gather_intermediaries,
        math.ceil(len(scatter) * min_gather_fraction) if scatter else 0,
    )
    passed = bool(
        pairs
        and len(valid_intermediaries) >= required
        and len(collectors) >= min_collectors
    )
    return {
        "passed": passed,
        "ordering": "scatter_before_gather" if passed else "not_ordered",
        "scatter_counterparties": len(scatter),
        "gather_intermediaries": len(valid_intermediaries),
        "gather_collectors": len(collectors),
        "scatter_gather_pairs": pairs,
        "required_gather_intermediaries": required,
        "gather_window_hours": window_hours,
        "median_scatter_to_gather_hours": (
            round(float(np.median(leads)), 6) if leads else None
        ),
    }


def detect_fanout(
    features: pd.DataFrame,
    *,
    graph: nx.DiGraph | None = None,
    mode: str = "outbound_fanout",
    threshold: int = 20,
    min_outbound: int = 20,
    min_ratio: float = 0.5,
    require_temporal_ordering: bool = False,
    gather_window_hours: float = 72,
    min_gather_fraction: float = 0.1,
    min_gather_intermediaries: int = 2,
    min_collectors: int = 1,
    deck_min_out_in_ratio: float = 5.0,
    deck_min_counterparties: int = 50,
) -> dict[tuple[str, str], dict[str, Any]]:
    df = features.copy()
    for col in ("unique_counterparties", "tx_count_out", "fanout_ratio", "out_in_ratio"):
        if col not in df.columns:
            df[col] = 0.0
    df["out_in_ratio"] = pd.to_numeric(df["out_in_ratio"], errors="coerce")
    deck_equivalent = (
        (df["out_in_ratio"].fillna(0) >= deck_min_out_in_ratio)
        & (df["unique_counterparties"] >= deck_min_counterparties)
    )
    if mode == "deck_out_in":
        mask = deck_equivalent
    else:
        mask = (
            (df["unique_counterparties"] >= threshold)
            & (df["tx_count_out"] >= min_outbound)
            & (pd.to_numeric(df["fanout_ratio"], errors="coerce").fillna(0) >= min_ratio)
        )

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for row in df.loc[mask].itertuples(index=False):
        out_in = getattr(row, "out_in_ratio", None)
        if graph is not None and require_temporal_ordering:
            temporal = _temporal_gather_evidence(
                graph, row.address,
                window_hours=gather_window_hours,
                min_gather_fraction=min_gather_fraction,
                min_gather_intermediaries=min_gather_intermediaries,
                min_collectors=min_collectors,
            )
            if not temporal["passed"]:
                continue
        else:
            temporal = {
                "passed": None,
                "ordering": "not_evaluated",
                "scatter_counterparties": int(row.unique_counterparties),
                "gather_intermediaries": None,
                "gather_collectors": None,
                "scatter_gather_pairs": None,
                "required_gather_intermediaries": None,
                "gather_window_hours": gather_window_hours,
                "median_scatter_to_gather_hours": None,
            }

        flags[(row.address, "fanout")] = {
            "signal_value": int(row.unique_counterparties),
            "strength": 1.0,
            "evidence": {
                "mode": mode,
                "unique_counterparties": int(row.unique_counterparties),
                "tx_count_out": int(row.tx_count_out),
                "fanout_ratio": float(row.fanout_ratio),
                "out_in_ratio": None if pd.isna(out_in) else float(out_in),
                "r2_deck_equivalent": bool(
                    (not pd.isna(out_in) and out_in >= deck_min_out_in_ratio)
                    and row.unique_counterparties >= deck_min_counterparties
                ),
                "temporal_ordering": temporal,
            },
        }
    return flags


# ── R3 ────────────────────────────────────────────────────────────────────────

def detect_sanctioned_1hop(
    graph: nx.DiGraph,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    sanctions: set[str] | None = None,
    transactions: pd.DataFrame | None = None,
    min_value_eth: float = 0.01,
    min_direct_strength: float = 0.1,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Direct sanctioned exposure with a dusting gate and continuous severity.

    [P6][P8] The seed set is the frozen sanctions snapshot union all labelled
    sanctioned addresses, never a train/test slice. [P6] Transfers below
    `min_value_eth` are ignored so dusting attacks cannot manufacture exposure.
    The reported `exposure_fraction` scales the contribution; a confirmed
    direct link keeps `min_direct_strength` so large legitimate volume cannot
    dilute it to zero.
    """
    del splits  # retained in the signature for callers; seeds no longer depend on it
    labelled = set()
    if labels is not None and not labels.empty and {"address", "label"} <= set(labels.columns):
        labelled = {
            str(address).strip().lower()
            for address in labels.loc[labels["label"] == "sanctioned", "address"]
        }
    seeds = {str(seed).strip().lower() for seed in (sanctions or set())} | labelled
    if not seeds:
        return {}

    matches: dict[str, dict[str, Any]] = {}
    exposure: dict[str, dict[str, float]] = {}

    def record(neighbour: str, seed: str, direction: str, signed_in: float, signed_out: float) -> None:
        if neighbour == seed or neighbour in seeds:
            return
        entry = matches.setdefault(neighbour, {"seeds": set(), "directions": set()})
        entry["seeds"].add(seed)
        entry["directions"].add(direction)
        bucket = exposure.setdefault(neighbour, {"in": 0.0, "out": 0.0})
        bucket["in"] += signed_in
        bucket["out"] += signed_out

    if transactions is not None and not transactions.empty:
        tx = transactions.copy()
        tx["from_address"] = tx["from_address"].astype(str).str.lower()
        tx["to_address"] = tx["to_address"].astype(str).str.lower()
        tx["tx_value"] = pd.to_numeric(tx["value_eth"], errors="coerce").fillna(0.0)
        inflow_total = tx.groupby("to_address")["tx_value"].sum()
        outflow_total = tx.groupby("from_address")["tx_value"].sum()

        qualifying = tx[(tx["tx_value"] >= min_value_eth)
                        & (tx["from_address"].isin(seeds) | tx["to_address"].isin(seeds))]
        for row in qualifying.itertuples(index=False):
            if row.from_address in seeds and row.to_address not in seeds:
                # Seed sends out: the neighbour receives sanctioned inflow.
                record(row.to_address, row.from_address, "out",
                       signed_in=float(row.tx_value), signed_out=0.0)
            if row.to_address in seeds and row.from_address not in seeds:
                # Neighbour sends to the seed: sanctioned outflow.
                record(row.from_address, row.to_address, "in",
                       signed_in=0.0, signed_out=float(row.tx_value))

        for addr, bucket in exposure.items():
            total_in = float(inflow_total.get(addr, 0.0))
            total_out = float(outflow_total.get(addr, 0.0))
            bucket["total_in"] = total_in
            bucket["total_out"] = total_out
            bucket["in_fraction"] = bucket["in"] / total_in if total_in > 0 else (1.0 if bucket["in"] > 0 else 0.0)
            bucket["out_fraction"] = bucket["out"] / total_out if total_out > 0 else (1.0 if bucket["out"] > 0 else 0.0)
    else:
        for seed in sorted(seeds):
            if seed not in graph:
                continue
            for direction, neighbours in (
                ("out", list(graph.successors(seed))),
                ("in", list(graph.predecessors(seed))),
            ):
                for neighbour in neighbours:
                    value = float(graph.edges[seed, neighbour].get("total_value_eth", 0.0)
                                  if direction == "out"
                                  else graph.edges[neighbour, seed].get("total_value_eth", 0.0))
                    if value < min_value_eth:
                        continue
                    record(neighbour, seed, direction,
                           signed_in=value if direction == "out" else 0.0,
                           signed_out=value if direction == "in" else 0.0)

        for addr, bucket in exposure.items():
            total_in = float(sum(graph.edges[p, addr].get("total_value_eth", 0.0)
                                 for p in graph.predecessors(addr))) if addr in graph else 0.0
            total_out = float(sum(graph.edges[addr, s].get("total_value_eth", 0.0)
                                  for s in graph.successors(addr))) if addr in graph else 0.0
            bucket["total_in"] = total_in
            bucket["total_out"] = total_out
            bucket["in_fraction"] = bucket["in"] / total_in if total_in > 0 else (1.0 if bucket["in"] > 0 else 0.0)
            bucket["out_fraction"] = bucket["out"] / total_out if total_out > 0 else (1.0 if bucket["out"] > 0 else 0.0)

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for addr in sorted(matches):
        entry = matches[addr]
        directions = sorted(entry["directions"])
        bucket = exposure.get(addr, {"in": 0.0, "out": 0.0, "total_in": 0.0,
                                     "total_out": 0.0, "in_fraction": 0.0,
                                     "out_fraction": 0.0})
        exposure_fraction = min(1.0, max(bucket["in_fraction"], bucket["out_fraction"]))
        strength = min(1.0, max(exposure_fraction, min_direct_strength))
        flags[(addr, "sanctioned_1hop")] = {
            "signal_value": 1,
            "strength": strength,
            "evidence": {
                "seed": sorted(entry["seeds"])[0],
                "direction": directions[0] if len(directions) == 1 else "both",
                "seeds": sorted(entry["seeds"]),
                "directions": directions,
                "min_value_eth": min_value_eth,
                "dusting_floor_eth": min_value_eth,
                "sanctioned_inflow_eth": round(bucket["in"], 18),
                "sanctioned_outflow_eth": round(bucket["out"], 18),
                "total_inflow_eth": round(bucket["total_in"], 18),
                "total_outflow_eth": round(bucket["total_out"], 18),
                "inflow_exposure_fraction": bucket["in_fraction"],
                "outflow_exposure_fraction": bucket["out_fraction"],
                "exposure_fraction": exposure_fraction,
                "strength": strength,
                "seed_count": len(seeds),
            },
        }
    return flags


# ── R4 ────────────────────────────────────────────────────────────────────────

def _percentile_ranks(scores: dict[str, float]) -> dict[str, float]:
    """Average-rank percentile in (0, 100], ties share one rank."""
    ordered = sorted(scores.items(), key=lambda item: item[1])
    n = len(ordered)
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        percentile = 100.0 * ((i + 1 + j + 1) / 2.0) / n
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = percentile
        i = j + 1
    return ranks


def compute_pagerank(
    graph: nx.DiGraph,
    *,
    percentile: float = 95,
    alpha: float = 0.85,
    weight: str = "total_value_eth",
    zero_floor: float = 1e-12,
    continuous: bool = False,
    min_percentile: float = 90,
    min_strength: float = 0.05,
    exclude_addresses: set[str] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    """Value-weighted PageRank with optional continuous, known-address-safe output.

    [P9] High-PageRank nodes are often exchanges, bridges and routers, so
    `exclude_addresses` suppresses them before flagging. [P3][P10] When
    `continuous` is set, the upper tail scores proportionally instead of being
    split by a hard top-5% gate; `min_strength` removes negligible tail mass.
    """
    def edge_weight(_u: str, _v: str, data: dict[str, Any]) -> float:
        return max(float(data.get(weight, 0.0)), zero_floor)

    scores = nx.pagerank(
        graph, alpha=alpha, weight=edge_weight, max_iter=100, tol=1e-6,
    )
    if not scores:
        return {}, {}

    values = list(scores.values())
    threshold = float(np.percentile(values, percentile))
    ranks = _percentile_ranks(scores) if continuous else {}
    excluded = exclude_addresses or set()
    flags: dict[tuple[str, str], dict[str, Any]] = {}

    for addr in sorted(graph.nodes):
        if addr in excluded:
            continue
        score = float(scores[addr])
        top5 = score >= threshold
        if continuous:
            pct = ranks.get(addr, 0.0)
            strength = max(0.0, min(1.0, (pct - min_percentile) / (100.0 - min_percentile)))
            if strength < min_strength and not top5:
                continue
            if strength <= 0.0:
                continue
            flags[(addr, "pagerank_top5")] = {
                "signal_value": 1 if top5 else 0,
                "strength": strength,
                "evidence": {
                    "pagerank": score,
                    "threshold": threshold,
                    "percentile": pct,
                    "min_percentile": min_percentile,
                    "top5": top5,
                    "strength": strength,
                },
            }
        elif score >= threshold:
            flags[(addr, "pagerank_top5")] = {
                "signal_value": 1,
                "strength": 1.0,
                "evidence": {"pagerank": score, "threshold": threshold, "top5": True},
            }
    return flags, scores


# ── DBSCAN ────────────────────────────────────────────────────────────────────

def run_dbscan(
    features: pd.DataFrame,
    *,
    eps: float = 3.195323,
    min_samples: int = 5,
    min_tx_total: int = 3,
) -> tuple[dict[str, int], dict[str, Any]]:
    scaled_cols = [c for c in features.columns if c.endswith("_scaled")]
    eligible = features[features["tx_count_total"] >= min_tx_total].copy()
    if eligible.empty or not scaled_cols:
        return {}, {"eligible": 0, "clusters": 0, "noise": 0, "eps": eps, "min_samples": min_samples}

    matrix = eligible[scaled_cols].to_numpy(dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
    mapping = dict(zip(eligible["address"].tolist(), labels.tolist(), strict=False))
    cluster_count = len(set(labels.tolist()) - {-1})
    noise_count = int((labels == -1).sum())
    return mapping, {
        "eligible": int(len(eligible)),
        "clusters": cluster_count,
        "noise": noise_count,
        "eps": eps,
        "min_samples": min_samples,
        "scaled_features": scaled_cols,
    }


# ── R5 ────────────────────────────────────────────────────────────────────────

def _boolean_table(
    features: pd.DataFrame,
    pagerank_flags: dict[tuple[str, str], dict[str, Any]],
    dbscan_labels: dict[str, int],
    train_positive: set[str],
    *,
    exclude_test: bool,
    test_addresses: set[str],
) -> pd.DataFrame:
    df = features.copy()
    if exclude_test:
        df = df[~df["address"].isin(test_addresses)].copy()
    table = pd.DataFrame({"address": df["address"].values})
    table["high_fanout"] = df["unique_counterparties"].fillna(0).values >= 20
    table["high_tx_count"] = df["tx_count_total"].fillna(0).values >= 10
    table["low_value"] = df["median_value_eth"].fillna(0).values < 0.1
    table["high_value"] = df["total_value_eth"].fillna(0).values >= 10
    table["many_round"] = df["round_number_ratio"].fillna(0).values >= 0.5
    table["is_contract"] = df["is_contract"].fillna(False).values.astype(bool)
    window_end = pd.to_datetime(df["last_seen"], utc=True).max() if "last_seen" in df.columns else pd.NaT
    if pd.isna(window_end):
        table["recent_account"] = False
    else:
        first_seen = pd.to_datetime(df["first_seen"], utc=True)
        table["recent_account"] = (first_seen >= (window_end - pd.DateOffset(days=7))).to_numpy(dtype=bool)
    pagerank_addresses = {
        key[0] for key, value in pagerank_flags.items()
        if key[1] == "pagerank_top5"
        and bool((value.get("evidence") or {}).get("top5", value.get("signal_value") == 1))
    }
    table["high_pagerank"] = df["address"].isin(pagerank_addresses).values
    table["dbscan_noise"] = df["address"].map(lambda a: dbscan_labels.get(a) == -1).values
    table["flagged"] = df["address"].isin(train_positive).values
    for col in APRIORI_ITEMS + ["flagged"]:
        table[col] = table[col].astype(bool)
    return table


def mine_apriori(
    features: pd.DataFrame,
    pagerank_flags: dict[tuple[str, str], dict[str, Any]],
    dbscan_labels: dict[str, int],
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    min_support: float = 0.05,
    min_lift: float = 2.0,
    min_confidence: float = 0.8,
    max_antecedent_size: int = 3,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], str]:
    train_addrs = set(splits.loc[splits["split"] == "train", "address"])
    test_addrs = set(splits.loc[splits["split"] == "test", "address"])
    train_positive = set(labels.loc[
        (labels["label"] != "negative") & (labels["address"].isin(train_addrs)),
        "address",
    ])
    if not train_positive:
        return {}, [], "no train positives"

    # Apriori is fitted on the labelled train cohort, not on every graph
    # counterparty. Mining the full graph dilutes `flagged` support below
    # min_support when labelled data is sparse.
    labelled_train = set(labels.loc[
        (labels["label"] != "context") & (labels["address"].isin(train_addrs)),
        "address",
    ]) & set(features["address"])
    mining_features = features[features["address"].isin(labelled_train)].copy()
    if len(mining_features) < 10 or not mining_features["address"].isin(train_positive).any():
        return {}, [], "labelled train cohort too small"

    mining = _boolean_table(
        mining_features, pagerank_flags, dbscan_labels, train_positive,
        exclude_test=True, test_addresses=test_addrs,
    )
    scoring = _boolean_table(
        features, pagerank_flags, dbscan_labels, train_positive,
        exclude_test=False, test_addresses=test_addrs,
    )
    if mining.empty or not mining["flagged"].any():
        return {}, [], "empty mining table"

    items = mining[APRIORI_ITEMS + ["flagged"]].astype(bool)
    frequent = apriori(
        items, min_support=min_support, use_colnames=True,
        max_len=max_antecedent_size + 1,
    )
    if frequent.empty:
        return {}, [], "no frequent itemsets"

    with warnings.catch_warnings():
        # mlxtend divides by zero when computing a certainty factor on constant
        # items; the lift/confidence fields used here are unaffected.
        warnings.simplefilter("ignore", RuntimeWarning)
        rules_all = association_rules(frequent, metric="lift", min_threshold=1.0)
    flagged_rules = []
    if not rules_all.empty:
        for row in rules_all.itertuples(index=False):
            antecedents = set(row.antecedents)
            consequents = set(row.consequents)
            if consequents != {"flagged"} or "flagged" in antecedents:
                continue
            flagged_rules.append(row)
    best_lift = max((float(row.lift) for row in flagged_rules), default=0.0)

    valid: list[dict[str, Any]] = []
    for row in flagged_rules:
        antecedents = set(row.antecedents)
        if len(antecedents) > max_antecedent_size:
            continue
        if float(row.lift) < min_lift:
            continue
        if float(row.confidence) < min_confidence:
            continue
        valid.append({
            "antecedents": tuple(sorted(antecedents)),
            "support": float(row.support),
            "confidence": float(row.confidence),
            "lift": float(row.lift),
        })

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for rule in valid:
        antecedent = list(rule["antecedents"])
        mask = scoring[antecedent].all(axis=1)
        for addr in scoring.loc[mask, "address"]:
            key = (addr, "apriori_match")
            candidate = {
                "signal_value": rule["confidence"],
                "evidence": {
                    "antecedent": antecedent,
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "confidence_min": min_confidence,
                },
            }
            if key not in flags or candidate["signal_value"] > flags[key]["signal_value"]:
                flags[key] = candidate
    return flags, valid, (
        f"{len(valid)} rules (best lift {best_lift:.3f}, min_lift {min_lift:.1f}, "
        f"min_confidence {min_confidence:.2f})"
    )


# ── Louvain ───────────────────────────────────────────────────────────────────

def detect_louvain(
    graph: nx.DiGraph,
    *,
    seed: int = 42,
    weight: str = "total_value_eth",
    resolution: float = 1.0,
) -> dict[tuple[str, str], dict[str, Any]]:
    if graph.number_of_nodes() == 0:
        return {}
    undirected = graph.to_undirected()
    communities = nx.community.louvain_communities(
        undirected, weight=weight, seed=seed, resolution=resolution,
    )
    # Stable ids: order by the smallest address in each community.
    ordered = sorted(communities, key=lambda c: min(c) if c else "")
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for community_id, members in enumerate(ordered):
        size = len(members)
        for addr in sorted(members):
            flags[(addr, "louvain_community")] = {
                "signal_value": community_id,
                "evidence": {"community_id": community_id, "community_size": size},
            }
    return flags


# ── orchestration ─────────────────────────────────────────────────────────────

def run_detection(
    con,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    rules_cfg,
    detection_cfg: dict[str, Any],
    known_addresses: pd.DataFrame | None = None,
) -> dict[str, Any]:
    # Sorted insertion keeps NetworkX/Louvain results reproducible when the
    # warehouse returns rows in a different order between runs.
    graph = build_graph(con, sorted(features["address"].tolist()))

    dbscan = detection_cfg.get("dbscan", {})
    dbscan_labels, dbscan_summary = run_dbscan(
        features,
        eps=float(dbscan.get("eps", 3.195323)),
        min_samples=int(dbscan.get("min_samples", 5)),
        min_tx_total=int(dbscan.get("min_tx_total", 3)),
    )

    cycle_params = next(r.params for r in rules_cfg.rules if r.id == "R1")
    cycle_flags, cycle_stats = detect_cycles(
        graph,
        min_length=int(cycle_params.get("cycle_length_min", 3)),
        max_length=int(cycle_params.get("cycle_length_max", 5)),
        value_tolerance=float(cycle_params.get("value_tolerance", 0.05)),
        window_hours=float(cycle_params.get("time_window_hours", 24)),
        per_scc_cap=int(detection_cfg.get("cycles", {}).get("per_scc_cap", 1000)),
        enum_cap=int(detection_cfg.get("cycles", {}).get("enum_cap", 5000)),
        max_scc_size=int(cycle_params.get("max_scc_size", 5000)),
    )

    r2_params = next(r.params for r in rules_cfg.rules if r.id == "R2")
    fanout_flags = detect_fanout(
        features,
        graph=graph,
        mode=str(r2_params.get("mode", "outbound_fanout")),
        threshold=int(r2_params.get("fanout_threshold", 20)),
        min_outbound=int(r2_params.get("min_outbound", 20)),
        min_ratio=float(r2_params.get("min_ratio", 0.5)),
        require_temporal_ordering=bool(r2_params.get("require_temporal_ordering", True)),
        gather_window_hours=float(r2_params.get("gather_window_hours", 72)),
        min_gather_fraction=float(r2_params.get("min_gather_fraction", 0.1)),
        min_gather_intermediaries=int(r2_params.get("min_gather_intermediaries", 2)),
        min_collectors=int(r2_params.get("min_collectors", 1)),
        deck_min_out_in_ratio=float(r2_params.get("deck_min_out_in_ratio", 5.0)),
        deck_min_counterparties=int(r2_params.get("deck_min_counterparties", 50)),
    )

    r3_params = next(r.params for r in rules_cfg.rules if r.id == "R3")
    sanctions = load_sanctions(str(r3_params.get("sanctions_path", "config/sanctions.json")))
    transactions = con.execute("""
        SELECT from_address, to_address, value_eth
        FROM transactions
    """).df()
    sanctioned_flags = detect_sanctioned_1hop(
        graph, labels, splits,
        sanctions=sanctions,
        transactions=transactions,
        min_value_eth=float(r3_params.get("min_value_eth", 0.01)),
        min_direct_strength=float(r3_params.get("min_direct_strength", 0.1)),
    )

    r4_params = next(r.params for r in rules_cfg.rules if r.id == "R4")
    pagerank_cfg = detection_cfg.get("pagerank", {})
    excluded_categories = {
        str(category).strip().lower()
        for category in r4_params.get("exclude_known_categories", [])
    }
    excluded_pagerank = set()
    if known_addresses is not None and not known_addresses.empty and excluded_categories:
        excluded_pagerank = {
            str(row.address).strip().lower()
            for row in known_addresses.itertuples(index=False)
            if str(getattr(row, "category", "")).strip().lower() in excluded_categories
        }
    pagerank_flags, pagerank_scores = compute_pagerank(
        graph,
        percentile=float(r4_params.get("percentile", 95)),
        alpha=float(r4_params.get("alpha", pagerank_cfg.get("alpha", 0.85))),
        weight=str(r4_params.get("weight", "total_value_eth")),
        zero_floor=float(r4_params.get("zero_weight_floor", 1e-12)),
        continuous=bool(r4_params.get("continuous", True)),
        min_percentile=float(r4_params.get("min_percentile", 90)),
        min_strength=float(r4_params.get("min_strength", 0.05)),
        exclude_addresses=excluded_pagerank,
    )

    louvain_cfg = detection_cfg.get("louvain", {})
    louvain_flags = detect_louvain(
        graph,
        seed=int(louvain_cfg.get("seed", 42)),
        weight=str(louvain_cfg.get("weight", "total_value_eth")),
        resolution=float(louvain_cfg.get("resolution", 1.0)),
    )

    r5_params = next(r.params for r in rules_cfg.rules if r.id == "R5")
    apriori_flags, apriori_rules, apriori_status = mine_apriori(
        features, pagerank_flags, dbscan_labels, labels, splits,
        min_support=float(r5_params.get("min_support", 0.05)),
        min_lift=float(r5_params.get("min_lift", 2.0)),
        min_confidence=float(r5_params.get("min_confidence", 0.8)),
        max_antecedent_size=int(r5_params.get("max_antecedent_size", 3)),
    )

    dbscan_flags: dict[tuple[str, str], dict[str, Any]] = {}
    for address, cluster_id in dbscan_labels.items():
        dbscan_flags[(address, "dbscan_cluster")] = {
            "signal_value": int(cluster_id),
            "evidence": {
                "cluster_id": int(cluster_id),
                "is_noise": cluster_id == -1,
                "eps": dbscan_summary.get("eps"),
                "min_samples": dbscan_summary.get("min_samples"),
            },
        }
        if cluster_id == -1:
            dbscan_flags[(address, "dbscan_noise")] = {
                "signal_value": 1,
                "evidence": {"reason": "DBSCAN noise; not a standalone verdict"},
            }

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for group in (dbscan_flags, cycle_flags, fanout_flags, sanctioned_flags,
                  pagerank_flags, louvain_flags, apriori_flags):
        flags.update(group)

    cycle_lengths: dict[str, int] = defaultdict(int)
    for payload in cycle_flags.values():
        cycle_lengths[str(int(payload.get("signal_value", 0)))] += 1
    sanctioned_strengths = [
        float(payload.get("strength", 1.0)) for payload in sanctioned_flags.values()
    ]
    pattern_stats = {
        "cycle_members": len(cycle_flags),
        "cycle_members_by_length": dict(sorted(cycle_lengths.items())),
        "fanout_members": len(fanout_flags),
        "sanctioned_members": len(sanctioned_flags),
        "sanctioned_mean_strength": (
            round(float(np.mean(sanctioned_strengths)), 6) if sanctioned_strengths else None
        ),
        "sanctions_seed_count": len(sanctions),
        "pagerank_members": len(pagerank_flags),
        "pagerank_top5_members": sum(
            1 for payload in pagerank_flags.values()
            if bool((payload.get("evidence") or {}).get("top5"))
        ),
        "pagerank_known_excluded": len(excluded_pagerank),
        "apriori_members": len(apriori_flags),
    }

    return {
        "flags": flags,
        "graph": graph,
        "dbscan_summary": dbscan_summary,
        "cycle_summary": cycle_stats,
        "apriori_rules": apriori_rules,
        "apriori_status": apriori_status,
        "pagerank_scores": pagerank_scores,
        "louvain_communities": len({v["signal_value"] for k, v in louvain_flags.items()}),
        "pattern_stats": pattern_stats,
    }


# ── serialisation ─────────────────────────────────────────────────────────────

def flags_to_frame(flags: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "address": address,
            "signal_name": signal,
            "signal_value": payload.get("signal_value", 0),
            "evidence": json.dumps(payload.get("evidence", {}), default=str),
        }
        for (address, signal), payload in sorted(flags.items())
    ]
    return pd.DataFrame(rows, columns=["address", "signal_name", "signal_value", "evidence"])


def flags_from_frame(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        payload: dict[str, Any] = {"signal_value": row.signal_value}
        if isinstance(row.evidence, str) and row.evidence:
            try:
                payload["evidence"] = json.loads(row.evidence)
            except json.JSONDecodeError:
                payload["evidence"] = {}
        else:
            payload["evidence"] = row.evidence or {}
        flags[(row.address, row.signal_name)] = payload
    return flags


def graph_edges_frame(graph: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {
            "from_address": u,
            "to_address": v,
            "tx_count": int(data.get("tx_count", 0)),
            "total_value_eth": float(data.get("total_value_eth", 0.0)),
            "first_tx_ts": data.get("first_tx_ts"),
            "last_tx_ts": data.get("last_tx_ts"),
        }
        for u, v, data in graph.edges(data=True)
    ]
    return pd.DataFrame(rows)
