"""Stage 4: pattern detection.

NetworkX graph (R1/R3/R4), DBSCAN (outlier flag), Apriori (R5, trained on the
train split only), and Louvain (display-only communities).

Order matters: DBSCAN and PageRank run before Apriori because R5 uses
`dbscan_noise` and `high_pagerank` as items.
"""

from __future__ import annotations

import json
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

def detect_cycles(
    graph: nx.DiGraph,
    *,
    min_length: int = 3,
    max_length: int = 5,
    value_tolerance: float = 0.05,
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
        try:
            for cycle in islice(nx.simple_cycles(sub, length_bound=max_length), enum_cap):
                if min_length <= len(cycle) <= max_length:
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

def detect_fanout(
    features: pd.DataFrame,
    *,
    mode: str = "outbound_fanout",
    threshold: int = 20,
    min_outbound: int = 20,
    min_ratio: float = 0.5,
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
        flags[(row.address, "fanout")] = {
            "signal_value": int(row.unique_counterparties),
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
            },
        }
    return flags


# ── R3 ────────────────────────────────────────────────────────────────────────

def detect_sanctioned_1hop(
    graph: nx.DiGraph,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    train_addrs = set(splits.loc[splits["split"] == "train", "address"])
    sanctioned = set(labels.loc[labels["label"] == "sanctioned", "address"])
    seeds = sorted(sanctioned & train_addrs)

    matches: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if seed not in graph:
            continue
        for direction, neighbours in (
            ("out", list(graph.successors(seed))),
            ("in", list(graph.predecessors(seed))),
        ):
            for neighbour in neighbours:
                if neighbour == seed or neighbour in seeds:
                    continue
                entry = matches.setdefault(neighbour, {"seeds": set(), "directions": set()})
                entry["seeds"].add(seed)
                entry["directions"].add(direction)

    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for addr in sorted(matches):
        entry = matches[addr]
        directions = sorted(entry["directions"])
        flags[(addr, "sanctioned_1hop")] = {
            "signal_value": 1,
            "evidence": {
                "seed": sorted(entry["seeds"])[0],
                "direction": directions[0] if len(directions) == 1 else "both",
                "seeds": sorted(entry["seeds"]),
                "directions": directions,
            },
        }
    return flags


# ── R4 ────────────────────────────────────────────────────────────────────────

def compute_pagerank(
    graph: nx.DiGraph,
    *,
    percentile: float = 95,
    alpha: float = 0.85,
    weight: str = "total_value_eth",
    zero_floor: float = 1e-12,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    def edge_weight(_u: str, _v: str, data: dict[str, Any]) -> float:
        return max(float(data.get(weight, 0.0)), zero_floor)

    scores = nx.pagerank(
        graph, alpha=alpha, weight=edge_weight, max_iter=100, tol=1e-6,
    )
    if not scores:
        return {}, {}
    threshold = float(np.percentile(list(scores.values()), percentile))
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    for addr in sorted(graph.nodes):
        score = float(scores[addr])
        if score >= threshold:
            flags[(addr, "pagerank_top5")] = {
                "signal_value": 1,
                "evidence": {"pagerank": score, "threshold": threshold},
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
    pagerank_addresses = {key[0] for key, value in pagerank_flags.items()
                          if key[1] == "pagerank_top5" and value.get("signal_value") == 1}
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
        mode=str(r2_params.get("mode", "outbound_fanout")),
        threshold=int(r2_params.get("fanout_threshold", 20)),
        min_outbound=int(r2_params.get("min_outbound", 20)),
        min_ratio=float(r2_params.get("min_ratio", 0.5)),
        deck_min_out_in_ratio=float(r2_params.get("deck_min_out_in_ratio", 5.0)),
        deck_min_counterparties=int(r2_params.get("deck_min_counterparties", 50)),
    )

    sanctioned_flags = detect_sanctioned_1hop(graph, labels, splits)

    r4_params = next(r.params for r in rules_cfg.rules if r.id == "R4")
    pagerank_cfg = detection_cfg.get("pagerank", {})
    pagerank_flags, pagerank_scores = compute_pagerank(
        graph,
        percentile=float(r4_params.get("percentile", 95)),
        alpha=float(r4_params.get("alpha", pagerank_cfg.get("alpha", 0.85))),
        weight=str(r4_params.get("weight", "total_value_eth")),
        zero_floor=float(r4_params.get("zero_weight_floor", 1e-12)),
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

    return {
        "flags": flags,
        "graph": graph,
        "dbscan_summary": dbscan_summary,
        "cycle_summary": cycle_stats,
        "apriori_rules": apriori_rules,
        "apriori_status": apriori_status,
        "pagerank_scores": pagerank_scores,
        "louvain_communities": len({v["signal_value"] for k, v in louvain_flags.items()}),
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
