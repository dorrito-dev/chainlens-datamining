"""NetworkX graph construction and Phase 4 graph signals."""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import islice
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import load_data_config, project_path
from chainlens.detection import (
    GRAPH_SIGNALS,
    load_detection_config,
    rule_by_id,
    write_flags,
)
from chainlens.fixtures import address_for


def build_graph(
    engine: Engine,
    parquet_path: str | Path | None = None,
    scaler_version: str | None = None,
) -> nx.DiGraph:
    """Build an aggregated directed graph from the clean Parquet."""
    config = load_data_config()
    path = Path(parquet_path) if parquet_path else project_path(
        config["ingest"]["clean_parquet"]
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Phase 2 ingest first")

    with engine.connect() as conn:
        addresses = list(
            conn.execute(
                text("SELECT address FROM address_features ORDER BY address")
            ).scalars()
        )
    if not addresses:
        raise RuntimeError("address_features is empty; run `make features` first")

    con = duckdb.connect()
    try:
        edges = con.execute(
            """
            SELECT
                lower(from_address) AS from_address,
                lower(to_address) AS to_address,
                COUNT(*)::BIGINT AS tx_count,
                SUM(value_eth) AS total_value_eth,
                MIN(block_timestamp) AS first_tx_ts,
                MAX(block_timestamp) AS last_tx_ts
            FROM read_parquet(?)
            WHERE from_address IS NOT NULL
              AND to_address IS NOT NULL
              AND from_address <> to_address
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [str(path)],
        ).df()
    finally:
        con.close()

    graph = nx.DiGraph()
    graph.add_nodes_from(addresses)
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.from_address,
            row.to_address,
            tx_count=int(row.tx_count),
            total_value_eth=float(row.total_value_eth),
            first_tx_ts=row.first_tx_ts,
            last_tx_ts=row.last_tx_ts,
        )
    graph.graph["parquet_path"] = str(path)
    graph.graph["scaler_version"] = scaler_version or ""
    print(f"Graph built: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


def _pagerank_flags(graph: nx.DiGraph) -> tuple[dict[str, float], list[dict], float, int]:
    params = rule_by_id("R4")["params"]
    percentile = float(params.get("percentile", 95))
    scores = nx.pagerank(
        graph,
        alpha=0.85,
        weight="total_value_eth",
        max_iter=100,
        tol=1e-6,
    )
    threshold = float(np.percentile(list(scores.values()), percentile))
    rows: list[dict] = []
    top_count = 0
    for address in sorted(graph.nodes):
        score = float(scores[address])
        rows.append(
            {
                "address": address,
                "signal_name": "pagerank",
                "signal_value": score,
                "evidence": {"pagerank": score},
            }
        )
        if score >= threshold:
            top_count += 1
            rows.append(
                {
                    "address": address,
                    "signal_name": "pagerank_top5",
                    "signal_value": 1,
                    "evidence": {"pagerank": score, "threshold": threshold},
                }
            )
    return scores, rows, threshold, top_count


def _qualifying_cycles(
    graph: nx.DiGraph,
) -> tuple[list[dict], int, int]:
    """SCC-first cycle search with value similarity and time-window checks."""
    params = rule_by_id("R1")["params"]
    min_length = int(params.get("cycle_length_min", 3))
    max_length = int(params.get("cycle_length_max", 5))
    tolerance = float(params.get("value_tolerance", 0.05))
    window_hours = float(params.get("time_window_hours", 24))
    graph_config = load_detection_config()["graph"]
    per_scc_cap = int(graph_config.get("max_cycles_per_scc", 1000))
    enum_cap = int(graph_config.get("cycle_enumeration_cap", 5000))

    qualifying: list[dict] = []
    sccs = [
        set(component)
        for component in nx.strongly_connected_components(graph)
        if len(component) >= min_length
    ]
    sccs.sort(key=lambda component: sorted(component))
    capped_sccs = 0

    for scc in sccs:
        subgraph = graph.subgraph(scc)
        found: list[tuple[str, ...]] = []
        for cycle in islice(
            nx.simple_cycles(subgraph, length_bound=max_length), enum_cap
        ):
            if min_length <= len(cycle) <= max_length:
                found.append(tuple(cycle))
        found.sort(key=lambda cycle: (len(cycle), tuple(sorted(cycle)), cycle))

        scc_qualifying: list[dict] = []
        for cycle in found:
            edges = [
                (cycle[index], cycle[(index + 1) % len(cycle)])
                for index in range(len(cycle))
            ]
            values = [
                float(graph.edges[from_address, to_address]["total_value_eth"])
                for from_address, to_address in edges
            ]
            if not values or min(values) <= 0:
                continue
            if max(values) / min(values) > 1 + tolerance:
                continue
            first_times = [
                graph.edges[from_address, to_address]["first_tx_ts"]
                for from_address, to_address in edges
            ]
            last_times = [
                graph.edges[from_address, to_address]["last_tx_ts"]
                for from_address, to_address in edges
            ]
            first_tx = min(first_times)
            last_tx = max(last_times)
            span_hours = (last_tx - first_tx).total_seconds() / 3600
            if span_hours > window_hours:
                continue

            edge_pairs = sorted(
                (
                    {
                        "from": from_address,
                        "to": to_address,
                        "value": float(
                            graph.edges[from_address, to_address]["total_value_eth"]
                        ),
                    }
                    for from_address, to_address in edges
                ),
                key=lambda edge: (edge["from"], edge["to"]),
            )
            scc_qualifying.append(
                {
                    "cycle_path": sorted(cycle),
                    "cycle_length": len(cycle),
                    "edge_values": [edge["value"] for edge in edge_pairs],
                    "edge_pairs": edge_pairs,
                    "max_min_ratio": round(max(values) / min(values), 6),
                    "time_span_hours": round(span_hours, 6),
                    "first_tx_ts": str(first_tx),
                    "last_tx_ts": str(last_tx),
                    "scc_size": len(scc),
                }
            )
            if len(scc_qualifying) >= per_scc_cap:
                capped_sccs += 1
                break

        qualifying.extend(scc_qualifying)

    qualifying.sort(
        key=lambda cycle: (
            cycle["time_span_hours"],
            tuple(cycle["cycle_path"]),
            tuple((edge["from"], edge["to"]) for edge in cycle["edge_pairs"]),
        )
    )
    return qualifying, len(sccs), capped_sccs


def _cycle_flags(cycles: list[dict]) -> list[dict]:
    by_address: dict[str, list[dict]] = defaultdict(list)
    for cycle in cycles:
        for address in cycle["cycle_path"]:
            by_address[address].append(cycle)

    rows: list[dict] = []
    for address in sorted(by_address):
        address_cycles = by_address[address]
        primary = address_cycles[0]
        evidence = dict(primary)
        evidence["cycle_count"] = len(address_cycles)
        evidence["cycles"] = address_cycles[:20]
        rows.append(
            {
                "address": address,
                "signal_name": "cycle_member",
                "signal_value": int(primary["cycle_length"]),
                "evidence": evidence,
            }
        )
    return rows


def _fanout_flags(engine: Engine) -> list[dict]:
    params = rule_by_id("R2")["params"]
    threshold = int(params.get("fanout_threshold", 20))
    min_outbound = int(params.get("min_outbound", 20))
    min_ratio = float(params.get("min_ratio", 0.5))
    query = text(
        """
        SELECT address, unique_counterparties, tx_count_out, fanout_ratio
        FROM address_features
        WHERE unique_counterparties >= :threshold
          AND tx_count_out >= :min_outbound
          AND fanout_ratio >= :min_ratio
        ORDER BY address
        """
    )
    with engine.connect() as conn:
        frame = pd.read_sql(
            query,
            conn,
            params={
                "threshold": threshold,
                "min_outbound": min_outbound,
                "min_ratio": min_ratio,
            },
        )
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "address": row.address,
                "signal_name": "fanout",
                "signal_value": int(row.unique_counterparties),
                "evidence": {
                    "unique_counterparties": int(row.unique_counterparties),
                    "tx_count_out": int(row.tx_count_out),
                    "fanout_ratio": float(row.fanout_ratio),
                },
            }
        )
    return rows


def _sanctioned_seeds(engine: Engine) -> dict[str, set[str]]:
    query = text(
        """
        SELECT l.address, l.source
        FROM labels l
        JOIN label_splits s ON s.address = l.address
        WHERE l.label = 'sanctioned' AND s.split = 'train'
        ORDER BY l.address, l.source
        """
    )
    seeds: dict[str, set[str]] = defaultdict(set)
    with engine.connect() as conn:
        for address, source in conn.execute(query):
            seeds[address].add(source)
    return dict(seeds)


def _sanctioned_flags(graph: nx.DiGraph, engine: Engine) -> tuple[list[dict], int]:
    seeds = _sanctioned_seeds(engine)
    seed_set = set(seeds)
    matches: dict[str, dict] = {}
    for seed in sorted(seed_set):
        if seed not in graph:
            continue
        directions = {
            "out": list(graph.successors(seed)),
            "in": list(graph.predecessors(seed)),
        }
        for direction, neighbours in directions.items():
            for neighbour in neighbours:
                if neighbour == seed or neighbour in seed_set:
                    continue
                entry = matches.setdefault(
                    neighbour,
                    {
                        "seeds": set(),
                        "sources": set(),
                        "directions": set(),
                    },
                )
                entry["seeds"].add(seed)
                entry["sources"].update(seeds[seed])
                entry["directions"].add(direction)

    rows = []
    for neighbour in sorted(matches):
        entry = matches[neighbour]
        ordered_seeds = sorted(entry["seeds"])
        ordered_directions = sorted(entry["directions"])
        rows.append(
            {
                "address": neighbour,
                "signal_name": "sanctioned_1hop",
                "signal_value": 1,
                "evidence": {
                    "seed": ordered_seeds[0],
                    "direction": (
                        ordered_directions[0]
                        if len(ordered_directions) == 1
                        else "both"
                    ),
                    "seed_source": sorted(entry["sources"])[0],
                    "seeds": ordered_seeds,
                    "directions": ordered_directions,
                },
            }
        )
    return rows, len(seed_set)


def _louvain_flags(graph: nx.DiGraph) -> tuple[list[dict], int]:
    communities = nx.community.louvain_communities(
        graph.to_undirected(), weight="total_value_eth", seed=42
    )
    ordered = sorted(communities, key=lambda community: sorted(community))
    rows = []
    for community_id, community in enumerate(ordered):
        for address in sorted(community):
            rows.append(
                {
                    "address": address,
                    "signal_name": "louvain_community",
                    "signal_value": community_id,
                    "evidence": {"community_size": len(community)},
                }
            )
    return rows, len(ordered)


def run_graph(engine: Engine) -> dict:
    """Compute PageRank, cycles, fan-out, sanctioned 1-hop and Louvain."""
    graph = build_graph(engine)
    pagerank, pagerank_rows, threshold, top_count = _pagerank_flags(graph)
    cycles, scc_count, capped_sccs = _qualifying_cycles(graph)
    cycle_rows = _cycle_flags(cycles)
    fanout_rows = _fanout_flags(engine)
    hop_rows, seed_count = _sanctioned_flags(graph, engine)
    louvain_rows, community_count = _louvain_flags(graph)

    rows = pagerank_rows + cycle_rows + fanout_rows + hop_rows + louvain_rows
    written = write_flags(engine, rows, reset_signals=GRAPH_SIGNALS)

    ranked = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
    exchange = address_for("EXCHANGE_HOT")
    exchange_rank = next(
        (index for index, (address, _) in enumerate(ranked, 1) if address == exchange),
        None,
    )
    print(
        f"PageRank: top-5%={top_count}, threshold={threshold:.8f}, "
        f"EXCHANGE_HOT rank={exchange_rank}"
    )
    print(
        f"Cycles: {len(cycles)} qualifying, {len(cycle_rows)} addresses, "
        f"SCCs>=3={scc_count}, capped SCCs={capped_sccs}"
    )
    print(
        f"Fan-out flags: {len(fanout_rows)}; sanctioned 1-hop: {len(hop_rows)} "
        f"from {seed_count} train seeds"
    )
    print(f"Louvain communities: {community_count}")
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "pagerank_threshold": threshold,
        "pagerank_top5": top_count,
        "exchange_rank": exchange_rank,
        "cycles": len(cycles),
        "cycle_addresses": len(cycle_rows),
        "fanout": len(fanout_rows),
        "sanctioned_1hop": len(hop_rows),
        "sanctioned_seeds": seed_count,
        "louvain_communities": community_count,
        "flags_written": written,
    }


def _bfs_nodes(
    graph: nx.DiGraph,
    seeds: list[str],
    hops: int,
    max_nodes: int,
) -> tuple[list[str], bool]:
    selected: list[str] = []
    seen: set[str] = set()
    cap_hit = False
    for seed in seeds:
        if seed not in graph or seed in seen:
            continue
        queue: deque[tuple[str, int]] = deque([(seed, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in seen:
                continue
            if len(selected) >= max_nodes:
                cap_hit = True
                return selected, cap_hit
            seen.add(node)
            selected.append(node)
            if depth >= hops:
                continue
            neighbours = set(graph.successors(node)) | set(graph.predecessors(node))
            for neighbour in sorted(neighbours):
                if neighbour not in seen:
                    queue.append((neighbour, depth + 1))
    return selected, cap_hit


def _edge_rows(graph: nx.DiGraph, nodes: set[str]) -> list[dict]:
    edges = []
    for from_address, to_address, data in graph.edges(data=True):
        if from_address in nodes and to_address in nodes:
            edges.append(
                {
                    "from": from_address,
                    "to": to_address,
                    "tx_count": int(data["tx_count"]),
                    "total_value_eth": float(data["total_value_eth"]),
                }
            )
    edges.sort(key=lambda edge: (edge["from"], edge["to"]))
    return edges


def curated_seed_addresses(engine: Engine, top_n: int | None = None) -> list[str]:
    config = load_detection_config()["graph"]
    top_n = int(top_n or config.get("curated_top_n", 200))
    with engine.connect() as conn:
        top = list(
            conn.execute(
                text(
                    "SELECT address FROM risk_scores "
                    "ORDER BY score DESC, address ASC LIMIT :limit"
                ),
                {"limit": top_n},
            ).scalars()
        )
        labeled = list(
            conn.execute(text("SELECT DISTINCT address FROM labels")).scalars()
        )
        flagged = list(
            conn.execute(
                text(
                    "SELECT DISTINCT address FROM address_flags "
                    "WHERE signal_name IN "
                    "('sanctioned_1hop', 'cycle_member', 'fanout')"
                )
            ).scalars()
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for address in top + sorted(labeled) + sorted(flagged):
        if address not in seen:
            seen.add(address)
            ordered.append(address)
    return ordered


def build_curated_subgraph(
    graph: nx.DiGraph,
    engine: Engine,
    max_nodes: int | None = None,
    hops: int | None = None,
) -> dict:
    config = load_detection_config()["graph"]
    max_nodes = int(max_nodes or config.get("curated_max_nodes", 2000))
    hops = int(hops or config.get("curated_hops", 2))
    seeds = curated_seed_addresses(engine)
    nodes, cap_hit = _bfs_nodes(graph, seeds, hops=hops, max_nodes=max_nodes)
    node_set = set(nodes)
    edges = _edge_rows(graph, node_set)
    return {
        "addresses": nodes,
        "edges": edges,
        "cap_hit": cap_hit,
        "node_cap": max_nodes,
        "hops": hops,
    }


def neighborhood_subgraph(
    graph: nx.DiGraph,
    address: str,
    hops: int = 2,
    max_nodes: int = 200,
) -> dict:
    nodes, cap_hit = _bfs_nodes(graph, [address], hops=hops, max_nodes=max_nodes)
    node_set = set(nodes)
    return {
        "addresses": nodes,
        "edges": _edge_rows(graph, node_set),
        "cap_hit": cap_hit,
        "node_cap": max_nodes,
        "hops": hops,
    }
