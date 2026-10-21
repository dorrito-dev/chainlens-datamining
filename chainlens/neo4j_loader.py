"""Idempotent curated-subgraph loader for the Neo4j showcase layer."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.graph import build_curated_subgraph, build_graph


def _chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def load_curated_subgraph(
    engine: Engine,
    driver,
    graph=None,
    max_nodes: int | None = None,
    hops: int | None = None,
    batch_size: int = 1000,
) -> dict:
    """Rebuild the ChainLens-tagged showcase subgraph.

    Only nodes tagged ``:ChainLens`` are removed, so the loader is re-runnable
    without wiping unrelated Neo4j data.
    """
    graph = graph or build_graph(engine)
    curated = build_curated_subgraph(
        graph, engine, max_nodes=max_nodes, hops=hops
    )
    addresses = curated["addresses"]
    if not addresses:
        raise RuntimeError("Curated subgraph is empty; run `make detect` first")

    with engine.connect() as conn:
        metadata_rows = conn.execute(
            text(
                """
                SELECT
                    rs.address,
                    rs.score,
                    rs.tier,
                    (l.address IS NOT NULL) AS is_labeled
                FROM risk_scores rs
                LEFT JOIN (SELECT DISTINCT address FROM labels) l
                    ON l.address = rs.address
                WHERE rs.address = ANY(:addresses)
                """
            ),
            {"addresses": addresses},
        ).all()
    metadata = {
        row.address: {
            "address": row.address,
            "score": int(row.score or 0),
            "tier": row.tier or "unscored",
            "is_labeled": bool(row.is_labeled),
        }
        for row in metadata_rows
    }
    nodes = [
        metadata.get(
            address,
            {
                "address": address,
                "score": 0,
                "tier": "unscored",
                "is_labeled": False,
            },
        )
        for address in addresses
    ]
    relationships = []
    for edge in curated["edges"]:
        relationships.append(
            {
                "from": edge["from"],
                "to": edge["to"],
                "pair": f"{edge['from']}|{edge['to']}",
                "tx_count": int(edge["tx_count"]),
                "total_value_eth": float(edge["total_value_eth"]),
            }
        )

    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address) "
            "REQUIRE a.address IS UNIQUE"
        )
        session.run("MATCH (a:ChainLens) DETACH DELETE a")
        for chunk in _chunks(nodes, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (a:Address:ChainLens {address: row.address})
                SET a.score = row.score,
                    a.tier = row.tier,
                    a.is_labeled = row.is_labeled
                """,
                rows=chunk,
            )
        for chunk in _chunks(relationships, batch_size):
            session.run(
                """
                UNWIND $rows AS rel
                MATCH (a:Address:ChainLens {address: rel.from})
                MATCH (b:Address:ChainLens {address: rel.to})
                MERGE (a)-[r:SENT {pair: rel.pair}]->(b)
                SET r.tx_count = rel.tx_count,
                    r.total_value_eth = rel.total_value_eth
                """,
                rows=chunk,
            )
        node_count = session.run(
            "MATCH (a:ChainLens) RETURN count(a) AS n"
        ).single()["n"]
        relationship_count = session.run(
            "MATCH (:ChainLens)-[r:SENT]->(:ChainLens) RETURN count(r) AS n"
        ).single()["n"]

    print(
        f"Neo4j curated subgraph: {node_count} nodes, "
        f"{relationship_count} relationships, cap_hit={curated['cap_hit']}"
    )
    return {
        "nodes": int(node_count),
        "relationships": int(relationship_count),
        "cap_hit": bool(curated["cap_hit"]),
        "node_cap": curated["node_cap"],
        "hops": curated["hops"],
    }


def run_graph_load(engine: Engine, driver) -> dict:
    return load_curated_subgraph(engine, driver)


def reset_chainlens_graph(driver) -> dict:
    with driver.session() as session:
        session.run("MATCH (a:ChainLens) DETACH DELETE a")
        remaining = session.run(
            "MATCH (a:ChainLens) RETURN count(a) AS n"
        ).single()["n"]
    print(f"Neo4j graph reset: {remaining} :ChainLens node(s) remain")
    return {"remaining": int(remaining)}
