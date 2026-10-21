from __future__ import annotations

import neo4j

from chainlens.db import get_neo4j_driver
from chainlens.graph import build_graph
from chainlens.neo4j_loader import load_curated_subgraph


def test_uri_schemes_are_passed_through_unchanged(monkeypatch):
    captured: list[str] = []

    class FakeDriver:
        def verify_connectivity(self):
            return None

        def close(self):
            return None

    def fake_driver(uri, auth):  # noqa: ANN001
        captured.append(uri)
        return FakeDriver()

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", fake_driver)
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    for uri in ("bolt://localhost:7687", "neo4j+s://example.databases.neo4j.io"):
        monkeypatch.setenv("NEO4J_URI", uri)
        driver = get_neo4j_driver()
        driver.close()
    assert captured == [
        "bolt://localhost:7687",
        "neo4j+s://example.databases.neo4j.io",
    ]


def test_curated_loader_is_idempotent_and_capped(pg_engine, neo4j_driver):
    graph = build_graph(pg_engine)
    first = load_curated_subgraph(pg_engine, neo4j_driver, graph=graph)
    second = load_curated_subgraph(pg_engine, neo4j_driver, graph=graph)

    assert first["nodes"] <= 2_000
    assert first["nodes"] == second["nodes"]
    assert first["relationships"] == second["relationships"]

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (a:ChainLens)-[r:SENT]->(b:ChainLens)
            RETURN
                count(r) AS relationships,
                count(DISTINCT a.address + '|' + b.address) AS pairs
            """
        ).single()
    assert int(result["relationships"]) == int(result["pairs"])
    assert int(result["relationships"]) == first["relationships"]
