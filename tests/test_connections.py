from __future__ import annotations

from sqlalchemy import text


def test_postgres_select_one(pg_engine):
    with pg_engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_neo4j_return_one(neo4j_driver):
    with neo4j_driver.session() as session:
        assert session.run("RETURN 1 AS one").single()["one"] == 1
