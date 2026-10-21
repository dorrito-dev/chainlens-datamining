"""ChainLens database helpers.

This module is import-safe: no connection is opened at import time.
Plain SQL DDL + pandas ``to_sql`` is intentional; there are no ORM models.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_pg_engine() -> Engine:
    """Read POSTGRES_* from env; return engine with pool_pre_ping=True."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "chainlens"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "chainlens"),
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def get_neo4j_driver():
    """Read NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.

    The same code path works for ``bolt://`` local Docker and ``neo4j+s://``
    Aura Free. Connectivity is verified immediately with a clear error message.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
        driver.close()
        raise RuntimeError(
            f"Cannot reach Neo4j at {uri!r} as user {user!r}. "
            "Check that the neo4j container is healthy (`make up`), that "
            "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD are set in .env, and that "
            "the database is reachable."
        ) from exc
    return driver


def _resolve_schema_path(schema_path: str) -> Path:
    path = Path(schema_path)
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate
    return path


def apply_schema(engine: Engine, schema_path: str = "sql/schema.sql") -> None:
    """Execute schema.sql. Idempotent (uses CREATE TABLE IF NOT EXISTS)."""
    path = _resolve_schema_path(schema_path)
    ddl = path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)


def write_df(
    engine: Engine,
    df: pd.DataFrame,
    table: str,
    if_exists: str = "replace",
) -> int:
    """Write a DataFrame with ``to_sql``; lowercase address columns first."""
    out = df.copy()
    for column in out.columns:
        if "address" in column.lower() and out[column].dtype == object:
            mask = out[column].notna()
            out.loc[mask, column] = out.loc[mask, column].astype(str).str.lower()
    out.to_sql(table, engine, if_exists=if_exists, index=False)
    return len(out)


def table_exists(engine: Engine, table: str) -> bool:
    """Check information_schema.tables (current schema)."""
    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = :table_name
        )
        """
    )
    with engine.connect() as conn:
        return bool(conn.execute(query, {"table_name": table}).scalar())
