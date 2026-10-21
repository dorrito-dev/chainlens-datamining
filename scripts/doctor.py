#!/usr/bin/env python3
"""Pre-flight environment doctor.

Checks that the same Neo4j secret is used by the driver and the container, then
verifies both databases answer a trivial query. Safe to run before any stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from chainlens.db import get_neo4j_driver, get_pg_engine  # noqa: E402

REQUIRED = [
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_env() -> None:
    missing = [name for name in REQUIRED if not os.getenv(name)]
    if missing:
        fail("missing environment variables: " + ", ".join(missing))
    print("PASS: required environment variables are present")


def check_neo4j_secret() -> None:
    password = os.environ["NEO4J_PASSWORD"]
    expected = f"neo4j/{password}"
    auth = os.getenv("NEO4J_AUTH")
    if auth:
        if auth != expected:
            fail(
                "NEO4J_AUTH does not match neo4j/NEO4J_PASSWORD "
                "(the driver and container would use different secrets)"
            )
        print("PASS: NEO4J_AUTH and NEO4J_PASSWORD agree")
        return

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["neo4j"].get("environment", {})
    declared = env.get("NEO4J_AUTH") if isinstance(env, dict) else None
    if declared != "neo4j/${NEO4J_PASSWORD}":
        fail(
            "docker-compose.yml must declare NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}; "
            f"found {declared!r}"
        )
    print("PASS: docker-compose.yml derives NEO4J_AUTH from NEO4J_PASSWORD")


def check_postgres() -> None:
    try:
        engine = get_pg_engine()
        with engine.connect() as conn:
            value = conn.execute(text("SELECT 1")).scalar_one()
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        fail(f"Postgres is unreachable: {exc}")
    if value != 1:
        fail("Postgres SELECT 1 returned an unexpected value")
    print("PASS: Postgres SELECT 1")


def check_neo4j() -> None:
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            value = session.run("RETURN 1 AS one").single()["one"]
        driver.close()
    except Exception as exc:  # noqa: BLE001
        fail(f"Neo4j is unreachable or credentials are wrong: {exc}")
    if value != 1:
        fail("Neo4j RETURN 1 returned an unexpected value")
    print("PASS: Neo4j RETURN 1")


def main() -> int:
    check_env()
    check_neo4j_secret()
    check_postgres()
    check_neo4j()
    print("DOCTOR OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
