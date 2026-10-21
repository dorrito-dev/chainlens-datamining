#!/usr/bin/env python3
"""Load the curated ChainLens subgraph into Neo4j."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.db import apply_schema, get_neo4j_driver, get_pg_engine  # noqa: E402
from chainlens.neo4j_loader import run_graph_load  # noqa: E402


def main() -> int:
    engine = get_pg_engine()
    apply_schema(engine)
    driver = get_neo4j_driver()
    try:
        run_graph_load(engine, driver)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
