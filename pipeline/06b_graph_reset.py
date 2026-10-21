#!/usr/bin/env python3
"""Delete only the :ChainLens showcase subgraph from Neo4j."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.db import get_neo4j_driver  # noqa: E402
from chainlens.neo4j_loader import reset_chainlens_graph  # noqa: E402


def main() -> int:
    driver = get_neo4j_driver()
    try:
        reset_chainlens_graph(driver)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
