from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from chainlens.db import get_neo4j_driver, get_pg_engine

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample" / "transactions.parquet"


@pytest.fixture(scope="session")
def pg_engine():
    try:
        engine = get_pg_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - skip with context
        pytest.skip(
            f"Postgres is unreachable ({exc}). Start it with `make up`; "
            "the non-container tests still run."
        )
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def score_run(pg_engine):
    """Run Phase 5 once per test session so trace/score assertions have data."""
    from chainlens.scoring import run_score

    return run_score()


@pytest.fixture(scope="session")
def neo4j_driver():
    try:
        driver = get_neo4j_driver()
    except Exception as exc:  # noqa: BLE001 - skip with context
        pytest.skip(
            f"Neo4j is unreachable ({exc}). Start it with `make up`; "
            "the non-container tests still run."
        )
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        pytest.skip(f"Missing {SAMPLE_PATH}; run `make sample` first.")
    return pd.read_parquet(SAMPLE_PATH)
