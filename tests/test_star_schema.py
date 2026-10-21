from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from chainlens.config import load_data_config, project_path
from chainlens.db import table_exists
from chainlens.star_schema import (
    STAR_SCHEMA_PATH,
    apply_star_schema,
    populate_star_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def test_star_schema_sql_is_idempotent(pg_engine):
    apply_star_schema(pg_engine)
    apply_star_schema(pg_engine)
    for table in ("fact_transactions", "dim_address", "dim_label"):
        assert table_exists(pg_engine, table)


def test_star_schema_population_matches_clean_parquet(pg_engine):
    clean_path = project_path(load_data_config()["ingest"]["clean_parquet"])
    source_rows = len(pd.read_parquet(clean_path))

    first = populate_star_schema(pg_engine, clean_path)
    second = populate_star_schema(pg_engine, clean_path)
    assert first["source_rows"] == source_rows
    assert first["fact_transactions"] == source_rows
    assert second["fact_transactions"] == source_rows
    assert first["dim_address"] > 0
    assert first["dim_label"] > 0

    with pg_engine.connect() as conn:
        fact_count = int(
            conn.execute(text("SELECT count(*) FROM fact_transactions")).scalar_one()
        )
    assert fact_count == source_rows


def test_no_pipeline_module_imports_fact_transactions():
    pattern = r"^\s*(from|import)\s+.*\bfact_transactions\b"
    result = subprocess.run(
        ["grep", "-R", "-n", "-E", pattern, "chainlens", "pipeline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
    assert result.stdout == ""


def test_star_schema_file_is_presentation_artifact():
    ddl = STAR_SCHEMA_PATH.read_text(encoding="utf-8")
    tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", ddl))
    assert tables == {"fact_transactions", "dim_address", "dim_label"}
