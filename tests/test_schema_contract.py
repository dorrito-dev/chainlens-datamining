from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from chainlens.db import apply_schema, table_exists

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
ADDRESSES = [f"0x{i:040x}" for i in range(1, 9)]


def schema_tables() -> list[str]:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    return re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", ddl)


def test_apply_schema_is_idempotent(pg_engine):
    apply_schema(pg_engine)
    apply_schema(pg_engine)


def test_every_schema_table_exists(pg_engine):
    apply_schema(pg_engine)
    tables = schema_tables()
    assert len(tables) == 10, tables
    for table in tables:
        assert table_exists(pg_engine, table), f"Missing table: {table}"


def test_address_features_has_no_flagged_column(pg_engine):
    apply_schema(pg_engine)
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'address_features'
          AND column_name = 'flagged'
        """
    )
    with pg_engine.connect() as conn:
        assert conn.execute(query).first() is None


def test_risk_scores_rejects_score_above_100(pg_engine):
    apply_schema(pg_engine)
    insert = text(
        """
        INSERT INTO risk_scores (address, raw_score, score, tier, rules_version)
        VALUES (:address, 101, 101, 'high', 'test')
        """
    )
    with pytest.raises(IntegrityError):
        with pg_engine.begin() as conn:
            conn.execute(insert, {"address": ADDRESSES[0]})


def test_minimal_insert_and_read_back_for_every_table(pg_engine):
    apply_schema(pg_engine)
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(
                text(
                    """
                    INSERT INTO labels (address, label, source, confidence, as_of)
                    VALUES (:address, 'sanctioned', 'test', 1.0, :as_of)
                    """
                ),
                {"address": ADDRESSES[0], "as_of": date(2025, 1, 1)},
            )
            conn.execute(
                text(
                    "INSERT INTO label_splits (address, split, seed) "
                    "VALUES (:address, 'train', 42)"
                ),
                {"address": ADDRESSES[0]},
            )
            conn.execute(
                text(
                    "INSERT INTO known_addresses (address, category, note) "
                    "VALUES (:address, 'other', 'contract test')"
                ),
                {"address": ADDRESSES[0]},
            )
            conn.execute(
                text(
                    "INSERT INTO address_features "
                    "(address, tx_count_out, tx_count_total, scaler_version) "
                    "VALUES (:address, 1, 1, 'test')"
                ),
                {"address": ADDRESSES[0]},
            )
            conn.execute(
                text(
                    "INSERT INTO feature_scaler "
                    "(feature_name, transform, mean, std, fitted_on_split, "
                    "row_count, scaler_version) "
                    "VALUES ('test_scaler_feature', 'log1p_std', 0.5, 0.25, 'train', 1, 'test')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO feature_scaler_dropped (feature_name, reason) "
                    "VALUES ('test_dropped_feature', 'zero_variance')"
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO address_flags
                        (address, signal_name, signal_value, evidence)
                    VALUES (:address, 'pagerank', 0.9, CAST(:evidence AS JSONB))
                    """
                ),
                {"address": ADDRESSES[0], "evidence": '{"rank": 1}'},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO risk_scores
                        (address, raw_score, score, tier, top_driver, rules_version)
                    VALUES (:address, 15, 15, 'low', 'R4', 'test')
                    """
                ),
                {"address": ADDRESSES[0]},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO score_trace (address, rules_version, trace)
                    VALUES (:address, 'test', CAST(:trace AS JSONB))
                    """
                ),
                {"address": ADDRESSES[0], "trace": '{"R4": {"hit": true}}'},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO data_manifest
                        (file_path, sha256, row_count, block_start, block_end)
                    VALUES ('data/sample/test.parquet', :sha256, 1, 1, 2)
                    """
                ),
                {"sha256": "0" * 64},
            )

            checks = {
                "labels": ("address = :address", {"address": ADDRESSES[0]}),
                "label_splits": ("address = :address", {"address": ADDRESSES[0]}),
                "known_addresses": ("address = :address", {"address": ADDRESSES[0]}),
                "address_features": ("address = :address", {"address": ADDRESSES[0]}),
                "feature_scaler": (
                    "feature_name = :feature_name",
                    {"feature_name": "test_scaler_feature"},
                ),
                "feature_scaler_dropped": (
                    "feature_name = :feature_name",
                    {"feature_name": "test_dropped_feature"},
                ),
                "address_flags": ("address = :address", {"address": ADDRESSES[0]}),
                "risk_scores": ("address = :address", {"address": ADDRESSES[0]}),
                "score_trace": ("address = :address", {"address": ADDRESSES[0]}),
                "data_manifest": (
                    "file_path = :file_path",
                    {"file_path": "data/sample/test.parquet"},
                ),
            }
            for table, (where, params) in checks.items():
                count = conn.execute(
                    text(f"SELECT count(*) FROM {table} WHERE {where}"),  # noqa: S608
                    params,
                ).scalar_one()
                assert count == 1, table
        finally:
            trans.rollback()
