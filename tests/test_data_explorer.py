"""Data Explorer tests: whitelist, read-only enforcement, and row caps."""

from __future__ import annotations

import pandas as pd
from components import table_browser


class FakeTransaction:
    def __init__(self):
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class FakeConnection:
    def __init__(self):
        self.transaction = FakeTransaction()
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def begin(self):
        return self.transaction

    def execute(self, clause, *args, **kwargs):
        self.executed.append(str(clause))
        return self


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_explorer_lists_all_nine_operational_tables():
    assert list(table_browser.TABLE_PHASE) == [
        "labels",
        "label_splits",
        "known_addresses",
        "address_features",
        "feature_scaler",
        "address_flags",
        "risk_scores",
        "score_trace",
        "pipeline_runs",
    ]
    assert table_browser.TABLE_PHASE["pipeline_runs"] == 7


def test_read_only_sql_rejects_dml_and_ddl():
    for sql in (
        "DELETE FROM labels;",
        "UPDATE risk_scores SET score = 0;",
        "DROP TABLE labels;",
        "TRUNCATE labels;",
        "SELECT 1; DROP TABLE labels;",
        "SELECT * FROM labels -- hidden write",
        "/* comment */ SELECT 1",
        "WITH x AS (DELETE FROM labels RETURNING *) SELECT * FROM x",
    ):
        ok, reason = table_browser.validate_read_only_sql(sql)
        assert not ok, sql
        assert reason


def test_read_only_sql_accepts_select_and_cte():
    assert table_browser.validate_read_only_sql("SELECT * FROM labels;")[0]
    assert table_browser.validate_read_only_sql(
        "WITH x AS (SELECT 1 AS a) SELECT * FROM x"
    )[0]


def test_fetch_rows_uses_default_5000_row_cap(monkeypatch):
    captured = {}

    def fake_read_sql(query, conn, params=None):
        captured["params"] = params
        return pd.DataFrame({"address": ["0xabc"]})

    monkeypatch.setattr(table_browser.pd, "read_sql", fake_read_sql)
    frame = table_browser.fetch_rows(
        FakeEngine(), "address_flags", limit=table_browser.DEFAULT_LIMIT
    )

    assert len(frame) == 1
    assert captured["params"]["limit"] == 5_000
    assert captured["params"]["offset"] == 0


def test_read_only_query_caps_result_and_rolls_back(monkeypatch):
    big = pd.DataFrame({"x": range(6_000)})
    monkeypatch.setattr(table_browser.pd, "read_sql", lambda *a, **k: big)
    engine = FakeEngine()

    frame, error = table_browser.run_read_only_query(
        engine, "SELECT * FROM address_flags", max_rows=5_000
    )

    assert error == ""
    assert frame is not None
    assert len(frame) == 5_000
    assert engine.connection.transaction.rolled_back is True
    assert any("READ ONLY" in statement for statement in engine.connection.executed)
