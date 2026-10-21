"""Admin page/action tests: confirmation gating and fixture-only deletion."""

from __future__ import annotations

from pathlib import Path

import neo4j
import pytest
from components import admin_actions, status
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "5_Admin.py"


class FakeResult:
    def __init__(self, *, scalar=1, rows=None, rowcount=0):
        self._scalar = scalar
        self._rows = rows or []
        self.rowcount = rowcount

    def scalar(self):
        return self._scalar

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self):
        self.statements: list[tuple[str, dict]] = []
        self.fixture_only = ["0xfixtureonly"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, clause, params=None):
        sql = str(clause)
        self.statements.append((sql, params or {}))
        if "SELECT DISTINCT l.address" in sql:
            return FakeResult(rows=self.fixture_only)
        if sql.strip().startswith("DELETE FROM labels WHERE source"):
            return FakeResult(rowcount=3)
        if "DELETE FROM label_splits" in sql:
            return FakeResult(rowcount=2)
        return FakeResult()


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *args):
        return False


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection

    def begin(self):
        return FakeBegin(self.connection)


def test_confirmation_requires_exact_token():
    assert admin_actions.confirmation_matches("DELETE", "DELETE") is True
    assert admin_actions.confirmation_matches("  DELETE ", "DELETE") is True
    assert admin_actions.confirmation_matches("delete", "DELETE") is False
    assert admin_actions.confirmation_matches("", "DELETE") is False


def test_drop_fixture_labels_targets_only_fixture_only_addresses():
    engine = FakeEngine()
    counts = admin_actions.drop_fixture_labels(engine)

    assert counts == {
        "labels_deleted": 3,
        "label_splits_deleted": 2,
        "fixture_only_addresses": 1,
    }
    split_statements = [
        (sql, params)
        for sql, params in engine.connection.statements
        if "DELETE FROM label_splits" in sql
    ]
    assert split_statements[0][1]["addresses"] == ["0xfixtureonly"]
    label_deletes = [
        sql
        for sql, _ in engine.connection.statements
        if sql.strip().startswith("DELETE FROM labels")
    ]
    assert label_deletes == ["DELETE FROM labels WHERE source = 'fixture'"]


class ExplodingConnection(FakeConnection):
    def execute(self, clause, params=None):
        sql = str(clause)
        if "DELETE FROM address_flags" in sql:
            raise RuntimeError("simulated database failure")
        return super().execute(clause, params)


class ExplodingEngine(FakeEngine):
    def __init__(self):
        self.connection = ExplodingConnection()

    def begin(self):
        return FakeBegin(self.connection)


def test_failed_danger_action_leaves_processed_files_untouched(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    sentinel = processed / "transactions_clean.parquet"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated database failure"):
        admin_actions.delete_all_pipeline_data(
            ExplodingEngine(), processed_dir=processed
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_admin_danger_button_needs_typed_confirmation(monkeypatch):
    monkeypatch.setattr(status, "get_engine_or_none", lambda: FakeEngine())
    monkeypatch.setattr(status, "running_count", lambda engine: 0)
    monkeypatch.setattr(status, "pipeline_runs_ready", lambda engine: True)
    monkeypatch.setattr(status, "version_snapshot", lambda engine: {})
    monkeypatch.setattr(status, "directory_usage", lambda path: 0)
    monkeypatch.setattr(
        neo4j.GraphDatabase, "driver", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off"))
    )

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert len(app.exception) == 0

    def delete_button():
        return next(button for button in app.button if button.label == "Delete all pipeline data")

    assert delete_button().disabled is True

    next(item for item in app.text_input if item.key == "confirm_delete").set_value("delete")
    app.run()
    assert delete_button().disabled is True

    next(item for item in app.text_input if item.key == "confirm_delete").set_value("DELETE")
    app.run()
    assert delete_button().disabled is False
