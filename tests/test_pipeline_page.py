"""Pipeline page tests with Streamlit AppTest.

The DB is mocked so CI never starts a real pipeline from a page test.
"""

from __future__ import annotations

from pathlib import Path

from components import stage_runner, status
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "1_Pipeline.py"


def _full_state():
    stages = {}
    for stage in status.STAGES:
        stages[stage] = {
            "stage": stage,
            "status": "success",
            "last": {"id": 1, "status": "success"},
            "success": {
                "id": 1,
                "finished_at": "2026-10-22T14:00:00+00:00",
                "row_counts": {},
            },
            "stale": False,
        }
    return {
        "db_ok": True,
        "runs_ready": True,
        "mode": "sample",
        "stages": stages,
        "active": None,
        "real_labels": 0,
        "fixture_labels": 3,
        "counts": {"address_features": 377, "address_flags": 500},
    }


def _button(app: AppTest, label: str):
    matches = [button for button in app.button if button.label == label]
    assert matches, f"Button {label!r} not found"
    return matches[0]


def test_pipeline_page_renders_with_empty_unreachable_db(monkeypatch):
    monkeypatch.setattr(status, "get_engine_or_none", lambda: None)
    monkeypatch.setattr(status, "default_mode", lambda engine: "sample")

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "Pipeline"
    assert "PostgreSQL is unavailable" in " ".join(
        warning.value for warning in app.warning
    )
    for label in ("Run Sample", "Run Ingest", "Run Features", "Run Detect", "Run Score"):
        assert _button(app, label).proto.disabled is True


def test_ingest_is_disabled_without_engine(monkeypatch):
    monkeypatch.setattr(status, "get_engine_or_none", lambda: None)
    monkeypatch.setattr(status, "default_mode", lambda engine: "sample")

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()

    assert len(app.exception) == 0
    assert _button(app, "Run Ingest").proto.disabled is True


def test_all_buttons_enabled_when_prerequisites_met(monkeypatch):
    monkeypatch.setattr(status, "get_engine_or_none", lambda: object())
    monkeypatch.setattr(status, "default_mode", lambda engine: "sample")
    monkeypatch.setattr(status, "pipeline_runs_ready", lambda engine: True)
    monkeypatch.setattr(status, "pipeline_state", lambda engine, mode: _full_state())
    monkeypatch.setattr(status, "latest_map", lambda engine, mode=None: {})
    monkeypatch.setattr(status, "real_label_count", lambda engine: 0)
    monkeypatch.setattr(stage_runner, "prerequisite_reason", lambda *a, **k: None)

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()

    assert len(app.exception) == 0
    for label in ("Run Sample", "Run Ingest", "Run Features", "Run Detect", "Run Score"):
        assert _button(app, label).proto.disabled is False
