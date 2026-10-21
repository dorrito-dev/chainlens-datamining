"""Logs page tests using Streamlit's AppTest (no database required)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from components import status
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Logs.py"


def _runs_frame(count: int = 50) -> pd.DataFrame:
    now = datetime(2026, 10, 22, 14, 0, tzinfo=UTC)
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": 100 + index,
                "stage": "score" if index % 2 == 0 else "detect",
                "mode": "sample",
                "status": "success",
                "started_at": now,
                "finished_at": now,
                "exit_code": 0,
                "row_counts": {"risk_scores": 377},
                "version_snapshot": {"rules_version": "3e96aa94e670"},
                "log_path": f"/tmp/missing-{index}.log",
                "error": None,
                "triggered_by": "ui",
            }
        )
    return pd.DataFrame(rows)


def test_logs_page_renders_50_runs_and_missing_log_fallback(monkeypatch):
    frame = _runs_frame(50)
    monkeypatch.setattr(status, "get_engine_or_none", lambda: object())
    monkeypatch.setattr(status, "recent_runs", lambda engine, limit=50, offset=0: frame)
    monkeypatch.setattr(status, "read_log_tail", lambda path, lines=1000: None)

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "Logs"
    assert len(app.dataframe) >= 1
    assert len(app.dataframe[0].value) == 50
    assert any("Log file missing" in warning.value for warning in app.warning)


def test_logs_page_shows_real_log_text(monkeypatch, tmp_path):
    log_file = tmp_path / "score_sample.log"
    log_file.write_text("$ scripts/run.sh score\nstatus=success\n", encoding="utf-8")
    frame = _runs_frame(1)
    frame.loc[0, "log_path"] = str(log_file)
    monkeypatch.setattr(status, "get_engine_or_none", lambda: object())
    monkeypatch.setattr(status, "recent_runs", lambda engine, limit=50, offset=0: frame)
    monkeypatch.setattr(
        status,
        "read_log_tail",
        lambda path, lines=1000: Path(path).read_text(encoding="utf-8"),
    )

    app = AppTest.from_file(str(PAGE), default_timeout=30).run()

    assert len(app.exception) == 0
    code_values = [str(item.value) for item in app.code]
    assert any("status=success" in value for value in code_values)
