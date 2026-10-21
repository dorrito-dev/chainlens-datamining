from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from streamlit import cache_data
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"


def _run_app(monkeypatch) -> AppTest:
    cache_data.clear()
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def test_all_tabs_render_with_neo4j_unreachable(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setenv("NEO4J_PASSWORD", "unreachable")
    app = _run_app(monkeypatch)

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "Ranked list",
        "Wallet breakdown",
        "Graph view",
    ]
    assert any("Full trace (JSON)" in expander.label for expander in app.expander)
    assert len(app.json) >= 1


def test_neo4j_link_is_hidden_when_unreachable(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setenv("NEO4J_PASSWORD", "unreachable")
    app = _run_app(monkeypatch)

    assert len(app.exception) == 0
    markdown_values = [str(getattr(item, "value", "")) for item in app.markdown]
    assert all("Neo4j Browser" not in value for value in markdown_values)


def test_empty_scores_state(monkeypatch):
    monkeypatch.setenv("CHAINLENS_FORCE_EMPTY_SCORES", "1")
    app = _run_app(monkeypatch)

    assert len(app.exception) == 0
    assert any(
        "No scores yet — run `make detect` then `make score`." in info.value
        for info in app.info
    )


def test_missing_trace_state(monkeypatch):
    monkeypatch.setenv("CHAINLENS_FORCE_MISSING_TRACE", "1")
    app = _run_app(monkeypatch)

    assert len(app.exception) == 0
    assert any(
        "Trace not found for rules_version" in info.value for info in app.info
    )


def test_tier_colors_and_style():
    spec = importlib.util.spec_from_file_location("chainlens_dashboard", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.TIER_COLORS["low"] == "#9e9e9e"
    assert module.TIER_COLORS["medium"] == "#ffb300"
    assert module.TIER_COLORS["high"] == "#d62728"
    high_row = pd.Series({"tier": "high", "score": 90})
    assert "d62728" in module._tier_row_style(high_row)[0]
