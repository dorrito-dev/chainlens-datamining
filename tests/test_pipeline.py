"""End-to-end sample pipeline and dashboard smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from scoring import validate_trace

ROOT = Path(__file__).resolve().parents[1]


def test_sample_pipeline_contract(sample_output: Path) -> None:
    required = [
        "transactions_raw.parquet", "transactions_clean.parquet",
        "wallet_features.parquet", "flags.parquet", "scores.parquet",
        "traces.parquet", "graph_edges.parquet", "labels.parquet",
        "splits.parquet", "known_addresses.parquet", "evaluation.json",
        "evaluation.md", "ablation.csv", "run_info.json", "scores.csv",
        "traces.csv", "chainlens.duckdb",
    ]
    for name in required:
        assert (sample_output / name).exists(), name

    scores = pd.read_parquet(sample_output / "scores.parquet")
    traces = pd.read_parquet(sample_output / "traces.parquet")
    features = pd.read_parquet(sample_output / "wallet_features.parquet")
    flags = pd.read_parquet(sample_output / "flags.parquet")
    edges = pd.read_parquet(sample_output / "graph_edges.parquet")
    run_info = json.loads((sample_output / "run_info.json").read_text())
    evaluation = json.loads((sample_output / "evaluation.json").read_text())

    assert len(scores) == len(traces) == len(features) == 377
    assert scores["address"].is_unique
    assert set(scores["tier"]) <= {"low", "medium", "high"}
    assert len(edges) > 0
    assert {"from_address", "to_address", "tx_count", "total_value_eth"} <= set(edges.columns)
    assert {"cycle_member", "fanout", "sanctioned_1hop", "pagerank_top5"} & set(flags["signal_name"])
    assert "dbscan_noise" in set(flags["signal_name"])
    assert evaluation["insufficient"] is True
    assert run_info["mode"] == "sample"
    assert run_info["data_source"] == "sample"
    assert run_info["api_key_present"] is False
    assert len(run_info["rules_sha256"]) == 64
    assert len(run_info["scaler_version"]) == 12
    assert run_info["patterns"]["graph_nodes"] == 377

    parsed = traces["trace_json"].map(json.loads)
    assert len(parsed) == 377
    for trace in parsed.head(10):
        validate_trace(trace)

    con = duckdb.connect(str(sample_output / "chainlens.duckdb"), read_only=True)
    try:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        for table in (
            "transactions", "wallet_features", "feature_scaler", "labels",
            "splits", "known_addresses", "wallet_flags", "risk_scores",
            "score_trace", "graph_edges", "run_info",
        ):
            assert table in tables, table
        assert con.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0] == 377
    finally:
        con.close()


def test_no_api_key_in_sample_artifacts(sample_output: Path) -> None:
    import fetch_data

    key = fetch_data.load_api_key(required=False)
    if not key:
        return
    for path in sample_output.glob("*"):
        if path.suffix in {".json", ".csv", ".md", ".yaml", ".txt"}:
            assert key not in path.read_text(encoding="utf-8", errors="ignore"), path.name


def test_dashboard_starts_without_exceptions(sample_output: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHAINLENS_OUTPUT_DIR", str(sample_output))
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90)
    app.run()
    assert len(app.exception) == 0, [e.value for e in app.exception]
    assert len(app.metric) >= 5

    radio = app.radio(key="nav_radio")
    assert set(radio.options) >= {
        "Overview", "Triage", "Wallet dossier", "Network",
        "Patterns", "Evaluation", "Data & export",
    }

    # Verify each view renders cleanly without exceptions
    for view in ["Triage", "Wallet dossier", "Network", "Patterns", "Evaluation", "Data & export"]:
        app.radio(key="nav_radio").set_value(view).run()
        assert len(app.exception) == 0, f"Error in view {view}: {[e.value for e in app.exception]}"


def test_dashboard_row_navigation(sample_output: Path, monkeypatch) -> None:
    """Row buttons must actually move to the dossier of the clicked wallet."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHAINLENS_OUTPUT_DIR", str(sample_output))
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90)
    app.run()

    labels = [button.label for button in app.button]
    if any(label.startswith("Dossier") for label in labels):
        index = next(i for i, label in enumerate(labels) if label.startswith("Dossier"))
        expected = app.button[index].key.removeprefix("top_")
        app.button[index].click().run()
        assert app.session_state["active_view"] == "Wallet dossier"
        assert app.session_state["selected_wallet"] == expected
        assert len(app.exception) == 0, [e.value for e in app.exception]

    app.radio(key="nav_radio").set_value("Triage").run()
    inspect = [button for button in app.button if button.label == "Inspect"]
    if inspect:
        expected = inspect[0].key.removeprefix("triage_btn_")
        inspect[0].click().run()
        assert app.session_state["active_view"] == "Wallet dossier"
        assert app.session_state["selected_wallet"] == expected
        assert any("Wallet Investigative Dossier" in item.value for item in app.markdown)
        assert len(app.exception) == 0, [e.value for e in app.exception]

    # The sidebar quick jump must always navigate, even when no row buttons exist.
    app.radio(key="nav_radio").set_value("Overview").run()
    address = app.session_state["selected_wallet"]
    app.text_input[0].set_value(address)
    next(b for b in app.button if b.label == "Open dossier").click()
    app.run()
    assert app.session_state["active_view"] == "Wallet dossier"
    assert app.session_state["selected_wallet"] == address
    assert len(app.exception) == 0, [e.value for e in app.exception]

    # Dossier exposes the full address in a copyable code block.
    assert len(app.code) >= 1


def test_dashboard_network_defaults_to_connected_wallet(sample_output: Path, monkeypatch) -> None:
    """Network view must not open on a lone wallet when hubs exist."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHAINLENS_OUTPUT_DIR", str(sample_output))
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90)
    app.run()
    app.radio(key="nav_radio").set_value("Network").run()
    assert len(app.exception) == 0, [e.value for e in app.exception]

    captions = [c.value for c in app.caption if c.value.startswith("Rendered subgraph")]
    assert captions, "subgraph caption missing"
    nodes = int(captions[0].split()[2])
    assert nodes >= 5, captions[0]
    assert len(app.selectbox(key="net_hub_pick").options) > 0


def test_dashboard_read_only_contract() -> None:
    content = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    for forbidden in ["import scoring", "import patterns", "import evaluate", "from scoring import", "from patterns import", "from evaluate import"]:
        assert forbidden not in content, f"Dashboard violates read-only contract by importing: {forbidden}"
