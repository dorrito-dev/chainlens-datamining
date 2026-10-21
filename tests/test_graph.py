from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from chainlens.fixtures import address_for
from chainlens.graph import _sanctioned_seeds


@pytest.fixture(autouse=True)
def _require_fixture_labels(pg_engine):
    count = pd.read_sql(
        "SELECT count(*) AS n FROM labels WHERE source = 'fixture'", pg_engine
    ).iloc[0]["n"]
    if not int(count):
        pytest.skip("Phase 4 fixture graph assertions require sample-mode labels")


def _flags(pg_engine, address: str, signal_name: str) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT signal_name, signal_value, evidence FROM address_flags "
        "WHERE address = %(address)s AND signal_name = %(signal)s",
        pg_engine,
        params={"address": address, "signal": signal_name},
    )


def _evidence(row: pd.Series) -> dict:
    value = row["evidence"]
    return value if isinstance(value, dict) else json.loads(value)


def test_cycle_fixture_is_flagged_with_constraints(pg_engine):
    for name in ("CYCLE_A", "CYCLE_B", "CYCLE_C"):
        frame = _flags(pg_engine, address_for(name), "cycle_member")
        assert len(frame) == 1, name
        evidence = _evidence(frame.iloc[0])
        assert int(frame.iloc[0]["signal_value"]) == 3
        assert float(evidence["time_span_hours"]) <= 24
        assert evidence["scc_size"] >= 3
        assert set(evidence["cycle_path"]) == {
            address_for("CYCLE_A"),
            address_for("CYCLE_B"),
            address_for("CYCLE_C"),
        }
        values = [float(value) for value in evidence["edge_values"]]
        assert max(values) / min(values) <= 1.05


def test_clean_users_have_no_cycle_or_fanout(pg_engine):
    for index in range(1, 11):
        address = address_for(f"CLEAN_USER_{index}")
        assert _flags(pg_engine, address, "cycle_member").empty
        assert _flags(pg_engine, address, "fanout").empty


def test_fanout_whale_uses_raw_outbound_counts(pg_engine):
    frame = _flags(pg_engine, address_for("FANOUT_WHALE"), "fanout")
    assert len(frame) == 1
    evidence = _evidence(frame.iloc[0])
    assert int(evidence["unique_counterparties"]) >= 20
    assert int(evidence["tx_count_out"]) >= 20
    assert float(evidence["fanout_ratio"]) == pytest.approx(1.0, abs=1e-6)
    assert int(frame.iloc[0]["signal_value"]) == int(
        evidence["unique_counterparties"]
    )


def test_exchange_hot_is_in_pagerank_top_five_percent(pg_engine):
    frame = _flags(pg_engine, address_for("EXCHANGE_HOT"), "pagerank_top5")
    assert len(frame) == 1
    evidence = _evidence(frame.iloc[0])
    assert float(evidence["pagerank"]) >= float(evidence["threshold"])


def test_sanctioned_one_hop_and_seed_split(pg_engine):
    frame = _flags(pg_engine, address_for("SANCTIONED_1HOP"), "sanctioned_1hop")
    assert len(frame) == 1
    evidence = _evidence(frame.iloc[0])
    assert evidence["seed"] == address_for("SANCTIONED_SEED")
    assert evidence["direction"] in {"in", "out", "both"}
    assert evidence["seed_source"] == "fixture"

    # The seed itself is labeled and must not receive the 1-hop signal.
    assert _flags(pg_engine, address_for("SANCTIONED_SEED"), "sanctioned_1hop").empty
    with pg_engine.connect() as conn:
        split = conn.execute(
            text("SELECT split FROM label_splits WHERE address = :address"),
            {"address": address_for("SANCTIONED_SEED")},
        ).scalar_one()
    assert split == "train"


def test_no_test_split_address_is_used_as_r3_seed(pg_engine):
    seeds = set(_sanctioned_seeds(pg_engine))
    test_addresses = set(
        pd.read_sql(
            "SELECT address FROM label_splits WHERE split = 'test'",
            pg_engine,
        )["address"]
    )
    assert seeds
    assert not (seeds & test_addresses)
    train_addresses = set(
        pd.read_sql(
            "SELECT address FROM label_splits WHERE split = 'train'",
            pg_engine,
        )["address"]
    )
    assert seeds <= train_addresses


def test_louvain_exists_for_every_node_but_is_not_a_rule(pg_engine):
    features = set(
        pd.read_sql("SELECT address FROM address_features", pg_engine)["address"]
    )
    louvain = set(
        pd.read_sql(
            "SELECT address FROM address_flags WHERE signal_name = 'louvain_community'",
            pg_engine,
        )["address"]
    )
    assert louvain == features

    rules_text = (
        Path(__file__).resolve().parents[1] / "config" / "rules.yaml"
    ).read_text(encoding="utf-8")
    assert "louvain" not in rules_text.lower()
