from __future__ import annotations

import json

import pandas as pd


def _evidence(value):
    return json.loads(value) if isinstance(value, str) else value


def test_every_fanout_flag_matches_r2_params(pg_engine):
    frame = pd.read_sql(
        "SELECT address, evidence FROM address_flags WHERE signal_name = 'fanout'",
        pg_engine,
    )
    assert not frame.empty
    for row in frame.itertuples(index=False):
        evidence = _evidence(row.evidence)
        assert evidence["unique_counterparties"] >= 20
        assert evidence["tx_count_out"] >= 20
        assert evidence["fanout_ratio"] >= 0.5


def test_every_sanctioned_one_hop_seed_is_train_only(pg_engine):
    flags = pd.read_sql(
        "SELECT address, evidence FROM address_flags "
        "WHERE signal_name = 'sanctioned_1hop'",
        pg_engine,
    )
    train = set(
        pd.read_sql(
            "SELECT address FROM label_splits WHERE split = 'train'",
            pg_engine,
        )["address"]
    )
    assert not flags.empty
    for row in flags.itertuples(index=False):
        evidence = _evidence(row.evidence)
        assert evidence["seed"] in train
        assert evidence["seed"] != row.address


def test_every_cycle_flag_matches_r1_bounds(pg_engine):
    frame = pd.read_sql(
        "SELECT address, signal_value, evidence FROM address_flags "
        "WHERE signal_name = 'cycle_member'",
        pg_engine,
    )
    assert not frame.empty
    for row in frame.itertuples(index=False):
        evidence = _evidence(row.evidence)
        assert 3 <= int(row.signal_value) <= 5
        cycles = evidence.get("cycles") or [evidence]
        for cycle in cycles:
            length = int(
                cycle.get("cycle_length") or len(cycle.get("cycle_path") or [])
            )
            assert 3 <= length <= 5
            assert float(cycle["time_span_hours"]) <= 24


def test_every_apriori_match_has_lift_at_least_two(pg_engine):
    frame = pd.read_sql(
        "SELECT evidence FROM address_flags WHERE signal_name = 'apriori_match'",
        pg_engine,
    )
    for row in frame.itertuples(index=False):
        evidence = _evidence(row.evidence)
        assert float(evidence["lift"]) >= 2.0


def test_pagerank_top5_is_approximately_five_percent(pg_engine):
    node_count = int(
        pd.read_sql("SELECT count(*) AS n FROM address_features", pg_engine).iloc[0][
            "n"
        ]
    )
    top_count = int(
        pd.read_sql(
            "SELECT count(*) AS n FROM address_flags "
            "WHERE signal_name = 'pagerank_top5'",
            pg_engine,
        ).iloc[0]["n"]
    )
    assert abs(top_count - 0.05 * node_count) <= 2
