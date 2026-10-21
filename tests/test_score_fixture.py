from __future__ import annotations

import pandas as pd

from chainlens.fixtures import address_for


def _score(pg_engine, address: str) -> dict:
    frame = pd.read_sql(
        "SELECT * FROM risk_scores WHERE address = %(address)s",
        pg_engine,
        params={"address": address},
    )
    assert len(frame) == 1, address
    return frame.iloc[0].to_dict()


def _trace(pg_engine, address: str) -> dict:
    frame = pd.read_sql(
        "SELECT trace FROM score_trace WHERE address = %(address)s",
        pg_engine,
        params={"address": address},
    )
    assert len(frame) == 1, address
    return frame.iloc[0]["trace"]


def _has_flag(pg_engine, address: str, signal_name: str) -> bool:
    frame = pd.read_sql(
        "SELECT 1 FROM address_flags WHERE address = %(address)s "
        "AND signal_name = %(signal)s",
        pg_engine,
        params={"address": address, "signal": signal_name},
    )
    return not frame.empty


def test_cycle_fixture_scores(pg_engine, score_run):
    for name in ("CYCLE_A", "CYCLE_B", "CYCLE_C"):
        address = address_for(name)
        score = _score(pg_engine, address)
        trace = _trace(pg_engine, address)
        assert score["raw_score"] == 25
        assert score["score"] == 25
        assert score["tier"] == "medium"
        assert score["top_driver"] == "R1"
        assert trace["rules"]["R1"]["hit"] is True
        assert trace["truncated"] is False


def test_sanctioned_one_hop_fixture_scores(pg_engine, score_run):
    address = address_for("SANCTIONED_1HOP")
    score = _score(pg_engine, address)
    trace = _trace(pg_engine, address)
    assert score["raw_score"] == 30
    assert score["score"] == 30
    assert score["tier"] == "medium"
    assert score["top_driver"] == "R3"
    assert trace["rules"]["R3"]["hit"] is True


def test_clean_users_score_zero(pg_engine, score_run):
    for index in range(1, 11):
        address = address_for(f"CLEAN_USER_{index}")
        score = _score(pg_engine, address)
        trace = _trace(pg_engine, address)
        assert score["score"] == 0
        assert score["raw_score"] == 0
        assert score["tier"] == "low"
        assert score["top_driver"] is None
        assert all(
            rule["hit"] is False for rule in trace["rules"].values()
        )


def test_seed_scores_zero_under_deck_pure_policy(pg_engine, score_run):
    address = address_for("SANCTIONED_SEED")
    score = _score(pg_engine, address)
    trace = _trace(pg_engine, address)
    assert score["score"] == 0
    assert score["raw_score"] == 0
    assert score["tier"] == "low"
    assert score["top_driver"] is None
    assert all(rule["hit"] is False for rule in trace["rules"].values())


def test_fanout_whale_score_depends_on_pagerank_flag(pg_engine, score_run):
    address = address_for("FANOUT_WHALE")
    score = _score(pg_engine, address)
    expected = 35 if _has_flag(pg_engine, address, "pagerank_top5") else 20
    assert score["raw_score"] == expected
    assert score["score"] == expected
    assert score["tier"] == ("medium" if expected == 35 else "low")
    assert score["top_driver"] == "R2"
