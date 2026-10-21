from __future__ import annotations

import pandas as pd

from chainlens.fixtures import address_for


def _score(pg_engine, address: str) -> dict:
    frame = pd.read_sql(
        "SELECT * FROM risk_scores WHERE address = %(address)s",
        pg_engine,
        params={"address": address},
    )
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


def _trace(pg_engine, address: str) -> dict:
    frame = pd.read_sql(
        "SELECT trace FROM score_trace WHERE address = %(address)s",
        pg_engine,
        params={"address": address},
    )
    assert len(frame) == 1
    return frame.iloc[0]["trace"]


def test_exchange_hot_is_known_exchange(pg_engine):
    frame = pd.read_sql(
        "SELECT category FROM known_addresses WHERE address = %(address)s",
        pg_engine,
        params={"address": address_for("EXCHANGE_HOT")},
    )
    assert len(frame) == 1
    assert frame.iloc[0]["category"] == "exchange"


def test_exchange_hot_is_downweighted_before_clamping(pg_engine, score_run):
    exchange = address_for("EXCHANGE_HOT")
    whale = address_for("FANOUT_WHALE")
    score = _score(pg_engine, exchange)
    trace = _trace(pg_engine, exchange)
    whale_score = _score(pg_engine, whale)

    assert score["raw_score"] == 35
    assert score["score"] == 10
    assert score["tier"] == "low"
    assert score["score"] < whale_score["score"]
    assert trace["downweight"]["applied"] is True
    assert trace["downweight"]["category"] == "exchange"
    assert trace["downweight"]["factor"] == 0.3
    assert trace["downweight"]["pre_downweight_raw"] == 35
    assert trace["downweight"]["post_downweight_raw"] == 10.5


def test_exchange_hot_is_not_in_the_top_ten(pg_engine, score_run):
    ranked = pd.read_sql(
        "SELECT address, score FROM risk_scores "
        "ORDER BY score DESC, raw_score DESC, address ASC",
        pg_engine,
    ).reset_index(drop=True)
    top_ten = set(ranked.head(10)["address"])
    assert address_for("EXCHANGE_HOT") not in top_ten
