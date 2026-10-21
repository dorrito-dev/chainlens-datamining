from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from chainlens.seed_scores import SEED_RULES_VERSION, seed_scores


def test_seed_scores_covers_every_feature_address_once(pg_engine):
    scores = pd.read_sql(
        "SELECT address, score, raw_score, tier, rules_version, top_driver "
        "FROM risk_scores",
        pg_engine,
    )
    features = set(
        pd.read_sql("SELECT address FROM address_features", pg_engine)["address"]
    )
    assert set(scores["address"]) == features
    assert scores["address"].is_unique
    assert len(scores) == len(features)


def test_seed_rows_are_zero_and_unscored(pg_engine):
    frame = pd.read_sql(
        "SELECT * FROM risk_scores WHERE rules_version = %(version)s",
        pg_engine,
        params={"version": SEED_RULES_VERSION},
    )
    if frame.empty:
        pytest.skip("Phase 5 scoring has replaced the Phase 4 seed rows")
    assert (frame["score"] == 0).all()
    assert (frame["raw_score"] == 0).all()
    assert (frame["tier"] == "unscored").all()
    assert (frame["rules_version"] == SEED_RULES_VERSION).all()
    assert frame["top_driver"].isna().all()


def test_seed_rows_are_idempotent(pg_engine):
    seed_scores(pg_engine)
    first = pd.read_sql("SELECT count(*) AS n FROM risk_scores", pg_engine).iloc[0, 0]
    seed_scores(pg_engine)
    second = pd.read_sql("SELECT count(*) AS n FROM risk_scores", pg_engine).iloc[0, 0]
    assert int(first) == int(second)


def test_delete_seed_version_in_transaction_removes_only_seed_rows(pg_engine):
    with pg_engine.connect() as conn:
        transaction = conn.begin()
        total_before = conn.execute(
            text("SELECT count(*) FROM risk_scores")
        ).scalar_one()
        seed_before = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        ).scalar_one()
        conn.execute(
            text("DELETE FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        )
        seed_after = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        ).scalar_one()
        total_after = conn.execute(
            text("SELECT count(*) FROM risk_scores")
        ).scalar_one()
        assert seed_after == 0
        assert total_after == total_before - seed_before
        transaction.rollback()

    with pg_engine.connect() as conn:
        restored = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        ).scalar_one()
    assert restored == seed_before
