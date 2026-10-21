from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from chainlens.seed_scores import SEED_RULES_VERSION, clear_seed_scores

TEMP_ADDRESS = "0x" + "9" * 40


def test_seed_rows_are_gone_after_scoring(pg_engine, score_run):
    seed_rows = int(
        pd.read_sql(
            "SELECT count(*) AS n FROM risk_scores WHERE rules_version = %(version)s",
            pg_engine,
            params={"version": SEED_RULES_VERSION},
        ).iloc[0]["n"]
    )
    unscored = int(
        pd.read_sql(
            "SELECT count(*) AS n FROM risk_scores WHERE tier = 'unscored'",
            pg_engine,
        ).iloc[0]["n"]
    )
    assert seed_rows == 0
    assert unscored == 0


def test_clear_seed_scores_deletes_and_asserts_transactionally(pg_engine, score_run):
    with pg_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO risk_scores
                    (address, raw_score, score, tier, top_driver, rules_version)
                VALUES (:address, 0, 0, 'unscored', NULL, :version)
                """
            ),
            {"address": TEMP_ADDRESS, "version": SEED_RULES_VERSION},
        )

    deleted = clear_seed_scores(pg_engine)
    assert deleted >= 1

    with pg_engine.connect() as conn:
        remaining = conn.execute(
            text(
                "SELECT count(*) FROM risk_scores "
                "WHERE rules_version = :version"
            ),
            {"version": SEED_RULES_VERSION},
        ).scalar_one()
        unscored = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
        ).scalar_one()
        temporary = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE address = :address"),
            {"address": TEMP_ADDRESS},
        ).scalar_one()
    assert remaining == 0
    assert unscored == 0
    assert temporary == 0
