"""Phase 4 placeholder score seed for the Streamlit shell."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

SEED_RULES_VERSION = "seed-phase4"


def clear_seed_scores(engine: Engine) -> int:
    """Delete Phase 4 placeholder rows and assert no ``unscored`` rows remain.

    The deletion and assertion share one transaction, so a failed assertion
    restores the seed rows instead of leaving a half-migrated score table.
    """
    with engine.begin() as conn:
        deleted = conn.execute(
            text("DELETE FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        ).rowcount
        unscored = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
        ).scalar_one()
        if unscored:
            raise RuntimeError(
                f"Seed cleanup failed: {unscored} tier='unscored' row(s) remain"
            )
    print(f"Seed cleanup: deleted {deleted} row(s) with rules_version='seed-phase4'")
    return int(deleted or 0)


def seed_scores(engine: Engine) -> dict:
    """Insert zero scores for addresses that do not already have a real score."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        )
        conn.execute(
            text(
                """
                INSERT INTO risk_scores (
                    address, raw_score, score, tier, top_driver, rules_version
                )
                SELECT
                    af.address, 0, 0, 'unscored', NULL, :version
                FROM address_features af
                WHERE NOT EXISTS (
                    SELECT 1 FROM risk_scores rs WHERE rs.address = af.address
                )
                ON CONFLICT (address) DO NOTHING
                """
            ),
            {"version": SEED_RULES_VERSION},
        )
    with engine.connect() as conn:
        seeded = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE rules_version = :version"),
            {"version": SEED_RULES_VERSION},
        ).scalar_one()
        total = conn.execute(text("SELECT count(*) FROM risk_scores")).scalar_one()
    print(
        f"Seed scores: {seeded} row(s) with rules_version={SEED_RULES_VERSION!r}; "
        f"risk_scores total={total}"
    )
    return {
        "seeded": int(seeded),
        "total": int(total),
        "rules_version": SEED_RULES_VERSION,
    }
