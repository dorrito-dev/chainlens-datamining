"""Transactional Admin actions for Phase 7.

Kept outside the Streamlit page so the cleanup rules can be unit-tested without
starting a web session. Every function runs inside one Postgres transaction and
raises on failure so the caller can report a rollback.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import PROJECT_ROOT

DANGER_TABLES = (
    "address_features",
    "address_flags",
    "risk_scores",
    "score_trace",
    "feature_scaler",
)


def confirmation_matches(typed: str, token: str) -> bool:
    """True only for the exact confirmation token (surrounding space ignored)."""
    return (typed or "").strip() == token


def _row_counts(conn, tables) -> dict[str, int]:
    return {
        table: int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())
        for table in tables
    }


def delete_all_pipeline_data(
    engine: Engine,
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Delete derived tables and processed files; ground truth stays intact."""
    with engine.begin() as conn:
        counts: dict[str, Any] = _row_counts(
            conn, [*DANGER_TABLES, "pipeline_runs"]
        )
        for table in DANGER_TABLES:
            conn.execute(text(f"DELETE FROM {table}"))
        conn.execute(text("DELETE FROM pipeline_runs"))

    processed = Path(processed_dir) if processed_dir else PROJECT_ROOT / "data" / "processed"
    deleted_files = 0
    errors: list[str] = []
    if processed.exists():
        for item in processed.iterdir():
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted_files += 1
            except OSError as exc:
                errors.append(f"{item.name}: {exc}")
    counts["processed_entries_deleted"] = deleted_files
    if errors:
        counts["processed_delete_errors"] = "; ".join(errors)
    return counts


def wipe_labels(engine: Engine) -> dict[str, int]:
    """Delete labels, splits and known addresses before switching modes."""
    with engine.begin() as conn:
        counts = _row_counts(conn, ["labels", "label_splits", "known_addresses"])
        conn.execute(text("DELETE FROM label_splits"))
        conn.execute(text("DELETE FROM labels"))
        conn.execute(text("DELETE FROM known_addresses"))
    return counts


def drop_fixture_labels(engine: Engine) -> dict[str, int]:
    """Delete only fixture labels and splits for fixture-only addresses."""
    with engine.begin() as conn:
        fixture_only = conn.execute(
            text(
                """
                SELECT DISTINCT l.address
                FROM labels l
                WHERE l.source = 'fixture'
                  AND NOT EXISTS (
                      SELECT 1 FROM labels other
                      WHERE other.address = l.address
                        AND other.source <> 'fixture'
                  )
                """
            )
        ).scalars().all()
        deleted_labels = conn.execute(
            text("DELETE FROM labels WHERE source = 'fixture'")
        ).rowcount
        deleted_splits = 0
        if fixture_only:
            deleted_splits = conn.execute(
                text("DELETE FROM label_splits WHERE address = ANY(:addresses)"),
                {"addresses": list(fixture_only)},
            ).rowcount
    return {
        "labels_deleted": int(deleted_labels or 0),
        "label_splits_deleted": int(deleted_splits or 0),
        "fixture_only_addresses": len(fixture_only),
    }
