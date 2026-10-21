"""Read/write helpers for Phase 7 operational state.

The database is the source of truth: the UI never keeps pipeline state in
memory. Every query in this module is defensive because the ops console must
still render when Postgres, Neo4j, or ``pipeline_runs`` is missing.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.db import get_pg_engine, table_exists

ROOT = Path(__file__).resolve().parents[2]

STAGES: tuple[str, ...] = ("sample", "ingest", "features", "detect", "score")
STAGE_TITLES = {
    "sample": "Sample",
    "ingest": "Ingest",
    "features": "Features",
    "detect": "Detect",
    "score": "Score",
}
STAGE_DESCRIPTIONS = {
    "sample": "Generate synthetic fixtures and reload labels/splits/known addresses.",
    "ingest": "Load labels, curate known addresses, split addresses and build clean Parquet.",
    "features": "Build one-row-per-address features and fit the train-only scaler.",
    "detect": "Run DBSCAN, graph signals, train-only Apriori and seed scores.",
    "score": "Apply locked R1–R5 weights and persist risk_scores + score_trace.",
}
STAGE_PREDECESSOR = {
    "ingest": "sample",
    "features": "ingest",
    "detect": "features",
    "score": "detect",
}

# Key input files used to detect a stale stage (file newer than the run).
STAGE_INPUT_PATHS: dict[str, dict[str, str]] = {
    "ingest": {
        "sample": "data/sample/transactions.parquet",
        "real": "data/raw/transactions.parquet",
    },
    "features": {
        "sample": "data/processed/transactions_clean.parquet",
        "real": "data/processed/transactions_clean.parquet",
    },
}

RUN_COLUMNS = [
    "id",
    "stage",
    "mode",
    "status",
    "started_at",
    "finished_at",
    "exit_code",
    "row_counts",
    "version_snapshot",
    "log_path",
    "error",
    "triggered_by",
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def get_engine_or_none() -> Engine | None:
    """Return a live engine, or None when Postgres is unreachable."""
    try:
        engine = get_pg_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:  # noqa: BLE001 - UI must render without a database
        return None


def pipeline_runs_ready(engine: Engine | None) -> bool:
    if engine is None:
        return False
    try:
        return table_exists(engine, "pipeline_runs")
    except Exception:  # noqa: BLE001
        return False


def empty_runs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RUN_COLUMNS)


def _as_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping) if row is not None else {}


# ---------------------------------------------------------------------------
# pipeline_runs reads
# ---------------------------------------------------------------------------
def recent_runs(engine: Engine | None, limit: int = 50, offset: int = 0) -> pd.DataFrame:
    if engine is None or not pipeline_runs_ready(engine):
        return empty_runs_frame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text(
                    """
                    SELECT id, stage, mode, status, started_at, finished_at,
                           exit_code, row_counts, version_snapshot, log_path,
                           error, triggered_by
                    FROM pipeline_runs
                    ORDER BY started_at DESC, id DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                conn,
                params={"limit": int(limit), "offset": int(offset)},
            )
    except Exception:  # noqa: BLE001
        return empty_runs_frame()


def get_run(engine: Engine | None, run_id: int) -> dict[str, Any] | None:
    if engine is None or not pipeline_runs_ready(engine):
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM pipeline_runs WHERE id = :run_id"),
                {"run_id": int(run_id)},
            ).first()
        return _as_dict(row) if row is not None else None
    except Exception:  # noqa: BLE001
        return None


def last_success_map(engine: Engine | None, mode: str | None = None) -> dict[str, dict[str, Any]]:
    if engine is None or not pipeline_runs_ready(engine):
        return {}
    mode_clause = "AND mode = :mode" if mode else ""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT DISTINCT ON (stage)
                           id, stage, mode, status, started_at, finished_at,
                           exit_code, row_counts, version_snapshot, log_path,
                           error, triggered_by
                    FROM pipeline_runs
                    WHERE status = 'success'
                      AND stage <> 'admin'
                      {mode_clause}
                    ORDER BY stage, finished_at DESC NULLS LAST, id DESC
                    """
                ),
                {"mode": mode} if mode else {},
            ).all()
        return {row._mapping["stage"]: _as_dict(row) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


def latest_map(engine: Engine | None, mode: str | None = None) -> dict[str, dict[str, Any]]:
    if engine is None or not pipeline_runs_ready(engine):
        return {}
    mode_clause = "AND mode = :mode" if mode else ""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT DISTINCT ON (stage)
                           id, stage, mode, status, started_at, finished_at,
                           exit_code, row_counts, version_snapshot, log_path,
                           error, triggered_by
                    FROM pipeline_runs
                    WHERE stage <> 'admin'
                      {mode_clause}
                    ORDER BY stage, started_at DESC, id DESC
                    """
                ),
                {"mode": mode} if mode else {},
            ).all()
        return {row._mapping["stage"]: _as_dict(row) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


def active_run(engine: Engine | None, max_age_minutes: int = 20) -> dict[str, Any] | None:
    """Return a recent running row. Older rows are treated as crashed, not live."""
    if engine is None or not pipeline_runs_ready(engine):
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT * FROM pipeline_runs
                    WHERE status = 'running'
                      AND started_at > now() - (:minutes * interval '1 minute')
                    ORDER BY started_at DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"minutes": int(max_age_minutes)},
            ).first()
        return _as_dict(row) if row is not None else None
    except Exception:  # noqa: BLE001
        return None


def running_count(engine: Engine | None) -> int:
    """Count any running row, regardless of age (Admin lockout)."""
    if engine is None or not pipeline_runs_ready(engine):
        return 0
    try:
        with engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT count(*) FROM pipeline_runs WHERE status = 'running'")
                ).scalar_one()
            )
    except Exception:  # noqa: BLE001
        return 0


def mark_stale_runs_failed(engine: Engine, max_age_minutes: int = 20) -> int:
    """Mark crashed running rows as failed. Used by the Admin page only."""
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE pipeline_runs
                SET status = 'failed',
                    finished_at = now(),
                    error = COALESCE(error, 'Marked failed by Admin: run exceeded '
                                              || CAST(:minutes AS TEXT) || ' minutes')
                WHERE status = 'running'
                  AND started_at <= now() - (:minutes * interval '1 minute')
                """
            ),
            {"minutes": int(max_age_minutes)},
        )
    return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# Counts and state
# ---------------------------------------------------------------------------
def count_table(engine: Engine | None, table: str) -> int | None:
    if engine is None:
        return None
    # The table name is always chosen from a static whitelist by the caller.
    if table not in {
        "labels",
        "label_splits",
        "known_addresses",
        "address_features",
        "feature_scaler",
        "address_flags",
        "risk_scores",
        "score_trace",
        "pipeline_runs",
        "data_manifest",
    }:
        return None
    try:
        if not table_exists(engine, table):
            return None
        with engine.connect() as conn:
            return int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())
    except Exception:  # noqa: BLE001
        return None


def real_label_count(engine: Engine | None) -> int | None:
    if engine is None or not table_exists(engine, "labels"):
        return None
    try:
        with engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT count(*) FROM labels WHERE source <> 'fixture'")
                ).scalar_one()
            )
    except Exception:  # noqa: BLE001
        return None


def fixture_label_count(engine: Engine | None) -> int | None:
    if engine is None or not table_exists(engine, "labels"):
        return None
    try:
        with engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT count(*) FROM labels WHERE source = 'fixture'")
                ).scalar_one()
            )
    except Exception:  # noqa: BLE001
        return None


def default_mode(engine: Engine | None) -> str:
    """Latest recorded mode, else sample if fixture labels exist, else sample."""
    if engine is not None and pipeline_runs_ready(engine):
        try:
            with engine.connect() as conn:
                mode = conn.execute(
                    text(
                        "SELECT mode FROM pipeline_runs "
                        "WHERE stage <> 'admin' "
                        "ORDER BY started_at DESC, id DESC LIMIT 1"
                    )
                ).scalar()
            if mode in {"sample", "real"}:
                return str(mode)
        except Exception:  # noqa: BLE001
            pass
    return "sample"


def latest_scored_rules_version(engine: Engine | None) -> str | None:
    if engine is None or not table_exists(engine, "risk_scores"):
        return None
    try:
        with engine.connect() as conn:
            value = conn.execute(
                text(
                    """
                    SELECT rules_version FROM risk_scores
                    WHERE rules_version <> 'seed-phase4'
                    ORDER BY scored_at DESC, rules_version DESC
                    LIMIT 1
                    """
                )
            ).scalar()
        return str(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def version_snapshot(engine: Engine | None) -> dict[str, Any]:
    """Read the current version chain from the database, never from memory."""
    snapshot: dict[str, Any] = {
        "scaler_version": None,
        "rules_version": None,
        "rules_version_label": None,
        "engine_version": None,
    }
    if engine is None:
        return snapshot
    try:
        if table_exists(engine, "feature_scaler"):
            with engine.connect() as conn:
                snapshot["scaler_version"] = conn.execute(
                    text(
                        "SELECT scaler_version FROM feature_scaler "
                        "ORDER BY fitted_at DESC, feature_name ASC LIMIT 1"
                    )
                ).scalar()
    except Exception:  # noqa: BLE001
        pass
    rules_version = latest_scored_rules_version(engine)
    snapshot["rules_version"] = rules_version
    if rules_version and table_exists(engine, "score_trace"):
        try:
            with engine.connect() as conn:
                trace = conn.execute(
                    text(
                        "SELECT trace FROM score_trace "
                        "WHERE rules_version = :version LIMIT 1"
                    ),
                    {"version": rules_version},
                ).scalar()
            if isinstance(trace, dict):
                snapshot["rules_version_label"] = trace.get("rules_version_label")
                snapshot["engine_version"] = trace.get("engine_version")
        except Exception:  # noqa: BLE001
            pass
    return snapshot


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def stage_is_stale(
    stage: str,
    mode: str,
    success_row: dict[str, Any] | None,
    last_successes: dict[str, dict[str, Any]],
    engine: Engine | None,
) -> bool:
    if not success_row:
        return False
    finished = success_row.get("finished_at")
    if finished is None:
        return True
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)

    relative = STAGE_INPUT_PATHS.get(stage, {}).get(mode)
    if relative:
        modified = _mtime(ROOT / relative)
        if modified is not None and modified > finished:
            return True

    predecessor = STAGE_PREDECESSOR.get(stage)
    if predecessor:
        predecessor_row = last_successes.get(predecessor)
        predecessor_finished = (predecessor_row or {}).get("finished_at")
        if predecessor_finished is not None:
            if predecessor_finished.tzinfo is None:
                predecessor_finished = predecessor_finished.replace(tzinfo=UTC)
            if predecessor_finished > finished:
                return True

    if stage == "score":
        current = latest_scored_rules_version(engine)
        snapshot = success_row.get("version_snapshot") or {}
        if isinstance(snapshot, str):
            snapshot = {}
        recorded = snapshot.get("rules_version")
        if current and recorded and current != recorded:
            return True
    return False


def pipeline_state(engine: Engine | None, mode: str) -> dict[str, Any]:
    """Everything the Pipeline page needs, read in one place."""
    ready = pipeline_runs_ready(engine)
    successes = last_success_map(engine, mode)
    latest = latest_map(engine, mode)
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        success = successes.get(stage)
        last = latest.get(stage)
        status_value = str(last["status"]) if last else "never"
        stages[stage] = {
            "stage": stage,
            "status": status_value,
            "last": last,
            "success": success,
            "stale": stage_is_stale(stage, mode, success, successes, engine),
        }
    return {
        "db_ok": engine is not None,
        "runs_ready": ready,
        "mode": mode,
        "stages": stages,
        "active": active_run(engine),
        "real_labels": real_label_count(engine),
        "fixture_labels": fixture_label_count(engine),
        "counts": {
            "address_features": count_table(engine, "address_features"),
            "address_flags": count_table(engine, "address_flags"),
        },
    }


# ---------------------------------------------------------------------------
# pipeline_runs writes
# ---------------------------------------------------------------------------
def insert_running_run(
    engine: Engine,
    stage: str,
    mode: str,
    log_path: str | None,
    triggered_by: str = "ui",
) -> int:
    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                """
                INSERT INTO pipeline_runs
                    (stage, mode, status, log_path, triggered_by, version_snapshot)
                VALUES (:stage, :mode, 'running', :log_path, :triggered_by,
                        CAST(:versions AS JSONB))
                RETURNING id
                """
            ),
            {
                "stage": stage,
                "mode": mode,
                "log_path": log_path,
                "triggered_by": triggered_by,
                "versions": "{}",
            },
        ).scalar_one()
    return int(run_id)


def finish_run(
    engine: Engine,
    run_id: int,
    *,
    status: str,
    exit_code: int | None,
    row_counts: dict[str, Any] | None,
    version_snapshot: dict[str, Any] | None,
    error: str | None,
) -> None:
    import json

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE pipeline_runs
                SET status = :status,
                    finished_at = now(),
                    exit_code = :exit_code,
                    row_counts = CAST(:row_counts AS JSONB),
                    version_snapshot = CAST(:version_snapshot AS JSONB),
                    error = :error
                WHERE id = :run_id
                """
            ),
            {
                "status": status,
                "exit_code": exit_code,
                "row_counts": json.dumps(row_counts or {}, default=str),
                "version_snapshot": json.dumps(version_snapshot or {}, default=str),
                "error": error,
                "run_id": int(run_id),
            },
        )


def log_admin_action(
    engine: Engine,
    action: str,
    *,
    row_counts: dict[str, Any] | None = None,
    status: str = "success",
    error: str | None = None,
    triggered_by: str = "ui",
) -> int:
    """Record an Admin action. The action description lives in ``error``."""
    import json

    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                """
                INSERT INTO pipeline_runs
                    (stage, mode, status, finished_at, exit_code, row_counts,
                     version_snapshot, error, triggered_by)
                VALUES ('admin', 'real', :status, now(), 0,
                        CAST(:row_counts AS JSONB), '{}', :error, :triggered_by)
                RETURNING id
                """
            ),
            {
                "status": status,
                "row_counts": json.dumps(row_counts or {}, default=str),
                "error": action if error is None else f"{action}: {error}",
                "triggered_by": triggered_by,
            },
        ).scalar_one()
    return int(run_id)


# ---------------------------------------------------------------------------
# Filesystem helpers (no shell commands)
# ---------------------------------------------------------------------------
def read_log_tail(log_path: str | Path | None, lines: int = 1000) -> str | None:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=max(1, int(lines))))


def directory_usage(relative: str | Path) -> int:
    path = Path(relative)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)
