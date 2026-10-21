"""Read-only table access for the Phase 7 Data Explorer."""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Static ownership mapping. Never infer the phase from a table's columns.
TABLE_PHASE: dict[str, int] = {
    "labels": 2,
    "label_splits": 2,
    "known_addresses": 2,
    "address_features": 3,
    "feature_scaler": 3,
    "address_flags": 4,
    "risk_scores": 5,
    "score_trace": 5,
    "pipeline_runs": 7,
}
TABLE_TITLES = {
    "labels": "labels — ground-truth label rows",
    "label_splits": "label_splits — leakage-safe address split",
    "known_addresses": "known_addresses — exchange/bridge/mixer/contract curation",
    "address_features": "address_features — one row per address",
    "feature_scaler": "feature_scaler — train-only scaler provenance",
    "address_flags": "address_flags — Phase 4 detection signals",
    "risk_scores": "risk_scores — current locked-rule scores",
    "score_trace": "score_trace — immutable per-version audit traces",
    "pipeline_runs": "pipeline_runs — Phase 7 operations history",
}
PHASES = [1, 2, 3, 4, 5, 7]
DEFAULT_LIMIT = 5_000
MAX_LIMIT = 50_000
LARGE_TABLES = {"address_features", "address_flags", "score_trace", "pipeline_runs"}

_ORDER_BY = {
    "labels": "address, source, label",
    "label_splits": "address",
    "known_addresses": "category, address",
    "address_features": "tx_count_total DESC NULLS LAST, address",
    "feature_scaler": "feature_name",
    "address_flags": "address, signal_name",
    "risk_scores": "score DESC, address",
    "score_trace": "created_at DESC, address",
    "pipeline_runs": "started_at DESC, id DESC",
}


def _check_table(table: str) -> None:
    if table not in TABLE_PHASE:
        raise ValueError(f"Unknown table: {table!r}")


def fetch_rows(
    engine: Engine,
    table: str,
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> pd.DataFrame:
    """Fetch at most ``limit`` rows. Table names are whitelisted, not user input."""
    _check_table(table)
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    query = (
        f"SELECT * FROM {table} "  # noqa: S608 - table is a static whitelist key
        f"ORDER BY {_ORDER_BY[table]} LIMIT :limit OFFSET :offset"
    )
    with engine.connect() as conn:
        return pd.read_sql(
            text(query),
            conn,
            params={"limit": limit, "offset": offset},
        )


def row_count(engine: Engine, table: str) -> int | None:
    _check_table(table)
    try:
        with engine.connect() as conn:
            return int(
                conn.execute(
                    text(f"SELECT count(*) FROM {table}")  # noqa: S608
                ).scalar_one()
            )
    except Exception:  # noqa: BLE001
        return None


def search_frame(frame: pd.DataFrame, term: str) -> pd.DataFrame:
    if not term or frame.empty:
        return frame
    mask = frame.astype(str).apply(
        lambda column: column.str.contains(term, case=False, na=False, regex=False)
    )
    return frame[mask.any(axis=1)]


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


_FORBIDDEN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"MERGE|COPY|CALL|DO|VACUUM|ANALYZE|SET|RESET|LOCK|COMMENT"
    r")\b",
    re.IGNORECASE,
)


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    """Accept a single SELECT/CTE statement and nothing else.

    Comments are rejected because ``--`` and ``/* */`` are the usual way to
    hide a second statement from a naive keyword check.
    """
    stripped = (sql or "").strip()
    if not stripped:
        return False, "Enter a SELECT statement."
    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return False, "SQL comments are not allowed in the read-only box."
    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if ";" in stripped:
        return False, "Only one statement is allowed."
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "Only SELECT (or WITH ... SELECT) queries are allowed."
    match = _FORBIDDEN.search(stripped)
    if match:
        return False, f"Write/DDL keyword rejected: {match.group(0).upper()}."
    return True, ""


def run_read_only_query(
    engine: Engine,
    sql: str,
    *,
    max_rows: int = DEFAULT_LIMIT,
) -> tuple[pd.DataFrame | None, str]:
    """Run a SELECT inside an explicit read-only transaction.

    Returns ``(frame, error)``; exactly one is non-empty/None. The connection is
    always rolled back, even if the query raises.
    """
    ok, reason = validate_read_only_sql(sql)
    if not ok:
        return None, reason
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            try:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                frame = pd.read_sql(text(sql), conn)
            finally:
                transaction.rollback()
    except Exception as exc:  # noqa: BLE001
        return None, f"{exc.__class__.__name__}: {exc}"
    return frame.head(max_rows), ""


def column_config(frame: pd.DataFrame) -> dict[str, Any]:
    """Small type-aware column config for ``st.dataframe``."""
    import streamlit as st

    config: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            config[column] = st.column_config.DatetimeColumn(column, format="YYYY-MM-DD HH:mm")
        elif pd.api.types.is_bool_dtype(series):
            config[column] = st.column_config.CheckboxColumn(column)
        elif pd.api.types.is_integer_dtype(series):
            config[column] = st.column_config.NumberColumn(column, format="%d")
        elif pd.api.types.is_float_dtype(series):
            config[column] = st.column_config.NumberColumn(column, format="%.4f")
        elif column in {"row_counts", "version_snapshot", "evidence", "trace"}:
            config[column] = st.column_config.JsonColumn(column)
    return config
