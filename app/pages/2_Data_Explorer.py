"""Phase 7 Data Explorer — browse every phase's tables, read-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
for _candidate in (str(ROOT), str(APP_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import streamlit as st  # noqa: E402
from components import status, table_browser  # noqa: E402

st.set_page_config(page_title="Data Explorer · ChainLens", page_icon="🔍", layout="wide")

st.title("Data Explorer")
st.caption(
    "Browse the tables produced by every phase. This page is read-only: there "
    "is no button that writes to the database."
)

engine = status.get_engine_or_none()
if engine is None:
    st.error(
        "PostgreSQL is unavailable. Start it with `make up`, apply `make schema`, "
        "and reload."
    )
    st.stop()

mode = status.default_mode(engine)
fixture_rows = status.fixture_label_count(engine) or 0
if mode == "sample" or fixture_rows:
    st.caption("🏷️ **Fixture source** — this database contains synthetic sample rows.")

phase_options = ["All", *[f"P{p}" for p in table_browser.PHASES]]
selected_phase = st.selectbox("Phase filter", phase_options, index=0)

available_tables = [
    table
    for table, phase in table_browser.TABLE_PHASE.items()
    if selected_phase == "All" or selected_phase == f"P{phase}"
]
if not available_tables:
    st.info("No tables for this phase.")
    st.stop()

table = st.selectbox(
    "Table",
    available_tables,
    format_func=lambda name: table_browser.TABLE_TITLES.get(name, name),
)

limit_key = f"explorer_limit_{table}"
offset_key = f"explorer_offset_{table}"
st.session_state.setdefault(limit_key, table_browser.DEFAULT_LIMIT)
st.session_state.setdefault(offset_key, 0)

try:
    total_rows = table_browser.row_count(engine, table)
    frame = table_browser.fetch_rows(
        engine,
        table,
        limit=st.session_state[limit_key],
        offset=st.session_state[offset_key],
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read {table}: {exc.__class__.__name__}")
    st.stop()

filter_column, visibility_column = st.columns([2, 2])
with filter_column:
    term = st.text_input("Filter rows", placeholder="Substring search across all columns")
with visibility_column:
    selected_columns = st.multiselect(
        "Visible columns",
        options=list(frame.columns),
        default=list(frame.columns),
        key=f"columns_{table}",
    )

if selected_columns:
    frame = frame[selected_columns]

if term:
    frame = table_browser.search_frame(frame, term)

st.caption(
    f"{len(frame):,} row(s) shown · table has "
    f"{total_rows:,} row(s)"
    if total_rows is not None
    else f"{len(frame):,} row(s) shown"
)
if total_rows is not None and total_rows > len(frame):
    st.caption(
        f"Showing at most {st.session_state[limit_key]:,} rows. Use Show more to "
        "page through the table."
    )

if frame.empty:
    st.info("No rows match the current filter.")
else:
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=table_browser.column_config(frame),
    )

download_column, more_column, reset_column = st.columns([2, 1, 1])
with download_column:
    st.download_button(
        "Download CSV",
        data=table_browser.to_csv_bytes(frame),
        file_name=f"{table}.csv",
        mime="text/csv",
        width="stretch",
    )
with more_column:
    if st.button("Show more", width="stretch", disabled=total_rows is None or total_rows <= len(frame)):
        st.session_state[limit_key] = min(
            st.session_state[limit_key] + table_browser.DEFAULT_LIMIT,
            table_browser.MAX_LIMIT,
        )
        st.rerun()
with reset_column:
    if st.button("Reset paging", width="stretch"):
        st.session_state[limit_key] = table_browser.DEFAULT_LIMIT
        st.session_state[offset_key] = 0
        st.rerun()

# ---------------------------------------------------------------------------
# JSONB trace viewer
# ---------------------------------------------------------------------------
if table == "score_trace" and not frame.empty and "address" in frame.columns:
    with st.expander("Inspect a score_trace JSON document", expanded=False):
        address = st.selectbox("Address", options=list(frame["address"]), key="trace_address")
        selected = frame[frame["address"] == address].iloc[0]
        trace = selected.get("trace")
        if isinstance(trace, str):
            try:
                trace = json.loads(trace)
            except json.JSONDecodeError:
                pass
        st.json(trace)

if table == "pipeline_runs" and not frame.empty:
    st.caption(
        "Run logs are rendered on the Logs page. `row_counts` and "
        "`version_snapshot` are JSONB."
    )

# ---------------------------------------------------------------------------
# Read-only SQL
# ---------------------------------------------------------------------------
with st.expander("Read-only SQL", expanded=False):
    st.caption(
        "Only one SELECT/CTE statement is accepted. It runs inside "
        "`SET TRANSACTION READ ONLY` and the transaction is rolled back. "
        "Comments and write/DDL keywords are rejected."
    )
    sql = st.text_area(
        "SQL",
        value="SELECT stage, status, count(*) AS runs\nFROM pipeline_runs\nGROUP BY stage, status\nORDER BY stage, status;",
        height=140,
    )
    if st.button("Run read-only query"):
        frame_sql, error = table_browser.run_read_only_query(engine, sql)
        if error:
            st.error(error)
        elif frame_sql is not None:
            st.caption(f"{len(frame_sql):,} row(s) returned (capped at 5,000).")
            st.dataframe(frame_sql, width="stretch", hide_index=True)
