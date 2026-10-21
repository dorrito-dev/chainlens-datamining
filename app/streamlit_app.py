"""ChainLens operations console — landing page.

Read-only summary of the pipeline state plus links to the five operations
pages. Nothing on this page writes to the database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _candidate in (str(ROOT), str(APP_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from components import status  # noqa: E402

st.set_page_config(page_title="ChainLens · Operations", page_icon="🔗", layout="wide")

st.title("ChainLens")
st.caption(
    "A wallet risk-scoring pipeline for Ethereum. This UI is the operations "
    "console; the analyst dashboard is one of the pages in the sidebar."
)

engine = status.get_engine_or_none()
mode = status.default_mode(engine)
snapshot = status.version_snapshot(engine)

if mode == "sample":
    st.error(
        "**SAMPLE MODE** — the current pipeline state is backed by synthetic "
        "fixtures. Sample data is never a real risk result."
    )
else:
    st.warning("**REAL MODE** — real pipeline data is loaded.")

st.subheader("Go to")
link_columns = st.columns(3)
with link_columns[0]:
    st.page_link(
        str(APP_DIR / "pages" / "1_Pipeline.py"), label="⚙️ Run pipeline", icon="⚙️"
    )
    st.caption("Run stages in order and watch live output.")
with link_columns[1]:
    st.page_link(
        str(APP_DIR / "pages" / "2_Data_Explorer.py"), label="🔍 Browse data", icon="🔍"
    )
    st.caption("Inspect every table, download CSV, run read-only SQL.")
with link_columns[2]:
    st.page_link(
        str(APP_DIR / "pages" / "3_Analyst_Dashboard.py"),
        label="📈 Analyst view",
        icon="📈",
    )
    st.caption("Ranked wallets, wallet breakdown, and graph.")

st.subheader("Current state")
if engine is None:
    st.warning(
        "PostgreSQL is unavailable. Start it with `make up`, then apply "
        "`make schema`. The pages still render; stage buttons stay disabled."
    )
    st.stop()

successes = status.last_success_map(engine, mode)
score_run = successes.get("score")
addresses_scored = status.count_table(engine, "risk_scores")
tier_frame = pd.DataFrame(columns=["tier", "count"])
try:
    with engine.connect() as conn:
        tier_frame = pd.read_sql(
            text(
                "SELECT tier, count(*) AS count FROM risk_scores "
                "GROUP BY tier ORDER BY tier"
            ),
            conn,
        )
except Exception:  # noqa: BLE001
    pass

state_columns = st.columns(4)
state_columns[0].metric("Mode", mode)
state_columns[1].metric("Addresses scored", f"{addresses_scored:,}" if addresses_scored is not None else "—")
state_columns[2].metric(
    "Last score run",
    status.format_timestamp(score_run["finished_at"]) if score_run else "never",
)
state_columns[3].metric("rules_version", snapshot.get("rules_version") or "—")

if not tier_frame.empty:
    st.caption(
        "Tier distribution: "
        + " · ".join(f"{row.tier} {int(row.count)}" for row in tier_frame.itertuples())
    )
else:
    st.caption("No scores yet — open Pipeline and run Detect, then Score.")

st.divider()
st.subheader("Recent runs")
recent = status.recent_runs(engine, limit=8)
if recent.empty:
    st.info(
        "No runs recorded yet. Open the Pipeline page and run the stages, or "
        "apply the Phase 7 schema with `make schema` if this is an older database."
    )
else:
    display = recent[
        ["id", "stage", "mode", "status", "started_at", "finished_at", "exit_code"]
    ].copy()
    st.dataframe(display, width="stretch", hide_index=True)

st.caption(
    "The UI never runs ablation, never runs notebooks, and never edits "
    "`config/rules.yaml`. Neo4j remains optional."
)
