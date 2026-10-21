"""Phase 7 Logs — pipeline run history, rendered logs and one-click rerun."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
for _candidate in (str(ROOT), str(APP_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import stage_runner, status  # noqa: E402

st.set_page_config(page_title="Logs · ChainLens", page_icon="📜", layout="wide")

STATUS_COLORS = {
    "success": "#d4edda",
    "failed": "#f8d7da",
    "running": "#fff3cd",
}


def _status_style(value: object) -> str:
    color = STATUS_COLORS.get(str(value), "")
    return f"background-color: {color};" if color else ""


st.title("Logs")
st.caption(
    "Last 50 stage invocations, newest first. Select a row to read the log tail "
    "(1,000 lines). Rerun writes a new row; history is never modified."
)

engine = status.get_engine_or_none()
if engine is None:
    st.error("PostgreSQL is unavailable. Start it with `make up` and reload.")
    st.stop()

runs = status.recent_runs(engine, limit=50)
if runs.empty:
    st.info("No runs recorded yet. Open the Pipeline page and run a stage.")
    st.stop()

display = runs[
    ["id", "stage", "mode", "status", "started_at", "finished_at", "exit_code", "triggered_by"]
].copy()
st.dataframe(
    display.style.map(_status_style, subset=["status"]),
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="logs_table",
)

event = st.session_state.get("logs_table")
selected_rows = []
if isinstance(event, dict):
    selected_rows = event.get("selection", {}).get("rows", [])
else:
    selected = getattr(event, "selection", None)
    selected_rows = list(getattr(selected, "rows", []) or [])

run_ids = [int(value) for value in runs["id"].tolist()]
if selected_rows:
    clicked_id = int(runs.iloc[selected_rows[0]]["id"])
    if clicked_id in run_ids:
        st.session_state["logs_run_select"] = clicked_id
if st.session_state.get("logs_run_select") not in run_ids:
    st.session_state["logs_run_select"] = run_ids[0]

selected_id = st.selectbox(
    "Open run",
    options=run_ids,
    format_func=lambda run_id: (
        f"#{run_id} · "
        f"{runs.loc[runs['id'] == run_id, 'stage'].iloc[0]} · "
        f"{runs.loc[runs['id'] == run_id, 'mode'].iloc[0]} · "
        f"{runs.loc[runs['id'] == run_id, 'status'].iloc[0]}"
    ),
    key="logs_run_select",
)
row = runs[runs["id"] == selected_id].iloc[0]

meta_columns = st.columns(4)
meta_columns[0].metric("Stage", str(row["stage"]))
meta_columns[1].metric("Mode", str(row["mode"]))
meta_columns[2].metric("Status", str(row["status"]))
meta_columns[3].metric("Exit code", row["exit_code"] if pd.notna(row["exit_code"]) else "—")
st.caption(
    f"run #{int(row['id'])} · started {status.format_timestamp(row['started_at'])} · "
    f"finished {status.format_timestamp(row['finished_at'])} · "
    f"triggered by {row['triggered_by']}"
)

if row["row_counts"]:
    st.write("Row counts")
    st.dataframe(
        pd.DataFrame(
            [{"table": key, "rows": value} for key, value in dict(row["row_counts"]).items()]
        ),
        width="stretch",
        hide_index=True,
    )
if row["version_snapshot"]:
    st.write("Version snapshot")
    st.json(dict(row["version_snapshot"]))
if row["error"]:
    st.error(str(row["error"]))

log_text = status.read_log_tail(row["log_path"], lines=1000)
if log_text is None:
    st.warning("Log file missing (stage may have run before Phase 7).")
else:
    if row["status"] == "success":
        st.success("Command exited successfully.")
    elif row["status"] == "failed":
        st.error("Command failed.")
    else:
        st.warning("Command is still running.")
    st.code(log_text, language="text")

st.divider()
rerun_columns = st.columns([1, 3])
with rerun_columns[0]:
    can_rerun = str(row["stage"]) in status.STAGES
    rerun = st.button(
        "Rerun this stage",
        disabled=not can_rerun,
        width="stretch",
    )
with rerun_columns[1]:
    if not can_rerun:
        st.caption("Admin actions are audit records and cannot be rerun.")
    else:
        st.caption(
            f"Reruns `{row['stage']}` in {row['mode']} mode through "
            "`scripts/run.sh` and writes a new `pipeline_runs` row."
        )

previous_rerun = st.session_state.pop("logs_rerun_result", None)
if previous_rerun:
    if previous_rerun["status"] == "success":
        st.success(f"Rerun #{previous_rerun['run_id']} succeeded.")
    else:
        st.error(f"Rerun #{previous_rerun['run_id']} failed: {previous_rerun.get('error')}")

if rerun:
    with st.status(f"Rerunning {row['stage']}…", expanded=True) as box:

        def _stream(line: str) -> None:
            box.write(line)

        result = stage_runner.run_stage(
            str(row["stage"]),
            str(row["mode"]),
            on_line=_stream,
            triggered_by="ui-rerun",
        )
        if result["status"] == "success":
            box.update(label="Rerun succeeded", state="complete", expanded=False)
        else:
            box.update(label="Rerun failed", state="error", expanded=True)
    st.session_state["logs_rerun_result"] = result
    st.rerun()
