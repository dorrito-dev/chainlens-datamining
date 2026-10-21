"""Phase 7 Pipeline control panel.

One button per stage, live output, and DB-backed state that survives a refresh.
Every invocation goes through ``components.stage_runner.run_stage``.
"""

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

st.set_page_config(page_title="Pipeline · ChainLens", page_icon="⚙️", layout="wide")

engine = status.get_engine_or_none()
if "mode" not in st.session_state:
    st.session_state["mode"] = status.default_mode(engine)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("Pipeline")
    st.caption(
        "Run the five stages in order. State is read from `pipeline_runs`, so a "
        "browser refresh or a second tab sees the same status."
    )
with header_right:
    mode = st.radio(
        "Mode",
        options=["sample", "real"],
        index=0 if st.session_state["mode"] == "sample" else 1,
        horizontal=True,
        key="mode_selector",
    )
    st.session_state["mode"] = mode

if mode == "sample":
    st.error(
        "🔴 **SAMPLE MODE** — synthetic fixtures. Running `sample` rewrites "
        "`labels`, `label_splits` and `known_addresses`."
    )
else:
    st.warning("🟠 **REAL MODE** — real pipeline data. The sample stage is blocked.")

if engine is None:
    st.warning(
        "PostgreSQL is unavailable. Start it with `make up` and apply "
        "`make schema`; stage buttons stay disabled until then."
    )
elif not status.pipeline_runs_ready(engine):
    st.warning(
        "`pipeline_runs` is missing. Apply the Phase 7 schema with "
        "`make schema` (idempotent)."
    )

state = status.pipeline_state(engine, mode)
active = state.get("active")
real_labels = state.get("real_labels") or 0
force = st.checkbox(
    "Force run (skip prerequisite checks)",
    help="Safety rules still apply: sample mode is required for the sample stage, "
    "and real labels still need explicit confirmation.",
)

sample_confirmed = True
if mode == "sample" and real_labels:
    st.error(
        f"`labels` contains **{real_labels:,}** real row(s). Running `sample` "
        "will wipe them. Switch to real mode, or confirm below. Undo with "
        "`make ingest` in real mode."
    )
    sample_confirmed = st.checkbox(
        "I understand that running Sample will wipe the real labels in this database."
    )

# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------
st.subheader("Stage status")
status_columns = st.columns(len(status.STAGES))
for column, stage in zip(status_columns, status.STAGES, strict=True):
    stage_state = state["stages"][stage]
    success = stage_state.get("success")
    if stage_state["status"] == "failed":
        icon = "❌"
    elif stage_state["status"] == "running":
        icon = "🔄"
    elif success and stage_state["stale"]:
        icon = "🕒"
    elif success:
        icon = "✅"
    else:
        icon = "⚪"
    with column:
        with st.container(border=True):
            st.markdown(f"### {icon} {status.STAGE_TITLES[stage]}")
            if success:
                st.caption(f"Last success: {status.format_timestamp(success['finished_at'])}")
                st.caption(f"run #{success['id']}")
                if stage_state["stale"]:
                    st.caption("Stale: an input changed after this run.")
            elif stage_state["last"]:
                st.caption(f"Last attempt: {stage_state['last']['status']}")
            else:
                st.caption("Never run in this mode.")
            count = (stage_state["success"] or {}).get("row_counts") or {}
            if count:
                st.caption(" · ".join(f"{key}={value:,}" for key, value in count.items()))

if active:
    st.info(
        f"Run #{active['id']} ({active['stage']}, {active['mode']}) is still "
        "running. Buttons are disabled until it finishes. A crashed run older "
        "than 20 minutes can be marked failed on the Admin page."
    )

# ---------------------------------------------------------------------------
# Stage buttons
# ---------------------------------------------------------------------------
st.subheader("Run a stage")
button_columns = st.columns(len(status.STAGES))
clicked_stage: str | None = None
for column, stage in zip(button_columns, status.STAGES, strict=True):
    reason = stage_runner.prerequisite_reason(
        stage,
        mode,
        engine,
        force=force,
        sample_confirmed=sample_confirmed,
    )
    disabled = (
        engine is None
        or not state.get("runs_ready")
        or active is not None
        or bool(reason)
    )
    with column:
        if st.button(
            f"Run {status.STAGE_TITLES[stage]}",
            disabled=disabled,
            width="stretch",
            key=f"run_{stage}",
        ):
            clicked_stage = stage
        if reason:
            st.caption(f"🔒 {reason}")
        else:
            st.caption(status.STAGE_DESCRIPTIONS[stage])

if clicked_stage:
    lines: list[str] = []
    with st.status(
        f"Running {status.STAGE_TITLES[clicked_stage]}…",
        expanded=True,
    ) as box:

        def _stream(line: str) -> None:
            box.write(line)

        result = stage_runner.run_stage(
            clicked_stage,
            mode,
            on_line=_stream,
            force=force,
            sample_confirmed=sample_confirmed,
            triggered_by="ui",
        )
        if result["status"] == "success":
            box.update(
                label=f"{status.STAGE_TITLES[clicked_stage]} succeeded",
                state="complete",
                expanded=False,
            )
        else:
            box.update(
                label=f"{status.STAGE_TITLES[clicked_stage]} failed",
                state="error",
                expanded=True,
            )
    st.session_state["last_result"] = result
    st.rerun()

# ---------------------------------------------------------------------------
# Last result
# ---------------------------------------------------------------------------
last_result = st.session_state.get("last_result")
if last_result:
    st.subheader("Last result (this browser session)")
    result_columns = st.columns(4)
    result_columns[0].metric("Status", last_result.get("status", "—"))
    result_columns[1].metric("Run id", last_result.get("run_id") or "—")
    result_columns[2].metric("Exit code", last_result.get("exit_code"))
    result_columns[3].metric("Elapsed", f"{last_result.get('elapsed_seconds', 0):.1f}s")
    if last_result.get("error"):
        st.error(last_result["error"])
    if last_result.get("row_counts"):
        st.write("Row counts after the run")
        st.dataframe(
            pd.DataFrame(
                [
                    {"table": table, "rows": count}
                    for table, count in last_result["row_counts"].items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if last_result.get("log_path"):
        st.caption(f"Log: `{last_result['log_path']}`")
    st.page_link(
        str(APP_DIR / "pages" / "4_Logs.py"),
        label="View run history and logs →",
        icon="📜",
    )
    if last_result.get("status") == "success" and last_result.get("stage") == "score":
        st.page_link(
            str(APP_DIR / "pages" / "3_Analyst_Dashboard.py"),
            label="Score complete — open the Analyst Dashboard →",
            icon="📈",
        )
elif state.get("runs_ready"):
    latest = status.latest_map(engine, mode)
    if latest:
        st.subheader("Most recent attempts")
        rows = [
            {
                "stage": stage,
                "status": row["status"],
                "started": row["started_at"],
                "finished": row["finished_at"],
                "exit_code": row["exit_code"],
                "run_id": row["id"],
            }
            for stage, row in sorted(latest.items())
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with st.expander("What the stages do"):
    for stage in status.STAGES:
        st.markdown(f"**{status.STAGE_TITLES[stage]}** — {status.STAGE_DESCRIPTIONS[stage]}")
    st.caption(
        "Every run goes through `scripts/run.sh <stage>`, so the container/local "
        "switch is preserved. Timeout: 15 minutes per stage."
    )
