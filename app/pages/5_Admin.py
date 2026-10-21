"""Phase 7 Admin — health, versions, disk usage and guarded cleanup."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
for _candidate in (str(ROOT), str(APP_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import admin_actions, status, table_browser  # noqa: E402
from sqlalchemy import text  # noqa: E402

from chainlens.manifest import sha256_file  # noqa: E402

st.set_page_config(page_title="Admin · ChainLens", page_icon="🛠️", layout="wide")

REQUIRED_TABLES = [*table_browser.TABLE_PHASE, "feature_scaler_dropped", "data_manifest"]

st.title("Admin")
st.caption(
    "Health, version provenance, disk usage and guarded cleanup. Destructive "
    "actions require a typed confirmation and are logged to `pipeline_runs`."
)

engine = status.get_engine_or_none()
if engine is None:
    st.error("PostgreSQL is unavailable. Start it with `make up` and reload.")
    st.stop()

mode = status.default_mode(engine)
if mode == "sample":
    st.error("🔴 **SAMPLE MODE** — this database contains synthetic fixture data.")
else:
    st.warning("🟠 **REAL MODE** — this database contains real pipeline data.")

running = status.running_count(engine)
lockout = running > 0
if lockout:
    st.warning(
        f"{running} run(s) are marked running. Destructive actions are disabled; "
        "mark crashed runs failed below if they are older than 20 minutes."
    )

admin_message = st.session_state.pop("admin_message", None)
if admin_message:
    level, message = admin_message
    getattr(st, level)(message)

# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
with st.expander("Health check", expanded=True):
    checks: list[dict[str, object]] = []
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks.append({"check": "PostgreSQL", "status": "✅", "detail": "SELECT 1 OK"})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "PostgreSQL", "status": "❌", "detail": str(exc)})

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", ""),
            ),
            connection_timeout=1.0,
        )
        try:
            driver.verify_connectivity()
            checks.append(
                {"check": "Neo4j (optional)", "status": "✅", "detail": "reachable"}
            )
        finally:
            driver.close()
    except Exception:  # noqa: BLE001
        checks.append(
            {
                "check": "Neo4j (optional)",
                "status": "⚠️",
                "detail": "not reachable — all pages work without it",
            }
        )

    for table in REQUIRED_TABLES:
        exists = False
        try:
            with engine.connect() as conn:
                exists = bool(
                    conn.execute(
                        text(
                            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                            "WHERE table_schema = current_schema() "
                            "AND table_name = :table)"
                        ),
                        {"table": table},
                    ).scalar()
                )
        except Exception:  # noqa: BLE001
            exists = False
        checks.append(
            {
                "check": f"table {table}",
                "status": "✅" if exists else "❌",
                "detail": "present" if exists else "missing — run `make schema`",
            }
        )

    tx_count_in = None
    try:
        with engine.connect() as conn:
            tx_count_in = conn.execute(
                text(
                    "SELECT count(*) FROM feature_scaler "
                    "WHERE feature_name = 'tx_count_in'"
                )
            ).scalar_one()
    except Exception:  # noqa: BLE001
        pass
    checks.append(
        {
            "check": "feature_scaler.tx_count_in",
            "status": "✅" if tx_count_in else "❌",
            "detail": (
                "present"
                if tx_count_in
                else "missing — re-run Features, or the dataset dropped it as zero-variance"
            ),
        }
    )
    checks.append(
        {
            "check": "pipeline_runs",
            "status": "✅" if status.pipeline_runs_ready(engine) else "❌",
            "detail": "Phase 7 operational table",
        }
    )
    st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 2. Version chain + ablation state
# ---------------------------------------------------------------------------
with st.expander("Version chain and ablation state", expanded=False):
    snapshot = status.version_snapshot(engine)
    versions: list[dict[str, object]] = [
        {"field": "rules_version", "value": snapshot.get("rules_version") or "—"},
        {
            "field": "rules_version_label",
            "value": snapshot.get("rules_version_label") or "—",
        },
        {"field": "scaler_version", "value": snapshot.get("scaler_version") or "—"},
        {"field": "engine_version", "value": snapshot.get("engine_version") or "—"},
    ]
    try:
        with engine.connect() as conn:
            scaler_versions = conn.execute(
                text("SELECT DISTINCT scaler_version FROM feature_scaler ORDER BY 1")
            ).scalars().all()
            rules_versions = conn.execute(
                text("SELECT DISTINCT rules_version FROM risk_scores ORDER BY 1")
            ).scalars().all()
        versions.append(
            {"field": "scaler versions in DB", "value": ", ".join(map(str, scaler_versions)) or "—"}
        )
        versions.append(
            {"field": "rules versions in DB", "value": ", ".join(map(str, rules_versions)) or "—"}
        )
    except Exception:  # noqa: BLE001
        pass

    baseline_summary = None
    baseline_files = sorted((ROOT / "data" / "processed").glob("score_baseline_*.summary.json"))
    if baseline_files:
        try:
            baseline_summary = json.loads(baseline_files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            baseline_summary = None
    versions.append(
        {
            "field": "baseline hash",
            "value": (baseline_summary or {}).get("content_sha256", "—"),
        }
    )
    versions.append(
        {
            "field": "baseline file",
            "value": baseline_files[-1].name if baseline_files else "—",
        }
    )
    st.dataframe(pd.DataFrame(versions), width="stretch", hide_index=True)

    rules_path = ROOT / "config" / "rules.yaml"
    current_hash = sha256_file(rules_path) if rules_path.exists() else None
    recorded_hash = None
    report_path = ROOT / "data" / "processed" / "ablation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            details = report.get("details", {})
            if isinstance(details, dict):
                recorded_hash = details.get("rules_sha256_before")
        except (OSError, json.JSONDecodeError):
            recorded_hash = None

    if current_hash and recorded_hash and current_hash != recorded_hash:
        st.error(
            "`config/rules.yaml` has been modified since the last ablation baseline "
            f"(`{current_hash[:12]}` vs recorded `{str(recorded_hash)[:12]}`). "
            "Re-run `make score` and `make baseline`, or restore the file from git. "
            "This page deliberately does not offer a restore button."
        )
    elif current_hash and recorded_hash:
        st.success("`config/rules.yaml` matches the last ablation baseline hash.")
    else:
        st.info(
            "No ablation hash recorded. Run `make baseline` and `make ablate` to "
            "establish the comparison."
        )

# ---------------------------------------------------------------------------
# 3. Disk usage
# ---------------------------------------------------------------------------
with st.expander("Disk usage", expanded=False):
    usage_rows = [
        {
            "path": relative,
            "size": status.format_bytes(status.directory_usage(relative)),
        }
        for relative in ("data/raw", "data/processed", "data/sample", "data/logs")
    ]
    st.dataframe(pd.DataFrame(usage_rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 4. Danger zone
# ---------------------------------------------------------------------------
st.subheader("Danger zone")
st.caption(
    "All actions below run in a Postgres transaction and are logged to "
    "`pipeline_runs` as `stage='admin'`. Type the exact word to enable a button."
)


def _confirm_box(key: str, token: str, label: str) -> bool:
    typed = st.text_input(label, key=f"confirm_{key}", placeholder=token)
    return admin_actions.confirmation_matches(typed, token)


def _run_danger(key: str, action: str, callable_, confirmation: bool) -> None:
    if st.button(
        action,
        type="primary",
        disabled=(not confirmation or lockout),
        key=f"button_{key}",
    ):
        try:
            counts = callable_(engine)
        except Exception as exc:  # noqa: BLE001
            st.session_state["admin_message"] = (
                "error",
                f"{action} failed and was rolled back: {exc.__class__.__name__}: {exc}",
            )
        else:
            try:
                status.log_admin_action(engine, action, row_counts=counts)
            except Exception as exc:  # noqa: BLE001
                st.session_state["admin_message"] = (
                    "warning",
                    f"{action} completed, but the audit row could not be written: {exc}",
                )
            else:
                st.session_state["admin_message"] = (
                    "success",
                    f"{action} completed and was logged to `pipeline_runs`.",
                )
        st.rerun()


with st.expander("Delete all pipeline data", expanded=False):
    st.markdown(
        "**Deletes**: `address_features`, `address_flags`, `risk_scores`, "
        "`score_trace`, `feature_scaler`, `pipeline_runs`, and `data/processed/*`.\n\n"
        "**Does not touch**: `labels`, `label_splits`, `known_addresses`."
    )
    confirmed = _confirm_box("delete", "DELETE", "Type DELETE to confirm")
    _run_danger(
        "delete",
        "Delete all pipeline data",
        admin_actions.delete_all_pipeline_data,
        confirmed,
    )
    st.caption("After this, re-run Features → Detect → Score.")

with st.expander("Wipe labels and known addresses", expanded=False):
    st.markdown(
        "**Deletes**: `labels`, `label_splits`, `known_addresses`. Use this before "
        "switching a database from sample fixtures to a real ingest."
    )
    confirmed = _confirm_box("reset", "RESET", "Type RESET to confirm")
    _run_danger(
        "reset",
        "Wipe labels and known addresses",
        admin_actions.wipe_labels,
        confirmed,
    )

with st.expander("Drop fixture labels", expanded=False):
    st.markdown(
        "Deletes only `labels WHERE source='fixture'` and the split rows for "
        "addresses that have no non-fixture label. Real labels are preserved."
    )
    confirmed = _confirm_box("fixture", "FIXTURE", "Type FIXTURE to confirm")
    _run_danger(
        "fixture",
        "Drop fixture labels",
        admin_actions.drop_fixture_labels,
        confirmed,
    )

with st.expander("Mark crashed runs failed", expanded=False):
    st.caption(
        "A run older than 20 minutes that is still marked running is treated as "
        "crashed. This updates it to failed; no pipeline data is deleted."
    )
    if st.button("Mark stale running runs failed", disabled=not lockout):
        try:
            changed = status.mark_stale_runs_failed(engine)
            status.log_admin_action(
                engine,
                "Mark stale running runs failed",
                row_counts={"runs_marked_failed": changed},
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["admin_message"] = (
                "error",
                f"Could not mark stale runs failed: {exc.__class__.__name__}: {exc}",
            )
        else:
            st.session_state["admin_message"] = (
                "success",
                f"Marked {changed} stale running row(s) as failed.",
            )
        st.rerun()
