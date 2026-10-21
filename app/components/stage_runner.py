"""The only Phase 7 module that launches shell commands.

Every stage invocation goes through ``scripts/run.sh <stage>`` so the
container/local switch is preserved. The runner is synchronous by design:
Streamlit blocks while the stage runs and receives output line by line.
"""

from __future__ import annotations

import os
import select
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from chainlens.db import get_pg_engine
from components import status

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data" / "logs"
STAGE_TIMEOUT_SECONDS = 15 * 60
STAGE_ROW_COUNTS: dict[str, list[str]] = {
    "sample": ["labels", "label_splits", "known_addresses"],
    "ingest": ["labels", "label_splits", "known_addresses", "data_manifest"],
    "features": ["address_features", "feature_scaler"],
    "detect": ["address_flags"],
    "score": ["risk_scores", "score_trace"],
}
VALID_MODES = ("sample", "real")

LineCallback = Callable[[str], None] | None


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _log_path(stage: str, mode: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{stage}_{mode}_{_utc_stamp()}.log"


def _success_exists(engine, stage: str, mode: str) -> bool:
    return stage in status.last_success_map(engine, mode)


def _input_path(mode: str) -> Path:
    relative = (
        "data/sample/transactions.parquet"
        if mode == "sample"
        else "data/raw/transactions.parquet"
    )
    return ROOT / relative


def prerequisite_reason(
    stage: str,
    mode: str,
    engine=None,
    *,
    force: bool = False,
    sample_confirmed: bool = False,
) -> str | None:
    """Return a human-readable blocking reason, or None when the stage may run."""
    if stage not in status.STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Safety rules are never bypassed by the force toggle.
    if stage == "sample" and mode != "sample":
        return (
            "The sample stage only runs in sample mode. Switch the mode selector "
            "to sample, or run `make ingest` for real data."
        )
    if stage == "sample":
        real_labels = status.real_label_count(engine)
        if real_labels and not sample_confirmed:
            return (
                f"labels contains {real_labels:,} real row(s). Running sample would "
                "wipe them. Switch to real mode, or confirm the warning checkbox. "
                "Undo afterwards with `make ingest` in real mode."
            )

    if force:
        return None

    if engine is None:
        return "PostgreSQL is unavailable. Start it with `make up` and apply `make schema`."

    if stage == "ingest":
        input_path = _input_path(mode)
        if not input_path.exists():
            return (
                f"Missing input {input_path.relative_to(ROOT)} for {mode} mode. "
                "Run the sample stage first, or stage the real BigQuery Parquet drop."
            )
        return None

    if stage == "features":
        clean = ROOT / "data" / "processed" / "transactions_clean.parquet"
        if not clean.exists():
            return "Missing data/processed/transactions_clean.parquet. Run Ingest first."
        if not _success_exists(engine, "ingest", mode):
            return f"No successful ingest run recorded for {mode} mode. Run Ingest first."
        return None

    if stage == "detect":
        if not _success_exists(engine, "features", mode):
            return f"No successful features run recorded for {mode} mode. Run Features first."
        count = status.count_table(engine, "address_features")
        if not count:
            return "address_features is empty. Run Features first."
        return None

    if stage == "score":
        if not _success_exists(engine, "detect", mode):
            return f"No successful detect run recorded for {mode} mode. Run Detect first."
        count = status.count_table(engine, "address_flags")
        if not count:
            return "address_flags is empty. Run Detect first."
        return None

    return None


def _emit(line: str, log_handle, on_line: LineCallback) -> None:
    text_line = line.rstrip("\r\n")
    if log_handle is not None:
        log_handle.write(text_line + "\n")
        log_handle.flush()
    if on_line is not None:
        try:
            on_line(text_line)
        except Exception:  # noqa: BLE001 - UI callback must never kill a run
            pass


def _stream_process(
    process: subprocess.Popen,
    timeout: int | float,
    log_handle,
    on_line: LineCallback,
) -> tuple[bool, int]:
    """Stream output until exit. Returns (timed_out, exit_code)."""
    started = time.monotonic()
    timed_out = False
    stream = process.stdout

    supports_select = False
    if stream is not None:
        try:
            stream.fileno()
            supports_select = True
        except (AttributeError, OSError, ValueError):
            supports_select = False

    while process.poll() is None:
        remaining = float(timeout) - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        if supports_select:
            try:
                ready, _, _ = select.select([stream], [], [], min(0.5, remaining))
            except (OSError, ValueError):
                ready = []
                time.sleep(min(0.05, remaining))
        else:
            ready = [stream] if stream is not None else []
            time.sleep(0.01)
        if not ready or stream is None:
            continue
        line = stream.readline()
        if line:
            _emit(line, log_handle, on_line)

    if timed_out:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    if stream is not None:
        try:
            for line in stream:
                _emit(line, log_handle, on_line)
        except (OSError, ValueError):
            pass
    try:
        exit_code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait()
    return timed_out, int(exit_code if exit_code is not None else -1)


def collect_row_counts(engine, stage: str) -> dict[str, int | None]:
    tables = STAGE_ROW_COUNTS.get(stage, [])
    return {table: status.count_table(engine, table) for table in tables}


def _failed_record(
    engine,
    stage: str,
    mode: str,
    reason: str,
    *,
    triggered_by: str,
) -> dict[str, Any]:
    """Log a blocked/aborted attempt as a failed pipeline_runs row."""
    run_id = None
    if engine is not None and status.pipeline_runs_ready(engine):
        try:
            run_id = status.insert_running_run(engine, stage, mode, None, triggered_by)
            status.finish_run(
                engine,
                run_id,
                status="failed",
                exit_code=None,
                row_counts={},
                version_snapshot=status.version_snapshot(engine),
                error=reason,
            )
        except Exception:  # noqa: BLE001
            run_id = None
    return {
        "blocked": True,
        "stage": stage,
        "mode": mode,
        "status": "failed",
        "run_id": run_id,
        "exit_code": None,
        "log_path": None,
        "row_counts": {},
        "version_snapshot": {},
        "error": reason,
        "elapsed_seconds": 0.0,
        "timed_out": False,
    }


def run_stage(
    stage: str,
    mode: str,
    *,
    on_line: LineCallback = None,
    timeout: int | float = STAGE_TIMEOUT_SECONDS,
    force: bool = False,
    sample_confirmed: bool = False,
    triggered_by: str = "ui",
) -> dict[str, Any]:
    """Run one stage synchronously and record exactly one pipeline_runs row.

    Returns ``{run_id, status, exit_code, log_path, row_counts,
    version_snapshot, error, elapsed_seconds, blocked}``. A blocked attempt is
    still logged (status ``failed``) so every UI action is auditable.
    """
    if stage not in status.STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}")

    try:
        engine = get_pg_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        reason = f"PostgreSQL is unavailable: {exc.__class__.__name__}"
        return _failed_record(None, stage, mode, reason, triggered_by=triggered_by)

    reason = prerequisite_reason(
        stage,
        mode,
        engine,
        force=force,
        sample_confirmed=sample_confirmed,
    )
    if reason:
        return _failed_record(engine, stage, mode, reason, triggered_by=triggered_by)

    log_path = _log_path(stage, mode)
    try:
        run_id = status.insert_running_run(engine, stage, mode, str(log_path), triggered_by)
    except Exception as exc:  # noqa: BLE001
        return {
            "blocked": True,
            "stage": stage,
            "mode": mode,
            "status": "failed",
            "run_id": None,
            "exit_code": None,
            "log_path": None,
            "row_counts": {},
            "version_snapshot": {},
            "error": f"Could not create pipeline_runs row: {exc}",
            "elapsed_seconds": 0.0,
            "timed_out": False,
        }

    env = os.environ.copy()
    if mode == "sample":
        env["CHAINLENS_SAMPLE_MODE"] = "1"
    else:
        env.pop("CHAINLENS_SAMPLE_MODE", None)

    command = [str(ROOT / "scripts" / "run.sh"), stage]
    started = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    error: str | None = None
    process = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        _emit(f"$ {' '.join(command)}", log_handle, on_line)
        _emit(f"mode={mode} CHAINLENS_SAMPLE_MODE={env.get('CHAINLENS_SAMPLE_MODE', '')}", log_handle, on_line)
        _emit(f"started_at={datetime.now(UTC).isoformat()}", log_handle, on_line)
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            timed_out, exit_code = _stream_process(process, timeout, log_handle, on_line)
            if timed_out:
                error = f"Stage timed out after {timeout} seconds and was killed."
            elif exit_code != 0:
                error = f"Stage exited with code {exit_code}. See the log for details."
        except Exception as exc:  # noqa: BLE001
            error = f"Could not start {command[0]}: {exc}"
            _emit(error, log_handle, on_line)
        finally:
            elapsed = time.monotonic() - started
            row_counts = collect_row_counts(engine, stage)
            snapshot = status.version_snapshot(engine)
            final_status = "success" if error is None and exit_code == 0 else "failed"
            if final_status == "failed" and error is None:
                error = "Stage failed without an exit code."
            try:
                status.finish_run(
                    engine,
                    run_id,
                    status=final_status,
                    exit_code=exit_code,
                    row_counts=row_counts,
                    version_snapshot=snapshot,
                    error=error,
                )
            except Exception as exc:  # noqa: BLE001
                _emit(f"Could not finalise pipeline_runs row: {exc}", log_handle, on_line)
            _emit(f"finished_at={datetime.now(UTC).isoformat()}", log_handle, on_line)
            _emit(
                f"status={final_status} exit_code={exit_code} "
                f"elapsed={elapsed:.1f}s",
                log_handle,
                on_line,
            )

    return {
        "blocked": False,
        "stage": stage,
        "mode": mode,
        "status": final_status,
        "run_id": run_id,
        "exit_code": exit_code,
        "log_path": str(log_path),
        "row_counts": row_counts,
        "version_snapshot": snapshot,
        "error": error,
        "elapsed_seconds": round(elapsed, 2),
        "timed_out": timed_out,
    }
