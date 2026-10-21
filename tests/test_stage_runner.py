"""Stage runner tests. No real pipeline, Postgres, or ``make`` command runs."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from components import stage_runner, status


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        return self


class FakeEngine:
    def connect(self):
        return FakeConnection()


class FakeProcess:
    def __init__(self, *, returncode: int = 0, output: str = "", never_exit: bool = False):
        self._returncode = returncode
        self._never_exit = never_exit
        self.stdout = io.StringIO(output)
        self.killed = False

    def poll(self):
        return None if self._never_exit else self._returncode

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        self.killed = True
        self._never_exit = False


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """Patch paths and DB helpers so the runner never touches real infra."""
    monkeypatch.setattr(stage_runner, "ROOT", tmp_path)
    monkeypatch.setattr(stage_runner, "LOG_DIR", tmp_path / "data" / "logs")
    monkeypatch.setattr(stage_runner, "get_pg_engine", lambda: FakeEngine())
    monkeypatch.setattr(status, "pipeline_runs_ready", lambda engine: True)
    monkeypatch.setattr(status, "real_label_count", lambda engine: 0)
    monkeypatch.setattr(status, "count_table", lambda engine, table: 3)
    monkeypatch.setattr(status, "version_snapshot", lambda engine: {"rules_version": "test"})

    inserted: list[dict] = []
    finished: list[dict] = []

    def insert(engine, stage, mode, log_path, triggered_by="ui"):
        inserted.append(
            {
                "stage": stage,
                "mode": mode,
                "log_path": log_path,
                "triggered_by": triggered_by,
            }
        )
        return len(inserted)

    def finish(engine, run_id, **kwargs):
        finished.append({"run_id": run_id, **kwargs})

    monkeypatch.setattr(status, "insert_running_run", insert)
    monkeypatch.setattr(status, "finish_run", finish)
    return {"inserted": inserted, "finished": finished, "root": tmp_path}


def test_features_blocked_when_clean_parquet_missing(runner_env, monkeypatch):
    def fail_popen(*args, **kwargs):
        raise AssertionError("subprocess must not start when prerequisites are unmet")

    monkeypatch.setattr(stage_runner.subprocess, "Popen", fail_popen)

    result = stage_runner.run_stage("features", "sample")

    assert result["blocked"] is True
    assert result["status"] == "failed"
    assert "transactions_clean.parquet" in result["error"]
    assert len(runner_env["inserted"]) == 1
    assert len(runner_env["finished"]) == 1
    assert runner_env["finished"][0]["status"] == "failed"
    assert "transactions_clean.parquet" in runner_env["finished"][0]["error"]


def test_command_is_scripts_run_sh_and_one_row_per_invocation(runner_env, monkeypatch):
    monkeypatch.setattr(stage_runner, "prerequisite_reason", lambda *a, **k: None)
    commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return FakeProcess(returncode=0, output="hello\n")

    monkeypatch.setattr(stage_runner.subprocess, "Popen", fake_popen)

    first = stage_runner.run_stage("score", "sample")
    second = stage_runner.run_stage("score", "sample")

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert len(runner_env["inserted"]) == 2
    assert len(runner_env["finished"]) == 2
    assert all(item["status"] == "success" for item in runner_env["finished"])
    assert commands[0][-1] == "score"
    assert Path(commands[0][0]).parts[-2:] == ("scripts", "run.sh")
    assert "make" not in commands[0]
    assert Path(first["log_path"]).read_text(encoding="utf-8").count("hello") == 1


def test_failed_exit_code_marks_row_failed(runner_env, monkeypatch):
    monkeypatch.setattr(stage_runner, "prerequisite_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        stage_runner.subprocess,
        "Popen",
        lambda *a, **k: FakeProcess(returncode=2, output="boom\n"),
    )

    result = stage_runner.run_stage("detect", "sample")

    assert result["status"] == "failed"
    assert result["exit_code"] == 2
    assert "exited with code 2" in result["error"]
    assert runner_env["finished"][0]["status"] == "failed"


def test_timeout_kills_process_and_marks_failed(runner_env, monkeypatch):
    monkeypatch.setattr(stage_runner, "prerequisite_reason", lambda *a, **k: None)
    process = FakeProcess(never_exit=True)
    monkeypatch.setattr(stage_runner.subprocess, "Popen", lambda *a, **k: process)

    result = stage_runner.run_stage("score", "sample", timeout=0.05)

    assert result["status"] == "failed"
    assert result["timed_out"] is True
    assert "timed out" in result["error"].lower()
    assert process.killed is True


def test_sample_mode_env_is_set_and_real_mode_is_unset(runner_env, monkeypatch, tmp_path):
    monkeypatch.setattr(stage_runner, "prerequisite_reason", lambda *a, **k: None)
    seen: list[dict] = []

    def fake_popen(command, **kwargs):
        seen.append(dict(kwargs["env"]))
        return FakeProcess(returncode=0)

    monkeypatch.setattr(stage_runner.subprocess, "Popen", fake_popen)

    stage_runner.run_stage("score", "sample")
    assert seen[0]["CHAINLENS_SAMPLE_MODE"] == "1"

    monkeypatch.setenv("CHAINLENS_SAMPLE_MODE", "1")
    stage_runner.run_stage("score", "real")
    assert "CHAINLENS_SAMPLE_MODE" not in seen[1]


def test_sample_real_mode_is_never_allowed(runner_env):
    result = stage_runner.run_stage("sample", "real")
    assert result["blocked"] is True
    assert "only runs in sample mode" in result["error"]
