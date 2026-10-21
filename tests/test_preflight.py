from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PREFLIGHT_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "00_preflight.py"
_spec = importlib.util.spec_from_file_location("chainlens_preflight", PREFLIGHT_PATH)
assert _spec and _spec.loader
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)


class FakeClient:
    """Minimal BigQuery client that only answers a dry-run job."""

    def __init__(self, bytes_processed: int) -> None:
        self.bytes_processed = bytes_processed
        self.calls = 0

    def query(self, sql: str, job_config: object | None = None) -> SimpleNamespace:
        assert sql
        assert getattr(job_config, "dry_run", False) is True
        assert getattr(job_config, "use_query_cache", True) is False
        self.calls += 1
        return SimpleNamespace(total_bytes_processed=self.bytes_processed)


def job_config() -> SimpleNamespace:
    return SimpleNamespace(dry_run=True, use_query_cache=False)


def test_dry_run_returns_bytes_and_cost_without_scanning():
    gib = 1024**3
    client = FakeClient(gib)
    recorded: list[dict] = []
    code = preflight.execute_preflight(
        sql="SELECT 1",
        client=client,
        job_config=job_config(),
        max_bytes=2**40,
        block_start=1,
        block_end=2,
        record=lambda **kwargs: recorded.append(kwargs),
    )
    assert code == 0
    assert client.calls == 1
    assert recorded[0]["bytes_processed"] == gib
    assert abs(preflight.estimate_usd(gib) - 6.25 / 1024) < 1e-12


def test_refuses_above_cap_and_writes_nothing():
    client = FakeClient(2 * 2**40)
    recorded: list[dict] = []
    code = preflight.execute_preflight(
        sql="SELECT 1",
        client=client,
        job_config=job_config(),
        max_bytes=2**40,
        block_start=1,
        block_end=2,
        record=lambda **kwargs: recorded.append(kwargs),
    )
    assert code == 2
    assert recorded == []


def test_parse_bytes_units():
    assert preflight.parse_bytes("1TiB") == 2**40
    assert preflight.parse_bytes("512GiB") == 512 * 2**30
    assert preflight.parse_bytes(123) == 123
