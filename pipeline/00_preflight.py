#!/usr/bin/env python3
"""BigQuery pre-flight cost gate.

Runs the export SQL as a dry run (no table scan), prints bytes processed, the
estimated USD cost at $6.25/TiB, and the remaining 1 TiB free-tier headroom.
If the estimate exceeds BQ_MAX_BYTES the script exits non-zero and deliberately
writes no manifest row.

Requires the optional ``google-cloud-bigquery`` package and credentials. The
runtime pipeline (Phases 3-6) never imports this module.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.config import load_data_config, project_path  # noqa: E402
from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.manifest import record_manifest, sha256_file  # noqa: E402

TIB = 2**40
FREE_TIER_BYTES = TIB
DEFAULT_PRICE_PER_TIB_USD = 6.25


class DryRunJob(Protocol):
    total_bytes_processed: int | None


class DryRunClient(Protocol):
    def query(self, sql: str, job_config: object | None = None) -> DryRunJob: ...


def parse_bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    raw = str(value).strip().replace("_", "")
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    lowered = raw.lower()
    for suffix in sorted(units, key=len, reverse=True):
        if lowered.endswith(suffix):
            number = lowered[: -len(suffix)].strip()
            return int(float(number) * units[suffix])
    return int(raw)


def estimate_usd(bytes_processed: int, price_per_tib: float = DEFAULT_PRICE_PER_TIB_USD) -> float:
    return bytes_processed / TIB * price_per_tib


def run_dry_run(
    client: DryRunClient,
    sql: str,
    job_config: object,
) -> int:
    """Execute a BigQuery dry run and return total_bytes_processed."""
    job = client.query(sql, job_config=job_config)
    if job.total_bytes_processed is None:
        raise RuntimeError(
            "BigQuery dry run returned no byte estimate; refusing to guess a cost"
        )
    return int(job.total_bytes_processed)


def load_labels_for_params() -> list[str]:
    path = project_path(load_data_config()["ingest"]["labels_parquet"])
    if not path.exists():
        return []
    import pandas as pd

    labels = pd.read_parquet(path)
    return labels["address"].drop_duplicates().tolist()


def build_job_config(project: str, window: dict, labels: list[str]) -> object:
    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=[
            bigquery.ScalarQueryParameter("block_start", "INT64", window["block_start"]),
            bigquery.ScalarQueryParameter("block_end", "INT64", window["block_end"]),
            bigquery.ScalarQueryParameter(
                "ref_block_start", "INT64", window["reference_block_start"]
            ),
            bigquery.ScalarQueryParameter(
                "ref_block_end", "INT64", window["reference_block_end"]
            ),
            bigquery.ArrayQueryParameter("label_addresses", "STRING", labels),
        ],
    )


def record_dry_run(
    sql_path: str,
    sql: str,
    bytes_processed: int,
    max_bytes: int,
    project: str,
    block_start: int,
    block_end: int,
) -> None:
    engine = get_pg_engine()
    apply_schema(engine)
    cost = estimate_usd(bytes_processed)
    metadata = {
        "stage": "bigquery_dry_run",
        "dry_run": True,
        "use_query_cache": False,
        "project": project,
        "sql_path": sql_path,
        "cap_bytes": max_bytes,
        "free_tier_bytes": FREE_TIER_BYTES,
        "free_tier_headroom_bytes": max(FREE_TIER_BYTES - bytes_processed, 0),
        "price_per_tib_usd": DEFAULT_PRICE_PER_TIB_USD,
        "within_cap": bytes_processed <= max_bytes,
    }
    record_manifest(
        engine,
        f"bigquery://dry-run/{sha256_file(project_path(sql_path))[:16]}",
        sha256_file(project_path(sql_path)),
        block_start=block_start,
        block_end=block_end,
        bytes_processed=bytes_processed,
        estimated_usd=round(cost, 6),
        metadata=metadata,
    )


def execute_preflight(
    *,
    sql: str,
    client: DryRunClient,
    job_config: object,
    max_bytes: int,
    block_start: int,
    block_end: int,
    sql_path: str = "pipeline/bigquery_export.sql",
    project: str = "",
    record: Callable[..., None] | None = None,
) -> int:
    bytes_processed = run_dry_run(client, sql, job_config)
    cost = estimate_usd(bytes_processed)
    headroom = FREE_TIER_BYTES - bytes_processed
    print(f"Dry-run bytes processed: {bytes_processed:,} ({bytes_processed / TIB:.6f} TiB)")
    print(f"Estimated cost at $6.25/TiB: ${cost:,.6f}")
    print(f"Free-tier headroom remaining: {headroom:,} bytes (1 TiB/month)")
    print(f"Cap: {max_bytes:,} bytes")

    if bytes_processed > max_bytes:
        print(
            "REFUSED: estimate exceeds BQ_MAX_BYTES. No manifest row was written "
            "and the export must not be run.",
            file=sys.stderr,
        )
        return 2

    if record is not None:
        record(
            sql_path=sql_path,
            sql=sql,
            bytes_processed=bytes_processed,
            max_bytes=max_bytes,
            project=project,
            block_start=block_start,
            block_end=block_end,
        )
    print("PASS: estimate is within the configured cap.")
    return 0


def main() -> int:
    config = load_data_config()
    window = config["window"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default=config["bigquery"]["export_sql"])
    parser.add_argument("--project", default=os.getenv("BIGQUERY_PROJECT", ""))
    parser.add_argument(
        "--max-bytes",
        default=os.getenv("BQ_MAX_BYTES", str(config["ingest"]["bq_max_bytes"])),
    )
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    sql_path = project_path(args.sql)
    sql = sql_path.read_text(encoding="utf-8")
    max_bytes = parse_bytes(args.max_bytes)

    try:
        from google.cloud import bigquery

        if not args.project or args.project.startswith("${"):
            raise RuntimeError("Set BIGQUERY_PROJECT to a GCP billing project")
        client = bigquery.Client(project=args.project)
        labels = load_labels_for_params()
        job_config = build_job_config(args.project, window, labels)
    except Exception as exc:  # noqa: BLE001 - print actionable setup error
        print(
            f"PREFLIGHT NOT RUN: {exc}. Install requirements-bigquery.txt and "
            "provide Google Cloud credentials; nothing was written.",
            file=sys.stderr,
        )
        return 3

    record = None if args.no_record else record_dry_run
    return execute_preflight(
        sql=sql,
        client=client,
        job_config=job_config,
        max_bytes=max_bytes,
        block_start=window["block_start"],
        block_end=window["block_end"],
        sql_path=args.sql,
        project=args.project,
        record=record,
    )


if __name__ == "__main__":
    raise SystemExit(main())
