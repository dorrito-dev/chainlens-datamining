#!/usr/bin/env python3
"""Window label-overlap gate.

Counts distinct ground-truth addresses touched inside the candidate block range.
If fewer than ``window.fallback_min_positives`` are found, the configured mode
is switched to ``label_anchored`` and the decision is written to the manifest.

The BigQuery path needs credentials and is never used by CI. Set
CHAINLENS_SAMPLE_MODE=1 to exercise the identical decision logic on the local
fixture dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.config import (  # noqa: E402
    is_sample_mode,
    load_data_config,
    project_path,
)
from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.labels import load_label_frames  # noqa: E402
from chainlens.manifest import record_file  # noqa: E402

DATA_CONFIG = ROOT / "config" / "data.yaml"


def load_labels() -> pd.DataFrame:
    labels_path = project_path(load_data_config()["ingest"]["labels_parquet"])
    if labels_path.exists():
        return pd.read_parquet(labels_path)
    labels, _ = load_label_frames()
    return labels


def count_sample_overlap(labels: pd.DataFrame, window: dict) -> int:
    config = load_data_config()
    source = project_path(config["paths"]["sample_parquet"])
    if not source.exists():
        raise FileNotFoundError(f"Missing sample Parquet: {source}")
    con = duckdb.connect()
    try:
        con.register("label_addresses", labels[["address"]].drop_duplicates())
        count = con.execute(
            """
            SELECT count(*) FROM (
                SELECT DISTINCT lower(from_address) AS address
                FROM read_parquet(?)
                WHERE block_number BETWEEN ? AND ?
                  AND lower(from_address) IN (SELECT address FROM label_addresses)
                UNION
                SELECT DISTINCT lower(to_address) AS address
                FROM read_parquet(?)
                WHERE block_number BETWEEN ? AND ?
                  AND lower(to_address) IN (SELECT address FROM label_addresses)
            )
            """,
            [
                str(source),
                window["block_start"],
                window["block_end"],
                str(source),
                window["block_start"],
                window["block_end"],
            ],
        ).fetchone()[0]
    finally:
        con.close()
    return int(count)


def count_bigquery_overlap(labels: pd.DataFrame, window: dict) -> int:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Install "
            "requirements-bigquery.txt or set CHAINLENS_SAMPLE_MODE=1."
        ) from exc

    project = os.getenv("BIGQUERY_PROJECT")
    if not project or project.startswith("${"):
        raise RuntimeError("Set BIGQUERY_PROJECT to your GCP billing project")
    client = bigquery.Client(project=project)
    query = """
        SELECT count(*) FROM (
            SELECT DISTINCT from_address AS address
            FROM `bigquery-public-data.crypto_ethereum.transactions`
            WHERE block_number BETWEEN @block_start AND @block_end
              AND from_address IN UNNEST(@labels)
            UNION DISTINCT
            SELECT DISTINCT to_address AS address
            FROM `bigquery-public-data.crypto_ethereum.transactions`
            WHERE block_number BETWEEN @block_start AND @block_end
              AND to_address IN UNNEST(@labels)
        )
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("block_start", "INT64", window["block_start"]),
            bigquery.ScalarQueryParameter("block_end", "INT64", window["block_end"]),
            bigquery.ArrayQueryParameter(
                "labels", "STRING", labels["address"].drop_duplicates().tolist()
            ),
        ],
        use_query_cache=False,
    )
    return int(client.query(query, job_config=job_config).result().one()[0])


def update_mode_file(mode: str) -> None:
    text = DATA_CONFIG.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^(\s*)mode:\s*(?:time|label_anchored)\s*(#.*)?$",
        lambda match: f"{match.group(1)}mode: {mode}"
        + (f"  {match.group(2).strip()}" if match.group(2) else ""),
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not find window.mode in config/data.yaml")
    DATA_CONFIG.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "bigquery", "sample"], default="auto")
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    config = load_data_config()
    window = config["window"]
    labels = load_labels()
    mode = args.mode
    if mode == "auto":
        mode = "sample" if is_sample_mode() else "bigquery"

    count = (
        count_sample_overlap(labels, window)
        if mode == "sample"
        else count_bigquery_overlap(labels, window)
    )
    threshold = int(window["fallback_min_positives"])
    decision = "time" if count >= threshold else "label_anchored"

    decision_path = project_path(config["ingest"]["window_decision"])
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": mode,
        "candidate_block_start": window["block_start"],
        "candidate_block_end": window["block_end"],
        "width_blocks": window["block_end"] - window["block_start"] + 1,
        "labeled_addresses_found": count,
        "fallback_min_positives": threshold,
        "previous_mode": window["mode"],
        "mode": decision,
        "switched": decision != window["mode"],
        "label_sources": sorted(labels["source"].unique().tolist()),
        "label_rows": int(len(labels)),
        "sample_mode": mode == "sample",
    }
    decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    engine = get_pg_engine()
    apply_schema(engine)
    record_file(
        engine,
        decision_path,
        file_path="data/processed/window_decision.json",
        row_count=count,
        block_start=window["block_start"],
        block_end=window["block_end"],
        metadata={
            "stage": "overlap_gate",
            "method": mode,
            "decision": decision,
            "threshold": threshold,
            "sample_mode": mode == "sample",
        },
    )

    print(
        f"Overlap gate ({mode}): {count:,} distinct labeled addresses found in "
        f"blocks {window['block_start']}-{window['block_end']} "
        f"(threshold {threshold}) -> mode={decision}"
    )
    if args.update_config:
        update_mode_file(decision)
        print(f"Updated config/data.yaml window.mode -> {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
