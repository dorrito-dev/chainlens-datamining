"""Reproducibility manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

UPSERT = text(
    """
    INSERT INTO data_manifest (
        file_path, sha256, row_count, block_start, block_end,
        bytes_processed, estimated_usd, metadata
    )
    VALUES (
        :file_path, :sha256, :row_count, :block_start, :block_end,
        :bytes_processed, :estimated_usd, CAST(:metadata AS JSONB)
    )
    ON CONFLICT (file_path) DO UPDATE SET
        sha256 = EXCLUDED.sha256,
        row_count = EXCLUDED.row_count,
        block_start = EXCLUDED.block_start,
        block_end = EXCLUDED.block_end,
        bytes_processed = EXCLUDED.bytes_processed,
        estimated_usd = EXCLUDED.estimated_usd,
        metadata = EXCLUDED.metadata,
        created_at = now()
    """
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_manifest(
    engine: Engine,
    file_path: str,
    sha256: str,
    *,
    row_count: int | None = None,
    block_start: int | None = None,
    block_end: int | None = None,
    bytes_processed: int | None = None,
    estimated_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            UPSERT,
            {
                "file_path": file_path,
                "sha256": sha256,
                "row_count": row_count,
                "block_start": block_start,
                "block_end": block_end,
                "bytes_processed": bytes_processed,
                "estimated_usd": estimated_usd,
                "metadata": json.dumps(metadata or {}, sort_keys=True),
            },
        )


def record_file(
    engine: Engine,
    path: str | Path,
    *,
    file_path: str | None = None,
    row_count: int | None = None,
    block_start: int | None = None,
    block_end: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    resolved = Path(path)
    record_manifest(
        engine,
        file_path or str(resolved),
        sha256_file(resolved),
        row_count=row_count,
        block_start=block_start,
        block_end=block_end,
        metadata=metadata,
    )
