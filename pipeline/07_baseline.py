#!/usr/bin/env python3
"""Write the Phase 6 baseline score/trace snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.ablation import snapshot_baseline  # noqa: E402


def main() -> int:
    result = snapshot_baseline()
    summary = result["summary"]
    print(f"Baseline parquet: {result['parquet_path']}")
    print(f"Baseline summary: {result['summary_path']}")
    print(f"rules_version: {summary['rules_version']}")
    print(f"rows: {summary['row_count']}")
    print(f"content_sha256: {summary['content_sha256']}")
    print(f"tier counts: {summary['counts_by_tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
