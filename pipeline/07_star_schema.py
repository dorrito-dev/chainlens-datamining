#!/usr/bin/env python3
"""One-off Phase 6 star-schema population (reporting artifact)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.star_schema import populate_star_schema  # noqa: E402


def main() -> int:
    result = populate_star_schema()
    print("Star schema populated:")
    for key in ("fact_transactions", "dim_address", "dim_label", "source_rows"):
        print(f"  {key}: {result[key]:,}")
    print(f"  parquet: {result['parquet_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
