#!/usr/bin/env python3
"""Phase 3 entrypoint: raw features, train-only scaler, address_features.

Usage::

    python pipeline/03_features.py
    python pipeline/03_features.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.features import print_check, run_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="print feature/scaler status without rebuilding"
    )
    args = parser.parse_args()
    if args.check:
        return print_check()
    summary = run_features()
    print(
        f"Phase 3 complete: {summary['row_count']} addresses, "
        f"scaler {summary['scaler_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
