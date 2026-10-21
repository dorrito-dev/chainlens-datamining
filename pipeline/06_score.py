#!/usr/bin/env python3
"""Phase 5 scoring entrypoint.

Usage::

    python pipeline/06_score.py
    python pipeline/06_score.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.scoring import print_score_check, run_score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="print score provenance/counts and verify postconditions",
    )
    args = parser.parse_args()
    if args.check:
        return print_score_check()
    run_score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
