#!/usr/bin/env python3
"""Phase 4 placeholder risk-score seed for the Streamlit shell."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.seed_scores import seed_scores  # noqa: E402


def main() -> int:
    engine = get_pg_engine()
    apply_schema(engine)
    seed_scores(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
