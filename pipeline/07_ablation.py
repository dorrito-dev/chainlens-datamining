#!/usr/bin/env python3
"""Run the Phase 6 per-rule ablation detection test and write its report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.ablation import (  # noqa: E402
    assert_baseline_restored,
    load_baseline,
    run_ablation,
    write_ablation_report,
)


def main() -> int:
    baseline = load_baseline()
    report, details = run_ablation(baseline=baseline)
    csv_path, json_path = write_ablation_report(report, details)
    assert_baseline_restored(baseline=baseline)
    print(report.to_string(index=False))
    print(f"Ablation report: {csv_path}")
    print(f"Ablation details: {json_path}")
    print("BASELINE RESTORE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
