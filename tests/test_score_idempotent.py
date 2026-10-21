from __future__ import annotations

import pandas as pd

from chainlens.scoring import run_score


def _snapshot(pg_engine) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    risk = pd.read_sql(
        "SELECT address, raw_score, score, tier, top_driver, rules_version, scored_at "
        "FROM risk_scores ORDER BY address",
        pg_engine,
    )
    trace = pd.read_sql(
        "SELECT address, rules_version, trace, created_at "
        "FROM score_trace ORDER BY address, rules_version",
        pg_engine,
    )
    return risk, trace, str(risk["rules_version"].iloc[0])


def test_two_score_runs_are_content_idempotent(pg_engine):
    # This test must not reuse a session score run: earlier tests may have
    # rewritten address_flags. Two back-to-back score runs share one evidence
    # snapshot, which is the idempotency contract.
    first_summary = run_score()
    assert first_summary["row_count"] > 0
    risk_before, trace_before, version_before = _snapshot(pg_engine)

    second_summary = run_score()
    risk_after, trace_after, version_after = _snapshot(pg_engine)

    assert version_before == version_after == second_summary["rules_version"]
    pd.testing.assert_frame_equal(risk_before, risk_after)
    pd.testing.assert_frame_equal(trace_before, trace_after)
    assert len(trace_after) == second_summary["row_count"]
