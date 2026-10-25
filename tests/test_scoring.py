"""Score arithmetic: slide-21 case, down-weighting, clamping, and summary."""

from __future__ import annotations

import pandas as pd
import pytest

from scoring import (
    evaluate_scores,
    load_rules_config,
    score_summary,
    traces_from_frame,
    traces_to_frame,
    validate_trace,
)


def addr(n: int) -> str:
    return f"0x{n:040x}"


def flag(address: str, signal: str, value: float = 1, evidence: dict | None = None) -> dict:
    return {(address, signal): {"signal_value": value, "evidence": evidence or {}}}


def test_slide_21_cycle_fanout_pagerank_scores_60_high() -> None:
    cfg = load_rules_config()
    a = addr(1)
    features = pd.DataFrame({"address": [a]})
    flags = {}
    flags.update(flag(a, "cycle_member", 3, {"cycle_count": 1}))
    flags.update(flag(a, "fanout", 25, {"unique_counterparties": 25}))
    flags.update(flag(a, "pagerank_top5", 1, {"pagerank": 0.01}))
    scores, traces = evaluate_scores(features, flags, pd.DataFrame(), cfg)
    row = scores.iloc[0]
    assert row["raw_score"] == 60
    assert row["score"] == 60
    assert row["tier"] == "high"
    assert row["top_driver"] == "R1"
    validate_trace(traces.iloc[0]["trace"])


def test_downweight_tiers_and_clamp() -> None:
    cfg = load_rules_config()
    a, b, c, d, e, f, g = (addr(i) for i in range(1, 8))
    features = pd.DataFrame({"address": [a, b, c, d, e, f, g]})
    flags = {}
    flags.update(flag(a, "cycle_member", 3))
    flags.update(flag(a, "fanout", 25))
    flags.update(flag(a, "pagerank_top5", 1))

    flags.update(flag(b, "cycle_member", 3))
    flags.update(flag(b, "fanout", 25))

    flags.update(flag(c, "fanout", 25))
    flags.update(flag(c, "pagerank_top5", 1))

    flags.update(flag(d, "fanout", 25))
    flags.update(flag(d, "pagerank_top5", 1))

    flags.update(flag(e, "cycle_member", 3))
    flags.update(flag(e, "fanout", 25))
    flags.update(flag(e, "pagerank_top5", 1))

    for signal, value in (
        ("cycle_member", 3), ("fanout", 25), ("sanctioned_1hop", 1),
        ("pagerank_top5", 1), ("apriori_match", 0.9),
    ):
        flags.update(flag(f, signal, value))

    known = pd.DataFrame([
        {"address": d, "category": "exchange", "note": "test"},
        {"address": e, "category": "contract", "note": "test"},
    ])
    scores, traces = evaluate_scores(features, flags, known, cfg)
    by_addr = scores.set_index("address")
    trace_by_addr = traces.set_index("address")["trace"]

    assert by_addr.loc[a, "raw_score"] == 60
    assert by_addr.loc[a, "score"] == 60
    assert by_addr.loc[a, "tier"] == "high"

    assert by_addr.loc[b, "raw_score"] == 45
    assert by_addr.loc[b, "tier"] == "medium"

    assert by_addr.loc[c, "raw_score"] == 35
    assert by_addr.loc[c, "tier"] == "medium"

    assert by_addr.loc[d, "raw_score"] == 35
    assert by_addr.loc[d, "score"] == 10  # 35 * 0.3 = 10.5 -> round-half-even
    assert by_addr.loc[d, "tier"] == "low"
    assert trace_by_addr[d]["downweight"]["applied"] is True
    assert trace_by_addr[d]["downweight"]["category"] == "exchange"
    assert trace_by_addr[d]["adjusted_raw_score"] == pytest.approx(10.5)

    assert by_addr.loc[e, "raw_score"] == 60
    assert by_addr.loc[e, "score"] == 18
    assert by_addr.loc[e, "tier"] == "low"
    assert trace_by_addr[e]["downweight"]["applied"] is True

    assert by_addr.loc[f, "raw_score"] == 110
    assert by_addr.loc[f, "score"] == 100
    assert trace_by_addr[f]["clamped"] is True
    assert by_addr.loc[f, "tier"] == "high"
    assert by_addr.loc[f, "top_driver"] == "R3"

    assert by_addr.loc[g, "raw_score"] == 0
    assert by_addr.loc[g, "score"] == 0
    assert by_addr.loc[g, "tier"] == "low"
    assert by_addr.loc[g, "top_driver"] is None
    assert trace_by_addr[g]["downweight"]["applied"] is False

    for trace in traces["trace"]:
        validate_trace(trace)

    summary = score_summary(scores, traces)
    assert summary["total_scored"] == 7
    assert summary["tiers"] == {"high": 2, "medium": 2, "low": 3}
    assert summary["clamped_count"] == 1
    assert summary["downweighted_count"] == 2
    assert len(summary["top_10"]) == 7


def test_trace_serialisation_round_trip() -> None:
    cfg = load_rules_config()
    a = addr(11)
    features = pd.DataFrame({"address": [a]})
    flags = flag(a, "cycle_member", 3)
    _, traces = evaluate_scores(features, flags, pd.DataFrame(), cfg)
    frame = traces_to_frame(traces)
    assert "trace_json" in frame.columns
    restored = traces_from_frame(frame)
    assert restored.iloc[0]["trace"]["raw_score"] == 25
    validate_trace(restored.iloc[0]["trace"])
