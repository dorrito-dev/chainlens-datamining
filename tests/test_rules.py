"""Rule contract, locked weights, tier boundaries, and R5 gate."""

from __future__ import annotations

import pandas as pd
import pytest

from scoring import (
    evaluate_scores,
    load_rules_config,
    tier_for_score,
    validate_trace,
)


def addr(n: int) -> str:
    return f"0x{n:040x}"


def flags_for(
    address: str,
    *,
    cycle: int | None = None,
    fanout: bool = False,
    sanctioned: bool = False,
    pagerank: bool = False,
    apriori: float | None = None,
) -> dict[tuple[str, str], dict]:
    flags: dict[tuple[str, str], dict] = {}
    if cycle is not None:
        flags[(address, "cycle_member")] = {
            "signal_value": cycle,
            "evidence": {"cycle_count": 1, "cycles": [{"cycle_path": [address]}]},
        }
    if fanout:
        flags[(address, "fanout")] = {
            "signal_value": 25,
            "evidence": {"unique_counterparties": 25, "tx_count_out": 25,
                         "fanout_ratio": 1.0, "out_in_ratio": 6.0,
                         "r2_deck_equivalent": True},
        }
    if sanctioned:
        flags[(address, "sanctioned_1hop")] = {
            "signal_value": 1,
            "evidence": {"seed": addr(999), "direction": "out", "seeds": [addr(999)]},
        }
    if pagerank:
        flags[(address, "pagerank_top5")] = {
            "signal_value": 1,
            "evidence": {"pagerank": 0.01, "threshold": 0.005},
        }
    if apriori is not None:
        flags[(address, "apriori_match")] = {
            "signal_value": apriori,
            "evidence": {"antecedent": ["high_fanout"], "confidence": apriori,
                         "support": 0.2, "lift": 3.0, "confidence_min": 0.8},
        }
    return flags


def test_locked_weights_and_tiers() -> None:
    cfg = load_rules_config()
    assert [rule.id for rule in cfg.rules] == ["R1", "R2", "R3", "R4", "R5"]
    assert [rule.weight for rule in cfg.rules] == [25, 20, 30, 15, 20]
    assert cfg.max_score == 100
    assert (cfg.tier_low_max, cfg.tier_medium_max, cfg.tier_high_min) == (24, 59, 60)
    assert tier_for_score(0, cfg) == "low"
    assert tier_for_score(24, cfg) == "low"
    assert tier_for_score(25, cfg) == "medium"
    assert tier_for_score(59, cfg) == "medium"
    assert tier_for_score(60, cfg) == "high"
    assert tier_for_score(100, cfg) == "high"


def test_r1_r5_hits_and_r5_confidence_gate() -> None:
    cfg = load_rules_config()
    a, b, c, d = addr(1), addr(2), addr(3), addr(4)
    features = pd.DataFrame({"address": [a, b, c, d]})
    flags = {}
    flags.update(flags_for(a, cycle=3, fanout=True, sanctioned=True, pagerank=True, apriori=0.9))
    flags.update(flags_for(b, cycle=6))  # length 6 is outside R1's 2-5
    flags.update(flags_for(c, apriori=0.79))  # below R5 confidence gate
    flags.update(flags_for(d, cycle=5, fanout=True))

    scores, traces = evaluate_scores(features, flags, pd.DataFrame(), cfg)
    by_addr = scores.set_index("address")

    assert by_addr.loc[a, "raw_score"] == 110
    assert by_addr.loc[a, "score"] == 100
    assert by_addr.loc[a, "tier"] == "high"
    assert by_addr.loc[a, "top_driver"] == "R3"  # 30 > 25 > 20 > 15

    assert by_addr.loc[b, "raw_score"] == 0
    assert by_addr.loc[b, "tier"] == "low"
    assert by_addr.loc[c, "raw_score"] == 0
    assert by_addr.loc[d, "raw_score"] == 45
    assert by_addr.loc[d, "tier"] == "medium"
    assert by_addr.loc[d, "top_driver"] == "R1"  # 25 > 20

    trace_a = traces.set_index("address").loc[a, "trace"]
    assert trace_a["rules"]["R1"]["hit"] is True
    assert trace_a["rules"]["R2"]["hit"] is True
    assert trace_a["rules"]["R3"]["hit"] is True
    assert trace_a["rules"]["R4"]["hit"] is True
    assert trace_a["rules"]["R5"]["hit"] is True
    assert trace_a["clamped"] is True
    assert trace_a["raw_score"] == 110
    assert trace_a["score"] == 100

    trace_c = traces.set_index("address").loc[c, "trace"]
    assert trace_c["rules"]["R5"]["hit"] is False
    assert "confidence" in trace_c["rules"]["R5"]["gate"]

    for trace in traces["trace"]:
        validate_trace(trace)


def test_continuous_rule_strengths_scale_contributions() -> None:
    cfg = load_rules_config()
    a, b = addr(10), addr(11)
    features = pd.DataFrame({"address": [a, b]})
    flags = {
        (a, "sanctioned_1hop"): {
            "signal_value": 1, "strength": 0.25,
            "evidence": {"exposure_fraction": 0.25, "seed": addr(999)},
        },
        (a, "pagerank_top5"): {
            "signal_value": 1, "strength": 0.5,
            "evidence": {"pagerank": 0.02, "threshold": 0.01, "top5": True},
        },
        (b, "pagerank_top5"): {
            "signal_value": 0, "strength": 0.01,
            "evidence": {"pagerank": 0.001, "threshold": 0.01, "top5": False},
        },
    }
    scores, traces = evaluate_scores(features, flags, pd.DataFrame(), cfg)
    by_addr = scores.set_index("address")
    assert by_addr.loc[a, "raw_score"] == pytest.approx(7.5 + 7.5)
    assert by_addr.loc[a, "top_driver"] == "R3"  # tie resolved by RULE_IDS order
    assert by_addr.loc[b, "raw_score"] == 0  # below R4 min_strength

    trace_a = traces.set_index("address").loc[a, "trace"]
    assert trace_a["rules"]["R3"]["strength"] == pytest.approx(0.25)
    assert trace_a["rules"]["R3"]["contribution"] == pytest.approx(7.5)
    assert trace_a["rules"]["R4"]["contribution"] == pytest.approx(7.5)
    trace_b = traces.set_index("address").loc[b, "trace"]
    assert trace_b["rules"]["R4"]["hit"] is False
    assert "strength" in trace_b["rules"]["R4"]["gate"]

    for trace in traces["trace"]:
        validate_trace(trace)


def test_rules_config_rejects_weight_change(tmp_path) -> None:
    text = (__import__("pathlib").Path(__file__).resolve().parents[1] / "config" / "rules.yaml").read_text()
    broken = text.replace("weight: 25", "weight: 26", 1)
    path = tmp_path / "rules.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked rule weights"):
        load_rules_config(path)


def test_trace_validation_detects_tampering() -> None:
    cfg = load_rules_config()
    address = addr(5)
    features = pd.DataFrame({"address": [address]})
    flags = flags_for(address, cycle=3)
    _, traces = evaluate_scores(features, flags, pd.DataFrame(), cfg)
    trace = traces.iloc[0]["trace"]
    validate_trace(trace)
    trace["rules"]["R1"]["contribution"] = 0
    with pytest.raises(RuntimeError, match="sum to raw_score"):
        validate_trace(trace)
