from __future__ import annotations

from pathlib import Path

import pandas as pd

from chainlens.rules import (
    REQUIRED_TRACE_KEYS,
    RULE_IDS,
    evaluate,
    load_rules,
    validate_trace,
)

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "rules.yaml"
SCALER_VERSION = "177dbaf33be2"
ADDRESS = "0x" + "2" * 40


def _config():
    return load_rules(RULES_PATH, SCALER_VERSION)


def test_every_stored_trace_has_the_full_contract(pg_engine, score_run):
    frame = pd.read_sql(
        "SELECT address, trace FROM score_trace ORDER BY address",
        pg_engine,
    )
    assert not frame.empty
    cfg = _config()
    for row in frame.itertuples(index=False):
        trace = row.trace
        assert REQUIRED_TRACE_KEYS <= set(trace)
        assert tuple(trace["rules"]) == RULE_IDS
        assert sum(
            trace["rules"][rule_id]["contribution"] for rule_id in RULE_IDS
        ) == trace["raw_score"]
        assert trace["downweight"]["pre_downweight_raw"] == trace["raw_score"]
        assert trace["downweight"]["post_downweight_raw"] == trace["adjusted_raw_score"]
        assert trace["max_score"] == 100
        validate_trace(trace, cfg)

        for rule_id in RULE_IDS:
            rule = trace["rules"][rule_id]
            if rule["hit"]:
                assert rule["inputs"] is not None
            else:
                assert rule["inputs"] is None
                assert rule["contribution"] == 0


def test_non_hit_inputs_are_null_not_empty_objects(pg_engine, score_run):
    frame = pd.read_sql("SELECT trace FROM score_trace", pg_engine)
    assert not frame.empty
    found_non_hit = False
    for row in frame.itertuples(index=False):
        for rule in row.trace["rules"].values():
            if not rule["hit"]:
                found_non_hit = True
                assert rule["inputs"] is None
                assert rule["inputs"] != {}
    assert found_non_hit


def test_r1_cycle_cap_marks_trace_truncated():
    cfg = _config()
    features = pd.DataFrame({"address": [ADDRESS]})
    flags = pd.DataFrame({"cycle_member": [3.0]}, index=[ADDRESS])
    cycles = [
        {
            "cycle_path": [ADDRESS, "0x" + f"{index:040x}"],
            "cycle_length": 3,
            "edge_values": [1.0, 1.0, 1.0],
            "time_span_hours": 1.0,
        }
        for index in range(6)
    ]
    evidence = {
        (ADDRESS, "cycle_member"): {
            "cycle_path": cycles[0]["cycle_path"],
            "cycle_count": 6,
            "cycles": cycles,
        }
    }
    _, traces = evaluate(
        features,
        flags,
        evidence,
        pd.DataFrame(columns=["address", "category"]),
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    validate_trace(trace, cfg)
    assert trace["truncated"] is True
    assert trace["rules"]["R1"]["inputs"]["cycle_count"] == 6
    assert len(trace["rules"]["R1"]["inputs"]["cycles"]) == 5
    assert trace["rules"]["R1"]["inputs"]["cycles"][0]["length"] == 3
    assert trace["rules"]["R1"]["inputs"]["cycles"][0]["max_min_ratio"] == 1.0
