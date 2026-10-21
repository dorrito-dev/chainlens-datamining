from __future__ import annotations

from pathlib import Path

import pandas as pd

from chainlens.rules import evaluate, load_rules, validate_trace

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "rules.yaml"
SCALER_VERSION = "177dbaf33be2"
ADDRESS = "0x" + "4" * 40


def _inputs():
    features = pd.DataFrame({"address": [ADDRESS]})
    flags = pd.DataFrame(
        {
            "cycle_member": [3.0],
            "fanout": [40.0],
            "sanctioned_1hop": [1.0],
            "pagerank_top5": [1.0],
            "apriori_match": [0.9],
        },
        index=[ADDRESS],
    )
    evidence = {
        (ADDRESS, "cycle_member"): {
            "cycle_count": 1,
            "cycles": [
                {
                    "cycle_path": [ADDRESS, "0x" + "5" * 40, "0x" + "6" * 40],
                    "cycle_length": 3,
                    "edge_values": [1.0, 1.0, 1.0],
                    "time_span_hours": 1.0,
                }
            ],
        },
        (ADDRESS, "fanout"): {
            "unique_counterparties": 40,
            "tx_count_out": 40,
            "fanout_ratio": 1.0,
        },
        (ADDRESS, "sanctioned_1hop"): {
            "seed": "0x" + "7" * 40,
            "direction": "out",
            "seed_source": "ofac",
        },
        (ADDRESS, "pagerank_top5"): {"pagerank": 0.02, "threshold": 0.01},
        (ADDRESS, "apriori_match"): {
            "antecedent": ["high_fanout"],
            "support": 0.2,
            "confidence": 0.9,
            "lift": 2.5,
        },
    }
    known = pd.DataFrame(columns=["address", "category"])
    return features, flags, evidence, known


def test_all_rules_clamp_to_max_score():
    cfg = load_rules(RULES_PATH, SCALER_VERSION)
    features, flags, evidence, known = _inputs()
    scores, traces = evaluate(
        features,
        flags,
        evidence,
        known,
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    validate_trace(trace, cfg)
    assert trace["raw_score"] == 110
    assert trace["adjusted_raw_score"] == 110.0
    assert trace["score"] == 100
    assert trace["clamped"] is True
    assert trace["tier"] == "high"
    assert int(scores.iloc[0]["score"]) == 100


def test_clamp_uses_runtime_max_score(tmp_path):
    yaml_text = RULES_PATH.read_text(encoding="utf-8")
    assert "max_score: 100" in yaml_text
    temporary = tmp_path / "rules.yaml"
    temporary.write_text(
        yaml_text.replace("max_score: 100", "max_score: 90", 1),
        encoding="utf-8",
    )
    cfg = load_rules(temporary, SCALER_VERSION)
    features, flags, evidence, known = _inputs()
    _, traces = evaluate(
        features,
        flags,
        evidence,
        known,
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    validate_trace(trace, cfg)
    assert trace["max_score"] == 90
    assert trace["raw_score"] == 110
    assert trace["adjusted_raw_score"] == 110.0
    assert trace["score"] == 90
    assert trace["clamped"] is True
    assert trace["tier"] == "high"
