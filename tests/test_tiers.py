from __future__ import annotations

from pathlib import Path

import pandas as pd

from chainlens.rules import evaluate, load_rules, tier_for_score

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "rules.yaml"
SCALER_VERSION = "177dbaf33be2"
ADDRESS = "0x" + "3" * 40


def _config():
    return load_rules(RULES_PATH, SCALER_VERSION)


def test_tier_thresholds_are_inclusive_low_and_medium():
    cfg = _config()
    expected = {
        0: "low",
        24: "low",
        25: "medium",
        59: "medium",
        60: "high",
        100: "high",
    }
    assert {score: tier_for_score(score, cfg) for score in expected} == expected


def test_tier_uses_final_score_not_raw_score():
    cfg = _config()
    features = pd.DataFrame({"address": [ADDRESS]})
    flags = pd.DataFrame(
        {"fanout": [30.0], "pagerank_top5": [1.0]},
        index=[ADDRESS],
    )
    evidence = {
        (ADDRESS, "fanout"): {
            "unique_counterparties": 30,
            "tx_count_out": 30,
            "fanout_ratio": 1.0,
        },
        (ADDRESS, "pagerank_top5"): {"pagerank": 0.01, "threshold": 0.005},
    }
    known = pd.DataFrame({"address": [ADDRESS], "category": ["exchange"]})
    scores, traces = evaluate(
        features,
        flags,
        evidence,
        known,
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    assert trace["raw_score"] == 35
    assert trace["adjusted_raw_score"] == 10.5
    assert trace["score"] == 10
    assert trace["tier"] == "low"
    assert scores.iloc[0]["tier"] == "low"
