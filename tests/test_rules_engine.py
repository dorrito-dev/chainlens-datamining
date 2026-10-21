from __future__ import annotations

from pathlib import Path

import pandas as pd

from chainlens.rules import (
    RULE_IDS,
    RulesConfig,
    compute_rules_version,
    evaluate,
    load_rules,
    validate_trace,
)

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "rules.yaml"
SCALER_VERSION = "177dbaf33be2"
ADDRESS = "0x" + "1" * 40


def _config() -> RulesConfig:
    return load_rules(RULES_PATH, SCALER_VERSION)


def test_load_rules_has_locked_weights_and_thresholds():
    cfg = _config()
    assert [rule.id for rule in cfg.rules] == list(RULE_IDS)
    assert [rule.weight for rule in cfg.rules] == [25, 20, 30, 15, 20]
    assert cfg.max_score == 100
    assert cfg.tier_low_max == 24
    assert cfg.tier_medium_max == 59
    assert cfg.rules_version_label == "v0-deck"
    assert cfg.known_address_factor == 0.3
    assert cfg.known_address_categories == {"exchange", "bridge", "mixer", "contract"}
    assert len(cfg.rules_version) == 12


def test_rules_version_is_stable_and_provenance_sensitive(tmp_path):
    yaml_text = RULES_PATH.read_text(encoding="utf-8")
    baseline = compute_rules_version(yaml_text, SCALER_VERSION)
    assert baseline == compute_rules_version(yaml_text, SCALER_VERSION)
    assert baseline != compute_rules_version(yaml_text, "different-scaler")

    edited = tmp_path / "rules.yaml"
    edited.write_text(yaml_text + "\n# a comment is part of the fingerprint\n", encoding="utf-8")
    assert load_rules(edited, SCALER_VERSION).rules_version != baseline


def test_contributions_and_top_driver_tie_break():
    cfg = _config()
    features = pd.DataFrame({"address": [ADDRESS]})
    flags = pd.DataFrame(
        {"fanout": [40.0], "apriori_match": [0.9]},
        index=[ADDRESS],
    )
    evidence = {
        (ADDRESS, "fanout"): {
            "unique_counterparties": 40,
            "tx_count_out": 40,
            "fanout_ratio": 1.0,
        },
        (ADDRESS, "apriori_match"): {
            "antecedent": ["high_fanout"],
            "support": 0.2,
            "confidence": 0.9,
            "lift": 2.5,
        },
    }
    scores, traces = evaluate(
        features,
        flags,
        evidence,
        pd.DataFrame(columns=["address", "category"]),
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    validate_trace(trace, cfg)

    assert traces.iloc[0]["address"] == ADDRESS
    assert trace["rules"]["R2"]["hit"] is True
    assert trace["rules"]["R2"]["contribution"] == 20
    assert trace["rules"]["R5"]["hit"] is True
    assert trace["rules"]["R5"]["contribution"] == 20
    assert trace["raw_score"] == 40
    assert int(scores.iloc[0]["raw_score"]) == 40
    # R2 and R5 both contribute 20; R2 must win the R1-R5 order tie-break.
    assert trace["top_driver"] == "R2"
    assert trace["rules"]["R1"]["inputs"] is None
    assert trace["rules"]["R3"]["inputs"] is None
    assert trace["rules"]["R4"]["inputs"] is None


def test_non_hit_rules_have_null_inputs_and_zero_contribution():
    cfg = _config()
    features = pd.DataFrame({"address": [ADDRESS]})
    flags = pd.DataFrame(index=[ADDRESS], columns=["pagerank_top5"], data=[[0.0]])
    scores, traces = evaluate(
        features,
        flags,
        {},
        pd.DataFrame(columns=["address", "category"]),
        cfg,
        evaluated_at="2026-10-21T00:00:00.000Z",
    )
    trace = traces.iloc[0]["trace"]
    validate_trace(trace, cfg)
    assert int(scores.iloc[0]["raw_score"]) == 0
    assert int(scores.iloc[0]["score"]) == 0
    assert trace["top_driver"] is None
    for rule_id in RULE_IDS:
        assert trace["rules"][rule_id]["hit"] is False
        assert trace["rules"][rule_id]["contribution"] == 0
        assert trace["rules"][rule_id]["inputs"] is None
