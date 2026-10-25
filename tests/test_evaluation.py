"""Held-out metrics, insufficient-data guard, and leave-one-rule-out ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import evaluate as eval_mod
from scoring import evaluate_scores, load_rules_config


def addr(n: int) -> str:
    return f"0x{n:040x}"


def test_metrics_at_high_threshold() -> None:
    positives = [addr(i) for i in range(1, 6)]
    negatives = [addr(i) for i in range(11, 16)]
    scores = pd.DataFrame({
        "address": positives + negatives,
        "score": [80, 70, 60, 50, 40, 20, 10, 30, 0, 55],
        "tier": ["high", "high", "high", "medium", "medium",
                 "low", "low", "medium", "low", "medium"],
    })
    labels = pd.DataFrame({
        "address": positives + negatives,
        "label": ["fraud"] * 5 + ["negative"] * 5,
        "source": ["real"] * 10,
    })
    splits = pd.DataFrame({
        "address": positives + negatives,
        "split": ["test"] * 10,
    })
    features = pd.DataFrame({"address": positives + negatives})

    result = eval_mod.evaluate(scores, labels, splits, features, threshold=60)
    assert result["insufficient"] is False
    assert result["n_positives"] == 5
    assert result["n_negatives"] == 5
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (3, 0, 5, 2)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(0.9266667, abs=1e-4)
    medium = result["operating_points"]["medium_25"]
    assert medium["precision"] == pytest.approx(5 / 7)
    assert medium["recall"] == pytest.approx(1.0)
    assert medium["f1"] == pytest.approx(5 / 6)


def test_insufficient_guard_returns_no_fake_metric() -> None:
    addresses = [addr(i) for i in range(1, 6)]
    scores = pd.DataFrame({"address": addresses, "score": [80, 70, 10, 5, 0], "tier": ["high"] * 2 + ["low"] * 3})
    labels = pd.DataFrame({
        "address": addresses,
        "label": ["fraud", "fraud", "negative", "negative", "negative"],
        "source": ["real"] * 5,
    })
    splits = pd.DataFrame({"address": addresses, "split": ["test"] * 5})
    features = pd.DataFrame({"address": addresses})
    result = eval_mod.evaluate(scores, labels, splits, features, threshold=60)
    assert result["insufficient"] is True
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None


def test_fixture_labels_are_excluded_by_default() -> None:
    positives = [addr(i) for i in range(1, 6)]
    negatives = [addr(i) for i in range(11, 16)]
    scores = pd.DataFrame({
        "address": positives + negatives,
        "score": [80, 70, 60, 50, 40, 20, 10, 30, 0, 55],
        "tier": ["high"] * 3 + ["medium"] * 2 + ["low"] * 3 + ["medium"] * 2,
    })
    labels = pd.DataFrame({
        "address": positives + negatives,
        "label": ["fraud"] * 5 + ["negative"] * 5,
        "source": ["real"] * 4 + ["fixture"] + ["real"] * 5,
    })
    splits = pd.DataFrame({"address": positives + negatives, "split": ["test"] * 10})
    features = pd.DataFrame({"address": positives + negatives})

    without = eval_mod.evaluate(scores, labels, splits, features, threshold=60, include_fixture=False)
    with_fixture = eval_mod.evaluate(scores, labels, splits, features, threshold=60, include_fixture=True)
    assert without["test_positives"] == 4
    assert with_fixture["test_positives"] == 5


def test_ablation_drops_each_rule_and_counts_tier_changes() -> None:
    positives = [addr(i) for i in range(1, 6)]
    negatives = [addr(i) for i in range(11, 16)]
    addresses = positives + negatives
    features = pd.DataFrame({"address": addresses})
    flags = {}
    for address in positives:
        flags[(address, "cycle_member")] = {"signal_value": 3, "evidence": {"cycle_count": 1}}
        flags[(address, "fanout")] = {"signal_value": 30, "evidence": {"unique_counterparties": 30}}
        flags[(address, "sanctioned_1hop")] = {"signal_value": 1, "evidence": {"seed": addr(999)}}

    labels = pd.DataFrame({
        "address": addresses,
        "label": ["fraud"] * 5 + ["negative"] * 5,
        "source": ["real"] * 10,
    })
    splits = pd.DataFrame({"address": addresses, "split": ["test"] * 10})
    cfg = load_rules_config()
    scores, _ = evaluate_scores(features, flags, pd.DataFrame(), cfg)

    baseline = eval_mod.evaluate(scores, labels, splits, features, threshold=60)
    assert baseline["f1"] == pytest.approx(1.0)

    ablation = eval_mod.ablation(
        scores, features, flags, pd.DataFrame(), labels, splits, cfg,
        threshold=60, include_fixture=False,
    )
    by_rule = ablation.set_index("rule")
    for rule in ("R1", "R2", "R3"):
        assert by_rule.loc[rule, "delta_f1"] == pytest.approx(-1.0)
        assert by_rule.loc[rule, "affected_wallets"] == 5
        assert by_rule.loc[rule, "tier_changes"] == 5
    for rule in ("R4", "R5"):
        assert by_rule.loc[rule, "delta_f1"] == pytest.approx(0.0)
        assert by_rule.loc[rule, "affected_wallets"] == 0
        assert by_rule.loc[rule, "tier_changes"] == 0


def test_write_evaluation_artifacts(tmp_path: Path) -> None:
    evaluation = {
        "insufficient": False, "test_addresses": 10, "test_positives": 5,
        "test_negatives": 5, "dropped_positives": 0, "dropped_negatives": 0,
        "threshold": 60, "precision": 0.5, "recall": 0.5, "f1": 0.5,
        "average_precision": 0.6, "tp": 2, "fp": 2, "tn": 3, "fn": 3,
    }
    ablation = pd.DataFrame([{"rule": "R1", "affected_wallets": 1, "delta_f1": -0.1}])
    eval_mod.write_evaluation(evaluation, ablation, tmp_path)
    assert json.loads((tmp_path / "evaluation.json").read_text())["f1"] == 0.5
    assert (tmp_path / "ablation.csv").exists()
    assert (tmp_path / "evaluation.md").exists()
