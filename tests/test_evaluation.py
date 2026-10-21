from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from chainlens.evaluation import (
    classification_metrics,
    ranked_list,
    rule_contribution_table,
)


def test_classification_metrics_refuses_n_equals_one():
    result = classification_metrics([1, 0, 0], [0.9, 0.1, 0.2])
    assert result == {"insufficient": True, "n_positives": 1}
    assert "f1" not in result
    assert "precision" not in result
    assert "recall" not in result
    assert "pr_auc" not in result


def test_classification_metrics_perfect_separation():
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_score = [0.95, 0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05, 0.01]
    result = classification_metrics(y_true, y_score)
    assert result.get("insufficient", False) is False
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["n_positives"] == 5


def test_classification_metrics_random_labels_are_bounded():
    rng = np.random.default_rng(42)
    y_true = np.array([1] * 5 + [0] * 7)
    y_score = rng.random(len(y_true))
    result = classification_metrics(y_true, y_score)
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0


def test_ranked_list_sorts_and_annotates():
    scores = pd.DataFrame(
        {
            "address": ["0xa", "0xb", "0xc"],
            "score": [10, 30, 20],
            "tier": ["low", "medium", "low"],
            "top_driver": ["R4", "R3", "R2"],
        }
    )
    labels = pd.DataFrame(
        {
            "address": ["0xb"],
            "label": ["sanctioned"],
            "source": ["ofac"],
            "as_of": ["2026-09-30"],
        }
    )
    splits = pd.DataFrame({"address": ["0xb"], "split": ["test"]})
    result = ranked_list(scores, labels, splits, top_n=2)
    assert list(result.columns) == [
        "address",
        "score",
        "tier",
        "top_driver",
        "is_labeled",
        "split",
        "source",
    ]
    assert len(result) == 2
    assert result["score"].tolist() == [30, 20]
    assert result.iloc[0]["is_labeled"]
    assert result.iloc[0]["split"] == "test"
    assert result.iloc[0]["source"] == "ofac"
    assert not result.iloc[1]["is_labeled"]


def test_rule_contribution_table_matches_score_trace(pg_engine):
    with pg_engine.connect() as conn:
        version = conn.execute(
            text("SELECT DISTINCT rules_version FROM risk_scores")
        ).scalar_one()
        traces = pd.read_sql(
            text(
                "SELECT trace FROM score_trace WHERE rules_version = :version"
            ),
            conn,
            params={"version": version},
        )
    table = rule_contribution_table(traces).set_index("rule_id")
    for rule_id in ("R1", "R2", "R3", "R4", "R5"):
        contributions = []
        for trace in traces["trace"]:
            rule = trace["rules"][rule_id]
            if rule["hit"]:
                contributions.append(int(rule["contribution"]))
        assert int(table.loc[rule_id, "hit_count"]) == len(contributions)
        assert int(table.loc[rule_id, "sum_contribution"]) == sum(contributions)
