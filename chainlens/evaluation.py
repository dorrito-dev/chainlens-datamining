"""Phase 6 evaluation helpers.

The fixture contains one test-split fixture label. It is deliberately *not*
counted as a real positive; every metric path therefore uses the
``source != 'fixture'`` evaluable-positive definition from the Phase 6 brief.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

RULE_IDS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")


def classification_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return metrics, or an explicit insufficient-data marker.

    A metric on fewer than three positives is not reported. This is a
    deliberate refusal to turn a single fixture label into a 1.0 headline.
    """
    truth = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()
    if truth.size != scores.size:
        raise ValueError("y_true and y_score must have the same length")
    n_positives = int((truth == 1).sum())
    if n_positives < 3:
        return {"insufficient": True, "n_positives": n_positives}

    predictions = (scores >= threshold).astype(int)
    if len(np.unique(truth)) < 2:
        return {"insufficient": True, "n_positives": n_positives}
    return {
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "n_positives": n_positives,
        "n_total": int(truth.size),
    }


def ranked_list(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Join scores with labels/splits and return the top-N ranked addresses."""
    required_scores = {"address", "score", "tier", "top_driver"}
    missing = required_scores - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing columns: {sorted(missing)}")

    labels = labels_df.copy()
    labels["address"] = labels["address"].astype(str).str.lower()
    labels["source"] = labels["source"].astype(str)
    labels["label"] = labels["label"].astype(str)
    labels_agg = (
        labels.groupby("address", as_index=False)
        .agg(
            label=("label", lambda values: ", ".join(sorted(set(values)))),
            source=("source", lambda values: ", ".join(sorted(set(values)))),
        )
        .assign(is_labeled=True)
    )

    splits = splits_df.copy()
    splits["address"] = splits["address"].astype(str).str.lower()
    splits = splits[["address", "split"]]

    result = scores_df.copy()
    result["address"] = result["address"].astype(str).str.lower()
    result = result.merge(labels_agg, on="address", how="left")
    result = result.merge(splits, on="address", how="left")
    result["is_labeled"] = result["is_labeled"].eq(True)
    result["split"] = result["split"].fillna("unlabeled")
    result["source"] = result["source"].fillna("")

    result = result.sort_values(
        ["score", "address"], ascending=[False, True]
    ).reset_index(drop=True)
    return result[
        [
            "address",
            "score",
            "tier",
            "top_driver",
            "is_labeled",
            "split",
            "source",
        ]
    ].head(top_n)


def _coerce_trace(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise ValueError(f"Unsupported trace value: {type(value)!r}")


def rule_contribution_table(traces_df: pd.DataFrame) -> pd.DataFrame:
    """Per-rule hit count, contribution sum, and median contribution on hits."""
    if "trace" in traces_df.columns:
        traces = traces_df["trace"].map(_coerce_trace)
    elif "trace_json" in traces_df.columns:
        traces = traces_df["trace_json"].map(_coerce_trace)
    else:
        raise ValueError("traces_df needs a trace or trace_json column")

    rows: list[dict[str, Any]] = []
    for rule_id in RULE_IDS:
        contributions: list[int] = []
        hit_count = 0
        for trace in traces:
            rule = trace.get("rules", {}).get(rule_id)
            if not rule or not rule.get("hit"):
                continue
            hit_count += 1
            contributions.append(int(rule.get("contribution", 0)))
        rows.append(
            {
                "rule_id": rule_id,
                "hit_count": hit_count,
                "sum_contribution": int(sum(contributions)),
                "median_contribution_on_hits": (
                    float(np.median(contributions)) if contributions else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluable_positives(
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return test-split, feature-bearing labels with non-fixture sources."""
    labels = labels_df.copy()
    splits = splits_df.copy()
    features = features_df.copy()
    for frame in (labels, splits, features):
        frame["address"] = frame["address"].astype(str).str.lower()
    return (
        labels.merge(splits, on="address", how="inner")
        .merge(features[["address"]], on="address", how="inner")
        .query("split == 'test' and source != 'fixture'")
        .reset_index(drop=True)
    )


def write_evaluation_snapshot(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    path: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Convenience helper used by the notebook to persist its ranked exhibit."""
    ranked = ranked_list(scores_df, labels_df, splits_df, top_n=top_n)
    ranked.to_csv(path, index=False)
    return ranked
