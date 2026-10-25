"""Evaluation: test-split precision/recall/F1 and leave-one-rule-out ablation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from scoring import RULE_IDS, SIGNAL_BY_RULE, RulesConfig, evaluate_scores

PROJECT_ROOT = Path(__file__).resolve().parent


def _dedupe_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels
    frame = labels.copy()
    priority = {"sanctioned": 0, "fraud": 1, "darklist": 2, "negative": 9, "context": 10}
    frame["_priority"] = frame["label"].map(priority).fillna(5)
    frame = frame.sort_values(["address", "_priority"]).drop_duplicates("address")
    return frame.drop(columns=["_priority"])


def prepare_evaluation(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    features: pd.DataFrame,
    *,
    include_fixture: bool = False,
) -> dict[str, Any]:
    test_addresses = set(splits.loc[splits["split"] == "test", "address"])
    labels = _dedupe_labels(labels)
    feature_addresses = set(features["address"]) if not features.empty else set(scores["address"])

    positive_labels = labels[
        (~labels["label"].isin(["negative", "context"]))
        & (labels["address"].isin(test_addresses))
    ]
    if not include_fixture:
        positive_labels = positive_labels[positive_labels["source"] != "fixture"]
    negatives = labels[
        (labels["label"] == "negative") & (labels["address"].isin(test_addresses))
    ]

    positives = set(positive_labels["address"]) & feature_addresses
    negatives_set = set(negatives["address"]) & feature_addresses

    score_by_address = dict(zip(scores["address"], scores["score"], strict=False))
    rows = []
    for address, y in [(a, 1) for a in sorted(positives)] + [(a, 0) for a in sorted(negatives_set)]:
        rows.append({"address": address, "y_true": y, "y_score": float(score_by_address.get(address, 0))})
    frame = pd.DataFrame(rows, columns=["address", "y_true", "y_score"])
    return {
        "frame": frame,
        "test_addresses": len(test_addresses),
        "test_positives": len(positives),
        "test_negatives": len(negatives_set),
        "dropped_positives": len(set(positive_labels["address"]) - feature_addresses),
        "dropped_negatives": len(set(negatives["address"]) - feature_addresses),
        "positive_rate": (len(positives) / len(frame)) if len(frame) else 0.0,
    }


def classification_metrics(
    frame: pd.DataFrame,
    *,
    threshold: int = 60,
    min_class: int = 3,
) -> dict[str, Any]:
    n_pos = int((frame["y_true"] == 1).sum()) if not frame.empty else 0
    n_neg = int((frame["y_true"] == 0).sum()) if not frame.empty else 0
    if n_pos < min_class or n_neg < min_class:
        return {
            "insufficient": True,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "threshold": threshold,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
        }

    predicted = frame["y_score"] >= threshold
    tp = int(((frame["y_true"] == 1) & predicted).sum())
    fp = int(((frame["y_true"] == 0) & predicted).sum())
    tn = int(((frame["y_true"] == 0) & ~predicted).sum())
    fn = int(((frame["y_true"] == 1) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    average_precision = None
    try:
        from sklearn.metrics import average_precision_score
        average_precision = float(average_precision_score(frame["y_true"], frame["y_score"]))
    except Exception:  # noqa: BLE001 - metric is optional evidence
        pass

    return {
        "insufficient": False,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
    }


def evaluate(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    features: pd.DataFrame,
    *,
    threshold: int = 60,
    secondary_threshold: int | None = 25,
    include_fixture: bool = False,
) -> dict[str, Any]:
    prepared = prepare_evaluation(
        scores, labels, splits, features, include_fixture=include_fixture,
    )
    metrics = classification_metrics(prepared["frame"], threshold=threshold)
    operating_points = {f"high_{threshold}": metrics}
    if secondary_threshold is not None:
        operating_points[f"medium_{secondary_threshold}"] = classification_metrics(
            prepared["frame"], threshold=secondary_threshold,
        )
    return {
        **metrics,
        "operating_points": operating_points,
        "test_addresses": prepared["test_addresses"],
        "test_positives": prepared["test_positives"],
        "test_negatives": prepared["test_negatives"],
        "dropped_positives": prepared["dropped_positives"],
        "dropped_negatives": prepared["dropped_negatives"],
        "positive_rate": prepared["positive_rate"],
        "case_control": True,
        "note": "P/R/F1 are sample metrics on a case-control sample, not population estimates.",
    }


def _affected_wallets(flags: dict[tuple[str, str], dict[str, Any]], rule_id: str) -> int:
    signal = SIGNAL_BY_RULE[rule_id]
    return len({address for (address, name) in flags if name == signal})


def ablation(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    flags: dict[tuple[str, str], dict[str, Any]],
    known: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: RulesConfig,
    *,
    threshold: int = 60,
    include_fixture: bool = False,
) -> pd.DataFrame:
    """Drop one rule at a time, re-score the same flags, and compare test F1."""
    baseline = evaluate(
        scores, labels, splits, features,
        threshold=threshold, include_fixture=include_fixture,
    )
    rows = []
    for rule_id in RULE_IDS:
        modified_rules = tuple(
            replace(rule, weight=0) if rule.id == rule_id else rule
            for rule in cfg.rules
        )
        modified = replace(cfg, rules=modified_rules)
        ablated_scores, _ = evaluate_scores(features, flags, known, modified)
        metrics = evaluate(
            ablated_scores, labels, splits, features,
            threshold=threshold, include_fixture=include_fixture,
        )
        baseline_f1 = baseline.get("f1")
        ablated_f1 = metrics.get("f1")
        delta = (
            (ablated_f1 - baseline_f1)
            if baseline_f1 is not None and ablated_f1 is not None
            else None
        )
        base_tiers = scores.set_index("address")["tier"]
        ablated_tiers = ablated_scores.set_index("address")["tier"]
        common = base_tiers.index.intersection(ablated_tiers.index)
        tier_changes = int((base_tiers.loc[common] != ablated_tiers.loc[common]).sum())
        rows.append({
            "rule": rule_id,
            "affected_wallets": _affected_wallets(flags, rule_id),
            "baseline_f1": baseline_f1,
            "ablated_f1": ablated_f1,
            "delta_f1": delta,
            "delta_precision": (
                (metrics.get("precision") - baseline.get("precision"))
                if metrics.get("precision") is not None and baseline.get("precision") is not None
                else None
            ),
            "delta_recall": (
                (metrics.get("recall") - baseline.get("recall"))
                if metrics.get("recall") is not None and baseline.get("recall") is not None
                else None
            ),
            "tier_changes": tier_changes,
        })
    return pd.DataFrame(rows)


def write_evaluation(
    evaluation: dict[str, Any],
    ablation_frame: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_path = out / "evaluation.json"
    eval_path.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")
    ablation_path = out / "ablation.csv"
    ablation_frame.to_csv(ablation_path, index=False)

    lines = [
        "# ChainLens evaluation",
        "",
        f"- Test addresses: {evaluation.get('test_addresses', 0)}",
        f"- Test positives: {evaluation.get('test_positives', 0)}",
        f"- Test negatives: {evaluation.get('test_negatives', 0)}",
        f"- Dropped (no activity): {evaluation.get('dropped_positives', 0)} positives, "
        f"{evaluation.get('dropped_negatives', 0)} negatives",
        f"- Operating point: High tier (score ≥ {evaluation.get('threshold', 60)})",
        "",
    ]
    if evaluation.get("insufficient"):
        lines.append("**Insufficient labelled test data for a meaningful F1.**")
    else:
        lines.extend([
            "| Metric | Value |",
            "|---|---:|",
            f"| Precision | {evaluation['precision']:.4f} |",
            f"| Recall | {evaluation['recall']:.4f} |",
            f"| F1 | {evaluation['f1']:.4f} |",
            f"| Average precision | {evaluation.get('average_precision') or float('nan'):.4f} |",
            f"| TP / FP / TN / FN | {evaluation['tp']} / {evaluation['fp']} / "
            f"{evaluation['tn']} / {evaluation['fn']} |",
            "",
        ])
        secondary = (evaluation.get("operating_points") or {}).get("medium_25")
        if secondary and not secondary.get("insufficient"):
            lines.extend([
                "Secondary operating point (Medium ≥ 25):",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Precision | {secondary['precision']:.4f} |",
                f"| Recall | {secondary['recall']:.4f} |",
                f"| F1 | {secondary['f1']:.4f} |",
                f"| TP / FP / TN / FN | {secondary['tp']} / {secondary['fp']} / "
                f"{secondary['tn']} / {secondary['fn']} |",
                "",
            ])
    lines.append("> Case-control sample: these are sample metrics, not population estimates.")
    (out / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return eval_path, ablation_path
