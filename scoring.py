"""Stage 5: locked R1-R5 risk scoring and the per-address audit trace.

Weights and tiers come from config/rules.yaml. This module never recomputes a
detection signal; it consumes the flags written by patterns.py.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
RULE_IDS = ("R1", "R2", "R3", "R4", "R5")
SIGNAL_BY_RULE = {
    "R1": "cycle_member",
    "R2": "fanout",
    "R3": "sanctioned_1hop",
    "R4": "pagerank_top5",
    "R5": "apriori_match",
}


@dataclass(frozen=True)
class RuleSpec:
    id: str
    name: str
    weight: int
    params: dict[str, Any]


@dataclass(frozen=True)
class RulesConfig:
    rules_version: str
    rules_version_label: str
    max_score: int
    tier_low_max: int
    tier_medium_max: int
    tier_high_min: int
    known_address_factor: float
    known_address_categories: set[str]
    rules: tuple[RuleSpec, ...]
    scaler_version: str = ""


def load_rules_config(
    rules_path: str | Path = "config/rules.yaml",
    scaler_version: str = "",
) -> RulesConfig:
    path = Path(rules_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw_text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text) or {}

    rules = tuple(
        RuleSpec(
            id=str(item["id"]),
            name=str(item["name"]),
            weight=int(item["weight"]),
            params=dict(item.get("params") or {}),
        )
        for item in parsed["rules"]
    )
    if tuple(rule.id for rule in rules) != RULE_IDS:
        raise RuntimeError(f"rules.yaml must define {RULE_IDS} in order")
    if [rule.weight for rule in rules] != [25, 20, 30, 15, 20]:
        raise RuntimeError("locked rule weights must be [25, 20, 30, 15, 20]")

    tiers = parsed["tiers"]
    low_max = int(tiers["low_max"])
    medium_max = int(tiers["medium_max"])
    high_min = int(tiers.get("high_min", medium_max + 1))
    if high_min != medium_max + 1:
        raise RuntimeError("tiers.high_min must equal tiers.medium_max + 1")

    downweight = parsed["downweight"]
    version = hashlib.sha256((raw_text + "|" + scaler_version).encode()).hexdigest()[:12]
    return RulesConfig(
        rules_version=version,
        rules_version_label=str(parsed["rules_version_label"]),
        max_score=int(parsed["max_score"]),
        tier_low_max=low_max,
        tier_medium_max=medium_max,
        tier_high_min=high_min,
        known_address_factor=float(downweight["known_address_factor"]),
        known_address_categories={
            str(c).strip().lower() for c in downweight["known_address_categories"]
        },
        rules=rules,
        scaler_version=scaler_version,
    )


def load_known_addresses(path: str | Path = "config/known_addresses.json") -> pd.DataFrame:
    from fetch_data import load_known_addresses as _load
    return _load(path)


def tier_for_score(score: int, cfg: RulesConfig) -> str:
    if score <= cfg.tier_low_max:
        return "low"
    if score <= cfg.tier_medium_max:
        return "medium"
    return "high"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return value[:1000]
    return value


def _rule_params(cfg: RulesConfig, rule_id: str) -> dict[str, Any]:
    return next(rule.params for rule in cfg.rules if rule.id == rule_id)


def _payload_strength(payload: dict[str, Any], default: float = 1.0) -> float:
    """Per-rule severity in [0, 1]; legacy binary flags default to 1.0."""
    value = payload.get("strength")
    if value is None:
        evidence = payload.get("evidence") or {}
        if isinstance(evidence, dict):
            value = evidence.get("strength")
    try:
        strength = float(value) if value is not None else default
    except (TypeError, ValueError):
        strength = default
    return max(0.0, min(1.0, strength))


def evaluate_scores(
    features: pd.DataFrame,
    flags: dict[tuple[str, str], dict[str, Any]],
    known_addresses: pd.DataFrame,
    cfg: RulesConfig,
    *,
    evaluated_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorised rule hits + per-address trace. Returns (scores, traces)."""
    evaluated_at = evaluated_at or datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    addresses = features["address"].tolist()

    index = {address: i for i, address in enumerate(addresses)}
    strength = {rule_id: np.zeros(len(addresses), dtype=float) for rule_id in RULE_IDS}
    apriori: dict[str, float] = {}
    sanctioned_seen: set[str] = set()
    pagerank_seen: set[str] = set()

    r1 = _rule_params(cfg, "R1")
    low = int(r1.get("cycle_length_min", 2))
    high = int(r1.get("cycle_length_max", 5))
    r3 = _rule_params(cfg, "R3")
    r3_min_strength = float(r3.get("min_strength", 0.0))
    r4 = _rule_params(cfg, "R4")
    r4_min_strength = float(r4.get("min_strength", 0.05))
    r5 = _rule_params(cfg, "R5")
    confidence_min = float(r5.get("min_confidence", 0.8))

    for (address, signal), payload in flags.items():
        i = index.get(address)
        if i is None:
            continue
        if signal == "cycle_member":
            try:
                length = int(payload.get("signal_value", 0))
            except (TypeError, ValueError):
                length = 0
            if low <= length <= high:
                strength["R1"][i] = max(strength["R1"][i], _payload_strength(payload))
        elif signal == "fanout":
            strength["R2"][i] = max(strength["R2"][i], _payload_strength(payload))
        elif signal == "sanctioned_1hop":
            sanctioned_seen.add(address)
            value = _payload_strength(payload)
            evidence = payload.get("evidence") or {}
            if "strength" not in payload and isinstance(evidence, dict) and "exposure_fraction" in evidence:
                try:
                    value = max(
                        float(evidence["exposure_fraction"]),
                        float(r3.get("min_direct_strength", 0.0)),
                    )
                except (TypeError, ValueError):
                    pass
            strength["R3"][i] = max(strength["R3"][i], min(1.0, value))
        elif signal == "pagerank_top5":
            pagerank_seen.add(address)
            strength["R4"][i] = max(strength["R4"][i], _payload_strength(payload))
        elif signal == "apriori_match":
            confidence = float(payload.get("signal_value", 0.0))
            apriori[address] = confidence
            if confidence >= confidence_min:
                strength["R5"][i] = 1.0

    # Continuous rules can carry a sub-point strength; drop the negligible tail.
    for rule_id, minimum in (("R3", r3_min_strength), ("R4", r4_min_strength)):
        if minimum > 0:
            strength[rule_id] = np.where(strength[rule_id] >= minimum, strength[rule_id], 0.0)

    hits = {rule_id: strength[rule_id] > 0 for rule_id in RULE_IDS}
    weights = {rule.id: rule.weight for rule in cfg.rules}
    contributions = {
        rule_id: np.round(strength[rule_id] * weights[rule_id], 6) for rule_id in RULE_IDS
    }
    raw_scores = np.round(sum(contributions.values()), 6)

    category_by_address: dict[str, str] = {}
    if known_addresses is not None and not known_addresses.empty:
        for row in known_addresses.itertuples(index=False):
            category_by_address[row.address] = row.category

    scores: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for i, address in enumerate(addresses):
        raw = round(float(raw_scores[i]), 6)
        category = category_by_address.get(address)
        downweighted = bool(category and category in cfg.known_address_categories)
        factor = cfg.known_address_factor if downweighted else 1.0
        adjusted = round(raw * factor, 6)
        clamped = adjusted > cfg.max_score
        final = cfg.max_score if clamped else int(round(adjusted))
        tier = tier_for_score(final, cfg)

        rule_traces: dict[str, Any] = {}
        for rule in cfg.rules:
            hit = bool(hits[rule.id][i])
            rule_traces[rule.id] = {
                "id": rule.id,
                "name": rule.name,
                "hit": hit,
                "weight": rule.weight,
                "strength": round(float(strength[rule.id][i]), 6),
                "contribution": round(float(contributions[rule.id][i]), 6),
                "inputs": _json_safe(flags.get((address, SIGNAL_BY_RULE[rule.id]), {}).get("evidence")) if hit else None,
            }
            if rule.id == "R5":
                rule_traces[rule.id]["confidence_min"] = confidence_min
                if not hit and address in apriori:
                    rule_traces[rule.id]["gate"] = (
                        f"confidence {apriori[address]:.3f} < {confidence_min:.2f}"
                    )
            elif rule.id in ("R3", "R4") and not hit:
                seen = sanctioned_seen if rule.id == "R3" else pagerank_seen
                minimum = r3_min_strength if rule.id == "R3" else r4_min_strength
                if address in seen:
                    rule_traces[rule.id]["gate"] = (
                        f"strength {float(strength[rule.id][i]):.3f} < {minimum:.3f}"
                    )

        if raw > 0:
            top_index = int(np.argmax([contributions[rule_id][i] for rule_id in RULE_IDS]))
            top_driver = RULE_IDS[top_index]
        else:
            top_driver = None

        trace = {
            "rules_version": cfg.rules_version,
            "rules_version_label": cfg.rules_version_label,
            "scaler_version": cfg.scaler_version,
            "engine_version": "v2",
            "evaluated_at": evaluated_at,
            "raw_score": raw,
            "adjusted_raw_score": adjusted,
            "score": final,
            "tier": tier,
            "clamped": clamped,
            "max_score": cfg.max_score,
            "top_driver": top_driver,
            "downweight": {
                "applied": downweighted,
                "category": category if downweighted else None,
                "factor": factor,
                "pre_downweight_raw": raw,
                "post_downweight_raw": adjusted,
            },
            "truncated": False,
            "rules": rule_traces,
        }
        scores.append({
            "address": address,
            "raw_score": raw,
            "score": final,
            "tier": tier,
            "top_driver": top_driver,
            "rules_version": cfg.rules_version,
            "rules_version_label": cfg.rules_version_label,
            "scaler_version": cfg.scaler_version,
        })
        traces.append({
            "address": address,
            "rules_version": cfg.rules_version,
            "trace": trace,
        })

    scores_df = pd.DataFrame(scores).sort_values(
        ["score", "address"], ascending=[False, True]
    ).reset_index(drop=True)
    traces_df = pd.DataFrame(traces)
    return scores_df, traces_df


def validate_trace(trace: dict[str, Any]) -> None:
    raw = float(trace["raw_score"])
    contribution_sum = sum(float(rule["contribution"]) for rule in trace["rules"].values())
    if not math.isclose(contribution_sum, raw, abs_tol=1e-5):
        raise RuntimeError("trace contributions do not sum to raw_score")
    factor = float(trace["downweight"]["factor"])
    if not math.isclose(float(trace["adjusted_raw_score"]), raw * factor, abs_tol=1e-5):
        raise RuntimeError("adjusted_raw_score != raw_score * factor")
    if not math.isclose(float(trace["downweight"]["pre_downweight_raw"]), raw, abs_tol=1e-5):
        raise RuntimeError("downweight.pre_downweight_raw != raw_score")
    if bool(trace["clamped"]) != (raw * factor > int(trace["max_score"])):
        raise RuntimeError("clamped flag disagrees with max_score")
    if not trace["clamped"] and int(trace["score"]) != int(round(raw * factor)):
        raise RuntimeError("score disagrees with rounded adjusted raw")


def traces_to_frame(traces: pd.DataFrame) -> pd.DataFrame:
    out = traces.copy()
    out["trace_json"] = out["trace"].map(lambda value: json.dumps(_json_safe(value), default=str))
    return out.drop(columns=["trace"])


def traces_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["trace"] = out["trace_json"].map(json.loads)
    return out.drop(columns=["trace_json"])


def score_summary(scores: pd.DataFrame, traces: pd.DataFrame) -> dict[str, Any]:
    tiers = scores["tier"].value_counts().to_dict()
    top = scores.head(10)[["address", "score", "tier", "top_driver"]].to_dict(orient="records")
    return {
        "total_scored": int(len(scores)),
        "tiers": {k: int(v) for k, v in tiers.items()},
        "top_10": top,
        "clamped_count": int(traces["trace"].map(lambda t: bool(t["clamped"])).sum()),
        "downweighted_count": int(traces["trace"].map(lambda t: bool(t["downweight"]["applied"])).sum()),
    }
