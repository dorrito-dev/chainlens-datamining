"""Phase 5 locked rule engine and reconstructable score traces.

Rule evaluation is vectorised over addresses: the five rules are evaluated as
pandas Series, then the per-address trace records are serialised. There is no
per-address rule scan or graph recomputation in the scoring path.
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

ENGINE_VERSION = "phase5-v1"
RULE_IDS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
SIGNAL_BY_RULE: dict[str, str] = {
    "R1": "cycle_member",
    "R2": "fanout",
    "R3": "sanctioned_1hop",
    "R4": "pagerank_top5",
    "R5": "apriori_match",
}
MAX_TRACE_CYCLES = 5
MAX_TRACE_STRING = 1_000

REQUIRED_TRACE_KEYS = {
    "rules_version",
    "rules_version_label",
    "scaler_version",
    "engine_version",
    "evaluated_at",
    "raw_score",
    "adjusted_raw_score",
    "score",
    "tier",
    "clamped",
    "max_score",
    "top_driver",
    "downweight",
    "truncated",
    "rules",
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
    known_address_factor: float
    known_address_categories: set[str]
    rules: tuple[RuleSpec, ...]
    scaler_version: str = ""


def compute_rules_version(yaml_text: str, scaler_version: str) -> str:
    """Return the provenance hash for a rules/scaler pair.

    This is a version identifier, not a security primitive. The raw YAML text
    is hashed intentionally so comments and whitespace changes are visible in
    the version.
    """
    payload = yaml_text + "|" + scaler_version
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_rules(path: str | Path, scaler_version: str) -> RulesConfig:
    """Read and validate ``rules.yaml`` and compute its version hash."""
    rules_path = Path(path)
    if not rules_path.exists():
        raise RuntimeError(f"Missing rules config: {rules_path}")
    try:
        raw_text = rules_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read rules config {rules_path}: {exc}") from exc

    try:
        parsed = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {rules_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Rules config {rules_path} must be a mapping")

    try:
        label = str(parsed["rules_version_label"])
        max_score = int(parsed["max_score"])
        tiers = parsed["tiers"]
        low_max = int(tiers["low_max"])
        medium_max = int(tiers["medium_max"])
        downweight = parsed["downweight"]
        factor = float(downweight["known_address_factor"])
        categories = {
            str(category).strip().lower()
            for category in downweight["known_address_categories"]
        }
        raw_rules = parsed["rules"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Rules config {rules_path} is missing a required Phase 5 field: {exc}"
        ) from exc

    if max_score <= 0:
        raise RuntimeError("max_score must be positive")
    if not (0 <= low_max < medium_max <= max_score):
        raise RuntimeError(
            "tier thresholds must satisfy 0 <= low_max < medium_max <= max_score"
        )
    if not (0.0 < factor <= 1.0):
        raise RuntimeError("known_address_factor must be in (0, 1]")
    if not categories:
        raise RuntimeError("known_address_categories must not be empty")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuntimeError("rules must be a non-empty list")

    rules: list[RuleSpec] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            raise RuntimeError("each rule must be a mapping")
        try:
            rule_id = str(item["id"])
            name = str(item["name"])
            weight = int(item["weight"])
            params = item.get("params") or {}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid rule entry: {exc}") from exc
        if not isinstance(params, dict):
            raise RuntimeError(f"params for {rule_id} must be a mapping")
        if weight < 0:
            raise RuntimeError(f"weight for {rule_id} must be non-negative")
        rules.append(RuleSpec(id=rule_id, name=name, weight=weight, params=dict(params)))

    ids = tuple(rule.id for rule in rules)
    if ids != RULE_IDS:
        raise RuntimeError(
            f"rules.yaml must contain exactly {RULE_IDS} in order; found {ids}"
        )

    return RulesConfig(
        rules_version=compute_rules_version(raw_text, scaler_version),
        rules_version_label=label,
        max_score=max_score,
        tier_low_max=low_max,
        tier_medium_max=medium_max,
        known_address_factor=factor,
        known_address_categories=categories,
        rules=tuple(rules),
        scaler_version=scaler_version,
    )


def tier_for_score(score: int, cfg: RulesConfig) -> str:
    """Map a final clamped integer score to its tier."""
    if score <= cfg.tier_low_max:
        return "low"
    if score <= cfg.tier_medium_max:
        return "medium"
    return "high"


def _utc_now_iso() -> str:
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _iso_timestamp(value: str | datetime | None) -> str:
    if value is None:
        return _utc_now_iso()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return (
            value.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    text = str(value).strip()
    if not text:
        return _utc_now_iso()
    return text


def _json_safe(value: Any) -> Any:
    """Convert numpy/Decimal/datetime values and cap long strings."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return value[:MAX_TRACE_STRING]
    return value


def _evidence(
    flag_evidence: dict[tuple[str, str], dict[str, Any]],
    address: str,
    signal_name: str,
) -> dict[str, Any]:
    evidence = flag_evidence.get((address, signal_name))
    if evidence is None:
        raise RuntimeError(
            f"Missing evidence for hit rule input {signal_name!r} on {address}"
        )
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Evidence for {signal_name!r} on {address} is not valid JSON"
            ) from exc
    if not isinstance(evidence, dict):
        raise RuntimeError(
            f"Evidence for {signal_name!r} on {address} must be a JSON object"
        )
    return evidence


def _cycle_inputs(evidence: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    raw_cycles = evidence.get("cycles")
    if not isinstance(raw_cycles, list):
        raw_cycles = [evidence] if evidence.get("cycle_path") else []
    cycles: list[dict[str, Any]] = []
    for raw_cycle in raw_cycles[:MAX_TRACE_CYCLES]:
        if not isinstance(raw_cycle, dict):
            continue
        path = raw_cycle.get("cycle_path", raw_cycle.get("path", []))
        path = [str(item) for item in path] if isinstance(path, list) else []
        values = raw_cycle.get("edge_values", raw_cycle.get("edge_values_eth", []))
        values = [float(item) for item in values] if isinstance(values, list) else []
        ratio = raw_cycle.get("max_min_ratio")
        if ratio is None and values and min(values) > 0:
            ratio = max(values) / min(values)
        cycles.append(
            {
                "path": path,
                "length": int(raw_cycle.get("cycle_length") or len(path)),
                "edge_values_eth": values,
                "max_min_ratio": (
                    round(float(ratio), 6) if ratio is not None else None
                ),
                "time_span_hours": (
                    float(raw_cycle["time_span_hours"])
                    if raw_cycle.get("time_span_hours") is not None
                    else None
                ),
            }
        )

    reported_count = evidence.get("cycle_count")
    try:
        cycle_count = int(reported_count) if reported_count is not None else len(raw_cycles)
    except (TypeError, ValueError):
        cycle_count = len(raw_cycles)
    cycle_count = max(cycle_count, len(raw_cycles))
    truncated = cycle_count > MAX_TRACE_CYCLES
    return {"cycle_count": cycle_count, "cycles": cycles}, truncated


def _rule_inputs(
    rule_id: str,
    evidence: dict[str, Any],
    cfg: RulesConfig,
) -> tuple[dict[str, Any], bool]:
    if rule_id == "R1":
        return _cycle_inputs(evidence)
    if rule_id == "R4":
        inputs = _json_safe(evidence)
        inputs["percentile"] = cfg.rules[3].params.get("percentile", 95)
        return inputs, False
    return _json_safe(evidence), False


def _normalise_flags_pivot(flags_pivot: pd.DataFrame) -> pd.DataFrame:
    pivot = flags_pivot.copy()
    pivot.index = [str(address).lower() for address in pivot.index]
    if pivot.index.duplicated().any():
        raise RuntimeError("flags_pivot contains duplicate addresses")
    for column in pivot.columns:
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce").fillna(0)
    return pivot


def _normalise_known(known: pd.DataFrame) -> pd.Series:
    if known.empty:
        return pd.Series(dtype=object)
    frame = known.copy()
    if "address" not in frame.columns or "category" not in frame.columns:
        raise RuntimeError("known frame must contain address and category columns")
    frame["address"] = frame["address"].astype(str).str.lower()
    if frame["address"].duplicated().any():
        raise RuntimeError("known_addresses contains duplicate addresses")
    return frame.set_index("address")["category"].astype(object)


def evaluate(
    features: pd.DataFrame,
    flags_pivot: pd.DataFrame,
    flag_evidence: dict[tuple[str, str], dict[str, Any]],
    known: pd.DataFrame,
    cfg: RulesConfig,
    evaluated_at: str | datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply R1-R5 to every feature address and return scores and traces."""
    if "address" not in features.columns:
        raise RuntimeError("features frame must contain an address column")
    addresses = features["address"].astype(str).str.lower().tolist()
    if not addresses:
        raise RuntimeError("features frame is empty")
    if len(set(addresses)) != len(addresses):
        raise RuntimeError("features frame contains duplicate addresses")

    pivot = _normalise_flags_pivot(flags_pivot).reindex(addresses).fillna(0)
    known_categories = _normalise_known(known).reindex(addresses)
    evaluated_at_iso = _iso_timestamp(evaluated_at)

    hits: dict[str, pd.Series] = {}
    inputs_by_rule: dict[str, dict[str, dict[str, Any]]] = {}
    truncated_by_address: dict[str, bool] = {address: False for address in addresses}

    for rule in cfg.rules:
        signal_name = SIGNAL_BY_RULE[rule.id]
        if signal_name in pivot.columns:
            values = pd.to_numeric(pivot[signal_name], errors="coerce").fillna(0)
        else:
            values = pd.Series(0.0, index=addresses)

        if rule.id == "R1":
            low = int(rule.params.get("cycle_length_min", 3))
            high = int(rule.params.get("cycle_length_max", 5))
            hit = values.gt(0) & values.between(low, high, inclusive="both")
        elif rule.id == "R4":
            hit = values.eq(1)
        else:
            hit = values.gt(0)

        hits[rule.id] = hit.astype(bool)
        rule_inputs: dict[str, dict[str, Any]] = {}
        for address in hits[rule.id].index[hits[rule.id]]:
            evidence = _evidence(flag_evidence, address, signal_name)
            inputs, truncated = _rule_inputs(rule.id, evidence, cfg)
            rule_inputs[address] = inputs
            if truncated:
                truncated_by_address[address] = True
        inputs_by_rule[rule.id] = rule_inputs

    contributions = pd.DataFrame(
        {
            rule.id: hits[rule.id].astype(int) * rule.weight
            for rule in cfg.rules
        },
        index=addresses,
    )
    raw_score = contributions.sum(axis=1).astype(int)

    applied = known_categories.isin(cfg.known_address_categories).fillna(False)
    factors = pd.Series(1.0, index=addresses)
    factors.loc[applied] = cfg.known_address_factor
    adjusted_raw_score = raw_score.astype(float) * factors
    clamped = adjusted_raw_score > cfg.max_score
    rounded = np.rint(adjusted_raw_score.to_numpy(dtype=float)).astype(int)
    final_scores = np.where(clamped.to_numpy(), cfg.max_score, rounded)

    top_drivers = (
        contributions.idxmax(axis=1).where(raw_score.gt(0), None).tolist()
    )

    score_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for position, address in enumerate(addresses):
        raw = int(raw_score.loc[address])
        adjusted = float(adjusted_raw_score.loc[address])
        score = int(final_scores[position])
        is_clamped = bool(clamped.loc[address])
        is_downweighted = bool(applied.loc[address])
        category = (
            str(known_categories.loc[address])
            if is_downweighted and not pd.isna(known_categories.loc[address])
            else None
        )
        factor = float(factors.loc[address])
        tier = tier_for_score(score, cfg)
        top_driver = top_drivers[position]

        rule_traces: dict[str, dict[str, Any]] = {}
        for rule in cfg.rules:
            hit = bool(hits[rule.id].loc[address])
            contribution = int(contributions.loc[address, rule.id])
            rule_traces[rule.id] = {
                "id": rule.id,
                "name": rule.name,
                "hit": hit,
                "weight": rule.weight,
                "contribution": contribution,
                "inputs": inputs_by_rule[rule.id].get(address) if hit else None,
            }

        trace = {
            "rules_version": cfg.rules_version,
            "rules_version_label": cfg.rules_version_label,
            "scaler_version": cfg.scaler_version,
            "engine_version": ENGINE_VERSION,
            "evaluated_at": evaluated_at_iso,
            "raw_score": raw,
            "adjusted_raw_score": adjusted,
            "score": score,
            "tier": tier,
            "clamped": is_clamped,
            "max_score": cfg.max_score,
            "top_driver": top_driver,
            "downweight": {
                "applied": is_downweighted,
                "category": category,
                "factor": factor,
                "pre_downweight_raw": raw,
                "post_downweight_raw": adjusted,
            },
            "truncated": truncated_by_address[address],
            "rules": rule_traces,
        }
        trace_rows.append(
            {
                "address": address,
                "rules_version": cfg.rules_version,
                "trace": trace,
            }
        )
        score_rows.append(
            {
                "address": address,
                "raw_score": raw,
                "score": score,
                "tier": tier,
                "top_driver": top_driver,
                "rules_version": cfg.rules_version,
                "adjusted_raw_score": adjusted,
            }
        )

    return (
        pd.DataFrame(score_rows).sort_values("address").reset_index(drop=True),
        pd.DataFrame(trace_rows).sort_values("address").reset_index(drop=True),
    )


def validate_trace(trace: dict[str, Any], cfg: RulesConfig) -> None:
    """Raise a named error if a trace violates the Phase 5 contract."""
    missing = REQUIRED_TRACE_KEYS - set(trace)
    if missing:
        raise RuntimeError(f"trace missing keys: {sorted(missing)}")
    rules = trace["rules"]
    if not isinstance(rules, dict) or tuple(rules) != RULE_IDS:
        raise RuntimeError(f"trace rules must be exactly {RULE_IDS}; got {list(rules)}")

    contributions = [int(rules[rule_id]["contribution"]) for rule_id in RULE_IDS]
    raw_score = int(trace["raw_score"])
    if sum(contributions) != raw_score:
        raise RuntimeError(
            f"sum(contributions)={sum(contributions)} != raw_score={raw_score}"
        )

    for rule_id in RULE_IDS:
        rule = rules[rule_id]
        if rule["id"] != rule_id:
            raise RuntimeError(f"rule key/id mismatch for {rule_id}")
        if bool(rule["hit"]) and rule["inputs"] is None:
            raise RuntimeError(f"hit rule {rule_id} has inputs=null")
        if not bool(rule["hit"]) and rule["inputs"] is not None:
            raise RuntimeError(f"non-hit rule {rule_id} has non-null inputs")
        if not bool(rule["hit"]) and int(rule["contribution"]) != 0:
            raise RuntimeError(f"non-hit rule {rule_id} has a contribution")

    adjusted = float(trace["adjusted_raw_score"])
    factor = float(trace["downweight"]["factor"])
    if not math.isclose(adjusted, raw_score * factor, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("adjusted_raw_score != raw_score * downweight.factor")
    if trace["downweight"]["pre_downweight_raw"] != raw_score:
        raise RuntimeError("downweight.pre_downweight_raw != raw_score")
    if not math.isclose(
        float(trace["downweight"]["post_downweight_raw"]),
        adjusted,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("downweight.post_downweight_raw != adjusted_raw_score")

    max_score = int(trace["max_score"])
    clamped = bool(trace["clamped"])
    if clamped != (adjusted > max_score):
        raise RuntimeError("clamped does not match adjusted_raw_score > max_score")
    expected_score = max_score if clamped else int(round(adjusted))
    if int(trace["score"]) != expected_score:
        raise RuntimeError(
            f"score={trace['score']} != expected {expected_score}"
        )
    if trace["tier"] != tier_for_score(int(trace["score"]), cfg):
        raise RuntimeError("tier does not match final score")

    best_contribution = max(contributions)
    expected_driver = (
        next(
            rule_id
            for rule_id, contribution in zip(RULE_IDS, contributions, strict=True)
            if contribution == best_contribution
        )
        if best_contribution > 0
        else None
    )
    if trace["top_driver"] != expected_driver:
        raise RuntimeError(
            f"top_driver={trace['top_driver']!r} != {expected_driver!r}"
        )
    if trace["rules_version"] != cfg.rules_version:
        raise RuntimeError("trace rules_version does not match config")
    if trace["rules_version_label"] != cfg.rules_version_label:
        raise RuntimeError("trace rules_version_label does not match config")
    if int(trace["max_score"]) != cfg.max_score:
        raise RuntimeError("trace max_score does not match config")


def validate_traces(traces: pd.DataFrame, cfg: RulesConfig) -> None:
    for row in traces.itertuples(index=False):
        try:
            validate_trace(row.trace, cfg)
        except RuntimeError as exc:
            raise RuntimeError(f"Invalid trace for {row.address}: {exc}") from exc
