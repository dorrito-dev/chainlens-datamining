"""Shared Phase 4 detection constants, flags writer and config loaders."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import PROJECT_ROOT

DETECTION_CONFIG_PATH = PROJECT_ROOT / "config" / "detection.yaml"
RULES_CONFIG_PATH = PROJECT_ROOT / "config" / "rules.yaml"

ALLOWED_SIGNALS = {
    "dbscan_cluster",
    "dbscan_noise",
    "pagerank",
    "pagerank_top5",
    "cycle_member",
    "fanout",
    "sanctioned_1hop",
    "louvain_community",
    "apriori_match",
}

GRAPH_SIGNALS = {
    "pagerank",
    "pagerank_top5",
    "cycle_member",
    "fanout",
    "sanctioned_1hop",
    "louvain_community",
}

CLUSTER_SIGNALS = {"dbscan_cluster", "dbscan_noise"}
APRIORI_SIGNALS = {"apriori_match"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_detection_config() -> dict[str, Any]:
    return _load_yaml(DETECTION_CONFIG_PATH)


def load_rules_config() -> dict[str, Any]:
    return _load_yaml(RULES_CONFIG_PATH)


def rule_by_id(rule_id: str) -> dict[str, Any]:
    for rule in load_rules_config().get("rules", []):
        if rule.get("id") == rule_id:
            return rule
    raise RuntimeError(f"Missing {rule_id} in config/rules.yaml")


def write_flags(
    engine: Engine,
    rows: Iterable[dict[str, Any]],
    reset_signals: Iterable[str],
) -> int:
    """Replace a signal group atomically, then insert deterministic flag rows.

    Rows are sorted by ``(address, signal_name)`` and duplicates are collapsed
    (last row wins) to respect the ``address_flags`` primary key.
    """
    reset = sorted(set(reset_signals))
    unknown = set(reset) - ALLOWED_SIGNALS
    if unknown:
        raise RuntimeError(f"Unknown signal names in reset: {sorted(unknown)}")

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        signal_name = str(row["signal_name"])
        if signal_name not in ALLOWED_SIGNALS:
            raise RuntimeError(f"Signal {signal_name!r} is not allowed")
        evidence = row.get("evidence") or {}
        if not isinstance(evidence, dict):
            raise RuntimeError(f"Evidence for {signal_name} must be a dict")
        address = str(row["address"]).lower()
        value = row.get("signal_value")
        if hasattr(value, "item"):
            value = value.item()
        deduped[(address, signal_name)] = {
            "address": address,
            "signal_name": signal_name,
            "signal_value": value,
            "evidence": json.dumps(evidence, sort_keys=True, default=str),
        }

    ordered = [deduped[key] for key in sorted(deduped)]
    with engine.begin() as conn:
        if reset:
            conn.execute(
                text(
                    "DELETE FROM address_flags "
                    "WHERE signal_name = ANY(:signals)"
                ),
                {"signals": reset},
            )
        if ordered:
            conn.execute(
                text(
                    """
                    INSERT INTO address_flags
                        (address, signal_name, signal_value, evidence)
                    VALUES
                        (:address, :signal_name, :signal_value,
                         CAST(:evidence AS JSONB))
                    ON CONFLICT (address, signal_name) DO UPDATE SET
                        signal_value = EXCLUDED.signal_value,
                        evidence = EXCLUDED.evidence,
                        created_at = now()
                    """
                ),
                ordered,
            )
    return len(ordered)


def delete_signals(engine: Engine, signals: Iterable[str]) -> None:
    reset = sorted(set(signals))
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM address_flags WHERE signal_name = ANY(:signals)"),
            {"signals": reset},
        )
