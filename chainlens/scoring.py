"""Phase 5 scoring pipeline helpers.

The CLI entrypoint is ``pipeline/06_score.py``. Keeping the DB orchestration
here makes the pre-flight, upsert and summary logic directly importable by the
test suite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from chainlens.config import project_path
from chainlens.db import apply_schema, get_pg_engine
from chainlens.fixtures import address_for
from chainlens.rules import (
    ENGINE_VERSION,
    RulesConfig,
    evaluate,
    load_rules,
    validate_traces,
)
from chainlens.seed_scores import clear_seed_scores

RULES_PATH = project_path("config/rules.yaml")
SCORE_SUMMARY_PATH = project_path("data/processed/score_summary.json")
UPSERT_CHUNK_SIZE = 500


def _preflight(engine: Engine) -> tuple[RulesConfig, str, int]:
    """Validate Phase 5 inputs and return ``(config, scaler_version, rows)``."""
    with engine.connect() as conn:
        feature_rows = int(
            conn.execute(text("SELECT count(*) FROM address_features")).scalar_one()
        )
        if feature_rows == 0:
            raise RuntimeError(
                "address_features is empty; run `make features` before `make score`"
            )
        flag_rows = int(
            conn.execute(text("SELECT count(*) FROM address_flags")).scalar_one()
        )
        if flag_rows == 0:
            raise RuntimeError(
                "address_flags is empty; run `make detect` before `make score`"
            )
        dbscan_rows = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM address_flags "
                    "WHERE signal_name = 'dbscan_cluster'"
                )
            ).scalar_one()
        )
        if dbscan_rows == 0:
            raise RuntimeError(
                "address_flags has no dbscan_cluster sentinel; run `make detect`"
            )
        known_rows = int(
            conn.execute(text("SELECT count(*) FROM known_addresses")).scalar_one()
        )
        if known_rows == 0:
            raise RuntimeError(
                "known_addresses is empty; run `make ingest` before `make score`"
            )
        scaler_rows = int(
            conn.execute(text("SELECT count(*) FROM feature_scaler")).scalar_one()
        )
        if scaler_rows == 0:
            raise RuntimeError(
                "feature_scaler is empty; run `make features` before `make score`"
            )

        feature_versions = set(
            conn.execute(
                text(
                    "SELECT DISTINCT scaler_version FROM address_features "
                    "WHERE scaler_version IS NOT NULL"
                )
            ).scalars()
        )
        null_feature_versions = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM address_features "
                    "WHERE scaler_version IS NULL"
                )
            ).scalar_one()
        )
        scaler_versions = set(
            conn.execute(
                text("SELECT DISTINCT scaler_version FROM feature_scaler")
            ).scalars()
        )

    if null_feature_versions:
        raise RuntimeError(
            f"{null_feature_versions} address_features row(s) have no scaler_version; "
            "run `make features`"
        )
    if len(feature_versions) != 1:
        raise RuntimeError(
            "address_features contains multiple scaler versions: "
            f"{sorted(feature_versions)}"
        )
    if len(scaler_versions) != 1:
        raise RuntimeError(
            "feature_scaler contains multiple scaler versions: "
            f"{sorted(scaler_versions)}"
        )
    feature_version = str(next(iter(feature_versions)))
    scaler_version = str(next(iter(scaler_versions)))
    if feature_version != scaler_version:
        raise RuntimeError(
            "scaler provenance mismatch: address_features="
            f"{feature_version!r}, feature_scaler={scaler_version!r}"
        )

    cfg = load_rules(RULES_PATH, scaler_version)
    return cfg, scaler_version, feature_rows


def _normalise_evidence(value: Any, signal_name: str, address: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON evidence for {signal_name!r} on {address}"
            ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Evidence for {signal_name!r} on {address} must be a JSON object"
        )
    return value


def _load_inputs(
    engine: Engine,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], dict],
    pd.DataFrame,
    datetime | None,
]:
    with engine.connect() as conn:
        features = pd.read_sql(
            text("SELECT address FROM address_features ORDER BY address"), conn
        )
        flags = pd.read_sql(
            text(
                "SELECT address, signal_name, signal_value, evidence "
                "FROM address_flags ORDER BY address, signal_name"
            ),
            conn,
        )
        known = pd.read_sql(
            text("SELECT address, category FROM known_addresses ORDER BY address"),
            conn,
        )
        evaluated_at = conn.execute(
            text("SELECT max(created_at) FROM address_flags")
        ).scalar_one()

    features["address"] = features["address"].astype(str).str.lower()
    flags["address"] = flags["address"].astype(str).str.lower()
    flags["signal_name"] = flags["signal_name"].astype(str)
    flags["signal_value"] = pd.to_numeric(flags["signal_value"], errors="coerce")
    known["address"] = known["address"].astype(str).str.lower()
    known["category"] = known["category"].astype(str).str.lower()

    flags_pivot = (
        flags.pivot(index="address", columns="signal_name", values="signal_value")
        .fillna(0)
    )
    flag_evidence: dict[tuple[str, str], dict] = {}
    for row in flags.itertuples(index=False):
        flag_evidence[(row.address, row.signal_name)] = _normalise_evidence(
            row.evidence, row.signal_name, row.address
        )
    return features, flags_pivot, flag_evidence, known, evaluated_at


def _chunks(items: list[dict[str, Any]], size: int = UPSERT_CHUNK_SIZE):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _upsert_scores(
    engine: Engine,
    scores: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    scored_at: datetime,
) -> None:
    metadata = MetaData()
    risk_table = Table("risk_scores", metadata, autoload_with=engine)
    trace_table = Table("score_trace", metadata, autoload_with=engine)

    risk_insert = pg_insert(risk_table)
    risk_statement = risk_insert.on_conflict_do_update(
        index_elements=[risk_table.c.address],
        set_={
            "raw_score": risk_insert.excluded.raw_score,
            "score": risk_insert.excluded.score,
            "tier": risk_insert.excluded.tier,
            "top_driver": risk_insert.excluded.top_driver,
            "rules_version": risk_insert.excluded.rules_version,
            "scored_at": risk_insert.excluded.scored_at,
        },
    )
    trace_insert = pg_insert(trace_table)
    trace_statement = trace_insert.on_conflict_do_update(
        index_elements=[trace_table.c.address, trace_table.c.rules_version],
        set_={
            "trace": trace_insert.excluded.trace,
            "created_at": trace_insert.excluded.created_at,
        },
    )

    risk_rows = [
        {
            "address": row.address,
            "raw_score": int(row.raw_score),
            "score": int(row.score),
            "tier": row.tier,
            "top_driver": row.top_driver,
            "rules_version": row.rules_version,
            "scored_at": scored_at,
        }
        for row in scores.itertuples(index=False)
    ]
    trace_rows = [
        {
            "address": row.address,
            "rules_version": row.rules_version,
            "trace": row.trace,
            "created_at": scored_at,
        }
        for row in traces.itertuples(index=False)
    ]

    with engine.begin() as conn:
        for chunk in _chunks(risk_rows):
            conn.execute(risk_statement, chunk)
        for chunk in _chunks(trace_rows):
            conn.execute(trace_statement, chunk)
        unscored = conn.execute(
            text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
        ).scalar_one()
        if unscored:
            raise RuntimeError(
                f"Score write failed: {unscored} tier='unscored' row(s) remain"
            )


def _build_summary(
    *,
    cfg: RulesConfig,
    scaler_version: str,
    scores: pd.DataFrame,
    traces: pd.DataFrame,
    seed_rows_deleted: int,
) -> dict[str, Any]:
    ranked = scores.sort_values(
        ["score", "raw_score", "address"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    tier_counts = {
        tier: int((scores["tier"] == tier).sum())
        for tier in ("low", "medium", "high")
    }
    driver_counts = (
        scores["top_driver"].fillna("none").value_counts().sort_index().to_dict()
    )
    top_10 = [
        {
            "address": row.address,
            "score": int(row.score),
            "raw_score": int(row.raw_score),
            "tier": row.tier,
            "top_driver": row.top_driver,
        }
        for row in ranked.head(10).itertuples(index=False)
    ]

    exchange_address = address_for("EXCHANGE_HOT")
    exchange_row = ranked[ranked["address"] == exchange_address]
    exchange_hot = None
    if not exchange_row.empty:
        exchange = exchange_row.iloc[0]
        exchange_trace = traces.loc[
            traces["address"] == exchange_address, "trace"
        ].iloc[0]
        exchange_hot = {
            "address": exchange_address,
            "score": int(exchange["score"]),
            "raw_score": int(exchange["raw_score"]),
            "rank": int(exchange["rank"]),
            "tier": exchange["tier"],
            "top_driver": exchange["top_driver"],
            "downweight": exchange_trace["downweight"],
        }

    trace_values = [trace for trace in traces["trace"]]
    return {
        "rules_version": cfg.rules_version,
        "rules_version_label": cfg.rules_version_label,
        "scaler_version": scaler_version,
        "engine_version": ENGINE_VERSION,
        "evaluated_at": (
            trace_values[0]["evaluated_at"] if trace_values else None
        ),
        "row_count": int(len(scores)),
        "seed_rows_deleted": int(seed_rows_deleted),
        "unscored_after": int((scores["tier"] == "unscored").sum()),
        "counts_by_tier": tier_counts,
        "counts_by_top_driver": {str(key): int(value) for key, value in driver_counts.items()},
        "top_10": top_10,
        "exchange_hot": exchange_hot,
        "clamped_count": int(
            sum(1 for trace in trace_values if bool(trace["clamped"]))
        ),
        "downweighted_count": int(
            sum(
                1
                for trace in trace_values
                if bool(trace["downweight"]["applied"])
            )
        ),
    }


def _write_summary(summary: dict[str, Any]) -> Path:
    SCORE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return SCORE_SUMMARY_PATH


def run_score() -> dict[str, Any]:
    """Run the complete Phase 5 scoring path and return the summary."""
    engine = get_pg_engine()
    apply_schema(engine)
    cfg, scaler_version, feature_rows = _preflight(engine)
    seed_rows_deleted = clear_seed_scores(engine)

    features, flags_pivot, flag_evidence, known, evaluated_at = _load_inputs(engine)
    scores, traces = evaluate(
        features,
        flags_pivot,
        flag_evidence,
        known,
        cfg,
        evaluated_at=evaluated_at,
    )
    validate_traces(traces, cfg)

    if len(scores) != feature_rows:
        raise RuntimeError(
            f"Scored {len(scores)} address(es), expected {feature_rows} feature rows"
        )
    scored_at = evaluated_at or datetime.now(UTC)
    _upsert_scores(engine, scores, traces, scored_at=scored_at)

    with engine.connect() as conn:
        score_count, distinct_scores = conn.execute(
            text("SELECT count(*), count(DISTINCT address) FROM risk_scores")
        ).one()
        trace_count = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM score_trace WHERE rules_version = :version"
                ),
                {"version": cfg.rules_version},
            ).scalar_one()
        )
        unscored = int(
            conn.execute(
                text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
            ).scalar_one()
        )
        seed_rows = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM risk_scores "
                    "WHERE rules_version = 'seed-phase4'"
                )
            ).scalar_one()
        )

    if int(score_count) != feature_rows or int(distinct_scores) != feature_rows:
        raise RuntimeError(
            "risk_scores row count mismatch: "
            f"rows={score_count}, distinct={distinct_scores}, features={feature_rows}"
        )
    if trace_count != feature_rows:
        raise RuntimeError(
            f"score_trace row count mismatch: traces={trace_count}, features={feature_rows}"
        )
    if unscored or seed_rows:
        raise RuntimeError(
            f"Post-score invariant failed: unscored={unscored}, seed_rows={seed_rows}"
        )

    summary = _build_summary(
        cfg=cfg,
        scaler_version=scaler_version,
        scores=scores,
        traces=traces,
        seed_rows_deleted=seed_rows_deleted,
    )
    summary_path = _write_summary(summary)
    print(f"Score summary: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def print_score_check() -> int:
    """Print provenance/counts and verify the Phase 5 postconditions."""
    engine = get_pg_engine()
    with engine.connect() as conn:
        feature_count = int(
            conn.execute(text("SELECT count(*) FROM address_features")).scalar_one()
        )
        score_count, distinct_scores = conn.execute(
            text("SELECT count(*), count(DISTINCT address) FROM risk_scores")
        ).one()
        unscored = int(
            conn.execute(
                text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
            ).scalar_one()
        )
        seed_rows = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM risk_scores "
                    "WHERE rules_version = 'seed-phase4'"
                )
            ).scalar_one()
        )
        risk_versions = list(
            conn.execute(
                text("SELECT DISTINCT rules_version FROM risk_scores ORDER BY 1")
            ).scalars()
        )
        trace_count = int(
            conn.execute(text("SELECT count(*) FROM score_trace")).scalar_one()
        )
        trace_versions = list(
            conn.execute(
                text("SELECT DISTINCT rules_version FROM score_trace ORDER BY 1")
            ).scalars()
        )
        current_trace_count = 0
        if len(risk_versions) == 1:
            current_trace_count = int(
                conn.execute(
                    text(
                        "SELECT count(*) FROM score_trace "
                        "WHERE rules_version = :version"
                    ),
                    {"version": risk_versions[0]},
                ).scalar_one()
            )
        tier_counts = dict(
            conn.execute(
                text(
                    "SELECT tier, count(*) FROM risk_scores "
                    "GROUP BY tier ORDER BY tier"
                )
            ).all()
        )

    print(f"address_features rows: {feature_count:,}")
    print(f"risk_scores rows: {score_count:,} (distinct {distinct_scores:,})")
    print(f"risk_scores versions: {risk_versions}")
    print(f"tier counts: {tier_counts}")
    print(f"tier='unscored' rows: {unscored}")
    print(f"seed-phase4 rows: {seed_rows}")
    print(f"score_trace rows: {trace_count:,}; versions: {trace_versions}")
    print(f"current-version trace rows: {current_trace_count:,}")

    failures: list[str] = []
    if int(score_count) == 0:
        failures.append("risk_scores is empty; run `make score`")
    if int(score_count) != feature_count or int(distinct_scores) != feature_count:
        failures.append("risk_scores does not cover address_features exactly once")
    if int(unscored) != 0:
        failures.append("tier='unscored' rows remain")
    if int(seed_rows) != 0:
        failures.append("seed-phase4 rows remain")
    if trace_count == 0:
        failures.append("score_trace is empty")
    if current_trace_count != int(score_count):
        failures.append("current-version score_trace does not cover risk_scores")
    if len(risk_versions) > 1:
        failures.append("risk_scores contains multiple rules versions")
    unexpected_tiers = set(tier_counts) - {"low", "medium", "high"}
    if unexpected_tiers:
        failures.append(f"unexpected tier values: {sorted(unexpected_tiers)}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("SCORE CHECK OK")
    return 0
