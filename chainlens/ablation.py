"""Phase 6 baseline snapshots and per-rule ablation.

The ablation is deliberately *in-memory*: a frozen copy of the loaded rules
config has one weight set to zero, the scorer is re-run against the same
feature/flag inputs, and the result is compared with the committed baseline
snapshot. The live ``risk_scores`` table and ``config/rules.yaml`` are never
mutated. This is stronger than a file-copy/restore dance because there is no
window in which the database is left ablated.

A global rule removal necessarily changes every wallet that hit that rule.
"Other wallets unchanged" therefore means wallets outside that rule's hit set;
the report includes both the hit-set size and the primary target required by
the Phase 6 brief.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import project_path
from chainlens.db import get_pg_engine
from chainlens.fixtures import address_for
from chainlens.manifest import sha256_file
from chainlens.rules import RULE_IDS, RulesConfig, evaluate, load_rules
from chainlens.scoring import RULES_PATH, _load_inputs

BASELINE_DIR = project_path("data/processed")
R4_TARGET_NOTE = (
    "Top-ranked pagerank_top5 wallet in the baseline top 10; all pagerank_top5 "
    "hits are in the rule's affected set."
)


def _parse_trace(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise ValueError(f"Unsupported trace value: {type(value)!r}")


def _trace_json(value: Any) -> str:
    return json.dumps(_parse_trace(value), sort_keys=True, default=str)


def _trace_for_hash(value: Any) -> str:
    """Canonical trace minus the volatile evaluation timestamp."""
    trace = _parse_trace(value)
    trace.pop("evaluated_at", None)
    return json.dumps(trace, sort_keys=True, default=str)


def baseline_content_hash(frame: pd.DataFrame) -> str:
    """Hash score fields and trace semantics, excluding ``evaluated_at``.

    The timestamp records when a scoring run happened; it is not part of the
    score. Excluding it lets a baseline restore check pass after a legitimate
    re-score with identical rule inputs while still covering every score and
    trace field that affects interpretation.
    """
    import hashlib

    ordered = frame.sort_values("address").reset_index(drop=True)
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False):
        trace = _trace_for_hash(row.trace_json)
        digest.update(
            (
                f"{row.address}|{int(row.raw_score)}|{int(row.score)}|"
                f"{row.tier}|{'' if pd.isna(row.top_driver) else row.top_driver}|"
                f"{row.rules_version}|{trace}\n"
            ).encode()
        )
    return digest.hexdigest()


def _current_frame(engine: Engine, rules_version: str | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        versions = list(
            conn.execute(
                text("SELECT DISTINCT rules_version FROM risk_scores ORDER BY 1")
            ).scalars()
        )
        if rules_version is None:
            if len(versions) != 1:
                raise RuntimeError(
                    "Expected exactly one current rules_version in risk_scores; "
                    f"found {versions}"
                )
            rules_version = str(versions[0])
        frame = pd.read_sql(
            text(
                """
                SELECT
                    rs.address,
                    rs.raw_score,
                    rs.score,
                    rs.tier,
                    rs.top_driver,
                    rs.rules_version,
                    st.trace
                FROM risk_scores rs
                JOIN score_trace st
                  ON st.address = rs.address
                 AND st.rules_version = rs.rules_version
                WHERE rs.rules_version = :version
                ORDER BY rs.address
                """
            ),
            conn,
            params={"version": rules_version},
        )
    if frame.empty:
        raise RuntimeError(f"No score/trace rows for rules_version={rules_version!r}")
    frame["trace_json"] = frame["trace"].map(_trace_json)
    return frame[
        [
            "address",
            "raw_score",
            "score",
            "tier",
            "top_driver",
            "rules_version",
            "trace_json",
        ]
    ]


def snapshot_baseline(
    engine: Engine | None = None,
    rules_version: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the Phase 6 baseline parquet and summary JSON."""
    engine = engine or get_pg_engine()
    frame = _current_frame(engine, rules_version)
    if rules_version is None:
        rules_version = str(frame["rules_version"].iloc[0])
    directory = Path(output_dir) if output_dir else BASELINE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / f"score_baseline_{rules_version}.parquet"
    summary_path = directory / f"score_baseline_{rules_version}.summary.json"

    frame.to_parquet(parquet_path, index=False)
    ordered = frame.sort_values(
        ["score", "raw_score", "address"], ascending=[False, False, True]
    ).reset_index(drop=True)
    ordered["rank"] = ordered.index + 1
    tier_counts = {
        tier: int((frame["tier"] == tier).sum())
        for tier in ("low", "medium", "high")
    }
    top_10 = [
        {
            "address": row.address,
            "score": int(row.score),
            "raw_score": int(row.raw_score),
            "tier": row.tier,
            "top_driver": row.top_driver,
            "rank": int(row.rank),
        }
        for row in ordered.head(10).itertuples(index=False)
    ]

    exchange_address = address_for("EXCHANGE_HOT")
    exchange_row = ordered[ordered["address"] == exchange_address]
    exchange_hot = None
    if not exchange_row.empty:
        exchange = exchange_row.iloc[0]
        trace = _parse_trace(exchange.trace_json)
        exchange_hot = {
            "address": exchange_address,
            "score": int(exchange["score"]),
            "raw_score": int(exchange["raw_score"]),
            "rank": int(exchange["rank"]),
            "tier": exchange["tier"],
            "top_driver": exchange["top_driver"],
            "downweight": trace["downweight"],
            "rules": {
                rule_id: {
                    "hit": bool(trace["rules"][rule_id]["hit"]),
                    "contribution": int(trace["rules"][rule_id]["contribution"]),
                }
                for rule_id in RULE_IDS
            },
        }

    first_trace = _parse_trace(frame["trace_json"].iloc[0])
    summary = {
        "rules_version": rules_version,
        "rules_version_label": first_trace["rules_version_label"],
        "scaler_version": first_trace["scaler_version"],
        "engine_version": first_trace["engine_version"],
        "row_count": int(len(frame)),
        "content_sha256": baseline_content_hash(frame),
        "parquet_sha256": sha256_file(parquet_path),
        "parquet_path": str(parquet_path),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts_by_tier": tier_counts,
        "top_10": top_10,
        "exchange_hot": exchange_hot,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {
        "frame": frame,
        "parquet_path": str(parquet_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def baseline_path(rules_version: str | None = None) -> Path:
    if rules_version:
        return BASELINE_DIR / f"score_baseline_{rules_version}.parquet"
    matches = sorted(BASELINE_DIR.glob("score_baseline_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            "No baseline snapshot. Run `make baseline` (or "
            "`pipeline/07_baseline.py`) before ablation."
        )
    return matches[-1]


def load_baseline(path: str | Path | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path) if path else baseline_path())
    required = {
        "address",
        "raw_score",
        "score",
        "tier",
        "top_driver",
        "rules_version",
        "trace_json",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline is missing columns: {sorted(missing)}")
    frame["trace_json"] = frame["trace_json"].map(_trace_json)
    return frame.sort_values("address").reset_index(drop=True)


def _primary_targets(
    rule_id: str, baseline: pd.DataFrame
) -> list[str]:
    if rule_id == "R1":
        return [address_for(name) for name in ("CYCLE_A", "CYCLE_B", "CYCLE_C")]
    if rule_id == "R2":
        return [address_for("FANOUT_WHALE")]
    if rule_id == "R3":
        return [address_for("SANCTIONED_1HOP")]
    if rule_id == "R4":
        ordered = baseline.sort_values(
            ["score", "raw_score", "address"], ascending=[False, False, True]
        )
        for row in ordered.itertuples(index=False):
            trace = _parse_trace(row.trace_json)
            if trace["rules"]["R4"]["hit"] and int(trace["rules"]["R4"]["contribution"]) > 0:
                return [row.address]
        raise RuntimeError("No pagerank_top5 wallet found in the baseline")
    return []


def _ablated_config(cfg: RulesConfig, rule_id: str) -> RulesConfig:
    rules = tuple(
        replace(rule, weight=0) if rule.id == rule_id else rule for rule in cfg.rules
    )
    return replace(
        cfg,
        rules=rules,
        rules_version=f"{cfg.rules_version}-ablate-{rule_id}",
    )


def _affected_addresses(frame: pd.DataFrame, rule_id: str) -> set[str]:
    affected: set[str] = set()
    for row in frame.itertuples(index=False):
        trace = _parse_trace(row.trace_json)
        if trace["rules"][rule_id]["hit"]:
            affected.add(str(row.address))
    return affected


def _compare(
    baseline: pd.DataFrame,
    ablated: pd.DataFrame,
    affected: set[str],
) -> tuple[pd.DataFrame, bool]:
    merged = baseline.merge(
        ablated[["address", "raw_score", "score"]],
        on="address",
        suffixes=("_baseline", "_ablated"),
        how="inner",
    )
    if len(merged) != len(baseline):
        raise RuntimeError("Ablated scores do not cover the baseline addresses")
    merged["other_wallet_changed"] = (
        (~merged["address"].isin(affected))
        & (
            (merged["raw_score_baseline"] != merged["raw_score_ablated"])
            | (merged["score_baseline"] != merged["score_ablated"])
        )
    )
    return merged, bool(merged["other_wallet_changed"].any())


def _trace_rule_contribution(trace_json: str, rule_id: str) -> int:
    return int(_parse_trace(trace_json)["rules"][rule_id]["contribution"])


def run_ablation(
    engine: Engine | None = None,
    baseline: pd.DataFrame | None = None,
    rules_path: str | Path = RULES_PATH,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run R1-R5 ablation and return ``(report, details)``.

    The baseline and current config must agree on ``rules_version``. The config
    file hash is checked in ``finally``; the file is never written.
    """
    engine = engine or get_pg_engine()
    rules_path = Path(rules_path)
    config_hash_before = sha256_file(rules_path)
    try:
        baseline = baseline if baseline is not None else load_baseline()
        baseline_version = str(baseline["rules_version"].iloc[0])
        with engine.connect() as conn:
            scaler_version = str(
                conn.execute(
                    text("SELECT DISTINCT scaler_version FROM feature_scaler")
                ).scalar_one()
            )
        cfg = load_rules(rules_path, scaler_version)
        if cfg.rules_version != baseline_version:
            raise RuntimeError(
                "Baseline/config mismatch: baseline rules_version="
                f"{baseline_version!r}, current config={cfg.rules_version!r}; "
                "re-run `make baseline` after any config change."
            )

        features, flags_pivot, flag_evidence, known, evaluated_at = _load_inputs(engine)
        rows: list[dict[str, Any]] = []
        per_target: dict[str, list[dict[str, Any]]] = {}
        exchange_details: dict[str, Any] | None = None

        for rule in cfg.rules:
            rule_id = rule.id
            primary = _primary_targets(rule_id, baseline)
            if rule_id == "R5":
                rows.append(
                    {
                        "rule_id": rule_id,
                        "target": "untestable on fixture",
                        "baseline_raw": None,
                        "ablated_raw": None,
                        "raw_delta": None,
                        "baseline_score": None,
                        "ablated_score": None,
                        "score_delta": None,
                        "expected_weight": rule.weight,
                        "affected_wallets": 0,
                        "other_wallets_changed": False,
                        "isolated": True,
                        "testable": False,
                        "note": (
                            "No Apriori rules/flags on fixture; covered by "
                            "tests/test_apriori.py."
                        ),
                    }
                )
                continue

            ablated_cfg = _ablated_config(cfg, rule_id)
            ablated_scores, _ = evaluate(
                features,
                flags_pivot,
                flag_evidence,
                known,
                ablated_cfg,
                evaluated_at=evaluated_at,
            )
            affected = _affected_addresses(baseline, rule_id)
            merged, other_changed = _compare(baseline, ablated_scores, affected)
            merged = merged.set_index("address")

            target_details: list[dict[str, Any]] = []
            for address in primary:
                if address not in merged.index:
                    raise RuntimeError(f"Primary target {address} is absent from baseline")
                row = merged.loc[address]
                baseline_raw = int(row["raw_score_baseline"])
                ablated_raw = int(row["raw_score_ablated"])
                target_details.append(
                    {
                        "address": address,
                        "baseline_raw": baseline_raw,
                        "ablated_raw": ablated_raw,
                        "raw_delta": baseline_raw - ablated_raw,
                        "baseline_score": int(row["score_baseline"]),
                        "ablated_score": int(row["score_ablated"]),
                        "score_delta": int(row["score_baseline"]) - int(row["score_ablated"]),
                    }
                )
            per_target[rule_id] = target_details
            target_rows = target_details
            isolated = all(
                detail["raw_delta"] == rule.weight for detail in target_rows
            ) and not other_changed

            first = target_rows[0]
            rows.append(
                {
                    "rule_id": rule_id,
                    "target": ", ".join(primary),
                    "baseline_raw": first["baseline_raw"],
                    "ablated_raw": first["ablated_raw"],
                    "raw_delta": first["raw_delta"],
                    "baseline_score": first["baseline_score"],
                    "ablated_score": first["ablated_score"],
                    "score_delta": first["score_delta"],
                    "expected_weight": rule.weight,
                    "affected_wallets": len(affected),
                    "other_wallets_changed": other_changed,
                    "isolated": isolated,
                    "testable": True,
                    "note": R4_TARGET_NOTE if rule_id == "R4" else "",
                }
            )

            if rule_id == "R2":
                exchange_address = address_for("EXCHANGE_HOT")
                exchange = merged.loc[exchange_address]
                exchange_details = {
                    "address": exchange_address,
                    "baseline_raw": int(exchange["raw_score_baseline"]),
                    "ablated_raw": int(exchange["raw_score_ablated"]),
                    "baseline_adjusted": float(
                        _parse_trace(
                            baseline.set_index("address").loc[
                                exchange_address, "trace_json"
                            ]
                        )["adjusted_raw_score"]
                    ),
                    "ablated_adjusted": round(
                        int(exchange["raw_score_ablated"]) * cfg.known_address_factor, 2
                    ),
                    "baseline_score": int(exchange["score_baseline"]),
                    "ablated_score": int(exchange["score_ablated"]),
                    "raw_delta": int(exchange["raw_score_baseline"])
                    - int(exchange["raw_score_ablated"]),
                    "score_delta": int(exchange["score_baseline"])
                    - int(exchange["score_ablated"]),
                    "arithmetic": (
                        "35 × 0.3 = 10.5 → 10; (35 − 20) × 0.3 = 4.5 → 4; "
                        "final delta = 6, not the raw 20."
                    ),
                }

        report = pd.DataFrame(rows)
        if strict:
            failures = report[
                (report["testable"])
                & (
                    (~report["isolated"])
                    | (report["other_wallets_changed"])
                )
            ]
            if not failures.empty:
                raise RuntimeError(
                    "Ablation isolation failed:\n"
                    + failures[["rule_id", "other_wallets_changed", "isolated"]].to_string(
                        index=False
                    )
                )

        details = {
            "rules_version": cfg.rules_version,
            "scaler_version": scaler_version,
            "baseline_content_sha256": baseline_content_hash(baseline),
            "primary_deltas": per_target,
            "r4_target": primary_by_rule(per_target).get("R4"),
            "exchange_hot_r2": exchange_details,
            "rules_sha256_before": config_hash_before,
        }
        return report, details
    finally:
        config_hash_after = sha256_file(rules_path)
        if config_hash_after != config_hash_before:
            raise RuntimeError(
                "config/rules.yaml changed during ablation; refusing to leave an "
                "ablated config behind"
            )


def primary_by_rule(per_target: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        rule_id: ", ".join(detail["address"] for detail in details)
        for rule_id, details in per_target.items()
    }


def assert_baseline_restored(
    engine: Engine | None = None, baseline: pd.DataFrame | None = None
) -> str:
    """Verify that current risk_scores/traces still match the baseline content."""
    engine = engine or get_pg_engine()
    baseline = baseline if baseline is not None else load_baseline()
    version = str(baseline["rules_version"].iloc[0])
    current = _current_frame(engine, version)
    expected = baseline_content_hash(baseline)
    actual = baseline_content_hash(current)
    if actual != expected:
        raise RuntimeError(
            "Baseline restore check failed: current score content hash "
            f"{actual} != baseline {expected}"
        )
    return actual


def write_ablation_report(
    report: pd.DataFrame,
    details: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    directory = Path(output_dir) if output_dir else BASELINE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "ablation_report.csv"
    json_path = directory / "ablation_report.json"
    report.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {"report": report.to_dict(orient="records"), "details": details},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def _main() -> int:
    report, details = run_ablation()
    csv_path, json_path = write_ablation_report(report, details)
    print(report.to_string(index=False))
    print(f"Ablation report: {csv_path}")
    print(f"Ablation details: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
