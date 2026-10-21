"""Phase 3 raw feature engineering and leakage-safe scaling.

The heavy work is one DuckDB statement. Python only handles the scaler and the
database write; there are no per-address feature loops.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import is_sample_mode, load_data_config, project_path
from chainlens.db import apply_schema, get_pg_engine
from chainlens.fixtures import FIXTURE_LABELS, FIXTURE_SPLITS, SEED, address_for
from chainlens.manifest import record_file

# The locked Phase 3 scaled columns. net_flow_eth is deliberately absent.
SCALER_SPECS: dict[str, str] = {
    "total_value_eth": "log1p_std",
    "tx_count_out": "log1p_std",
    "tx_count_in": "log1p_std",
    "unique_counterparties": "log1p_std",
    "fanout_ratio": "std",
}

# token_transfer_count is checked for variance and logged, but the locked schema
# has no token_transfer_count_scaled column.
VARIANCE_CHECK_ONLY = ("token_transfer_count",)

RAW_COLUMNS = [
    "address",
    "tx_count_out",
    "tx_count_in",
    "tx_count_total",
    "unique_counterparties",
    "fanout_ratio",
    "total_value_eth",
    "avg_value_eth",
    "median_value_eth",
    "net_flow_eth",
    "token_transfer_count",
    "is_contract",
    "first_seen",
    "last_seen",
    "active_days",
    "round_number_ratio",
]


def _optional_path(relative: str) -> Path | None:
    path = project_path(relative)
    return path if path.exists() else None


def _contract_source_sql(parquet_path: str) -> tuple[str, list[str]]:
    contracts_path = _optional_path("data/raw/contracts.parquet")
    if contracts_path:
        return (
            "raw_contracts AS ("
            "SELECT DISTINCT lower(address) AS address FROM read_parquet(?) "
            "WHERE address IS NOT NULL)",
            [str(contracts_path)],
        )
    return (
        "raw_contracts AS (SELECT CAST(NULL AS VARCHAR) AS address WHERE FALSE)",
        [],
    )


def _token_source_sql() -> tuple[str, list[str]]:
    token_path = _optional_path("data/processed/token_transfers.parquet") or _optional_path(
        "data/raw/token_transfers.parquet"
    )
    if token_path:
        return (
            "token_raw AS ("
            "SELECT lower(from_address) AS from_address, lower(to_address) AS to_address "
            "FROM read_parquet(?)), "
            "token_counts AS ("
            "SELECT address, COUNT(*)::BIGINT AS token_transfer_count FROM ("
            "SELECT from_address AS address FROM token_raw "
            "UNION ALL SELECT to_address FROM token_raw"
            ") GROUP BY 1)",
            [str(token_path)],
        )
    return (
        "token_counts AS (SELECT CAST(NULL AS VARCHAR) AS address, "
        "CAST(0 AS BIGINT) AS token_transfer_count WHERE FALSE)",
        [],
    )


def build_raw_features(
    parquet_path: str,
    known_contract_addresses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build one row per address using a single DuckDB statement.

    The address universe is the UNION of senders and receivers, so a pure
    receiver is included. Accounting values remain DECIMAL inside DuckDB; only
    the later log/standardisation step converts ML features to float.
    """
    known = sorted(
        {
            str(address).strip().lower()
            for address in (known_contract_addresses or [])
            if address is not None
        }
    )
    raw_contracts_sql, raw_contract_params = _contract_source_sql(parquet_path)
    token_sql, token_params = _token_source_sql()

    known_cte = (
        "known_contracts AS (SELECT DISTINCT address FROM known_contracts_input)"
        if known
        else "known_contracts AS (SELECT CAST(NULL AS VARCHAR) AS address WHERE FALSE)"
    )

    sql = f"""
    WITH tx AS (
        SELECT
            lower(from_address) AS from_address,
            lower(to_address) AS to_address,
            CAST(value_eth AS DECIMAL(38,18)) AS value_eth,
            block_timestamp,
            lower(receipt_contract_address) AS receipt_contract_address
        FROM read_parquet(?)
        WHERE from_address IS NOT NULL AND to_address IS NOT NULL
    ),
    universe AS (
        SELECT from_address AS address FROM tx
        UNION
        SELECT to_address FROM tx
    ),
    outbound AS (
        SELECT
            from_address AS address,
            COUNT(*)::BIGINT AS tx_count_out,
            COUNT(DISTINCT to_address)::BIGINT AS unique_counterparties,
            SUM(value_eth) AS total_value_eth,
            CAST(AVG(value_eth) AS DECIMAL(38,18)) AS avg_value_eth,
            CAST(MEDIAN(value_eth) AS DECIMAL(38,18)) AS median_value_eth,
            COUNT(*) FILTER (WHERE value_eth = FLOOR(value_eth))::BIGINT
                AS whole_eth_count
        FROM tx
        GROUP BY 1
    ),
    inbound AS (
        SELECT
            to_address AS address,
            COUNT(*)::BIGINT AS tx_count_in,
            SUM(value_eth) AS total_value_in
        FROM tx
        GROUP BY 1
    ),
    activity AS (
        SELECT
            address,
            MIN(block_timestamp) AS first_seen,
            MAX(block_timestamp) AS last_seen,
            COUNT(DISTINCT CAST(block_timestamp AS DATE))::BIGINT AS active_days
        FROM (
            SELECT from_address AS address, block_timestamp FROM tx
            UNION ALL
            SELECT to_address AS address, block_timestamp FROM tx
        )
        GROUP BY 1
    ),
    {raw_contracts_sql},
    {known_cte},
    contract_addresses AS (
        SELECT address FROM raw_contracts
        UNION
        SELECT address FROM known_contracts
        UNION
        SELECT receipt_contract_address
        FROM tx
        WHERE receipt_contract_address IS NOT NULL
    ),
    {token_sql}
    SELECT
        u.address,
        COALESCE(o.tx_count_out, 0)::BIGINT AS tx_count_out,
        COALESCE(i.tx_count_in, 0)::BIGINT AS tx_count_in,
        (COALESCE(o.tx_count_out, 0) + COALESCE(i.tx_count_in, 0))::BIGINT
            AS tx_count_total,
        COALESCE(o.unique_counterparties, 0)::BIGINT AS unique_counterparties,
        -- R2 is about outbound behaviour: distinct outbound counterparties
        -- divided by outbound transaction count (1.0 for FANOUT_WHALE).
        CAST(
            CASE
                WHEN COALESCE(o.tx_count_out, 0) = 0 THEN 0
                ELSE CAST(COALESCE(o.unique_counterparties, 0) AS DOUBLE)
                    / o.tx_count_out
            END
            AS DECIMAL(18,10)
        ) AS fanout_ratio,
        CAST(COALESCE(o.total_value_eth, 0) AS DECIMAL(38,18)) AS total_value_eth,
        CAST(COALESCE(o.avg_value_eth, 0) AS DECIMAL(38,18)) AS avg_value_eth,
        CAST(COALESCE(o.median_value_eth, 0) AS DECIMAL(38,18)) AS median_value_eth,
        CAST(
            COALESCE(i.total_value_in, 0) - COALESCE(o.total_value_eth, 0)
            AS DECIMAL(38,18)
        ) AS net_flow_eth,
        COALESCE(t.token_transfer_count, 0)::BIGINT AS token_transfer_count,
        (u.address IN (SELECT address FROM contract_addresses)) AS is_contract,
        a.first_seen,
        a.last_seen,
        COALESCE(a.active_days, 0)::BIGINT AS active_days,
        CAST(
            CASE
                WHEN COALESCE(o.tx_count_out, 0) = 0 THEN 0
                ELSE CAST(o.whole_eth_count AS DOUBLE) / o.tx_count_out
            END
            AS DECIMAL(18,10)
        ) AS round_number_ratio
    FROM universe AS u
    LEFT JOIN outbound AS o ON o.address = u.address
    LEFT JOIN inbound AS i ON i.address = u.address
    LEFT JOIN activity AS a ON a.address = u.address
    LEFT JOIN token_counts AS t ON t.address = u.address
    ORDER BY u.address
    """

    params: list[object] = [str(parquet_path)]
    params.extend(raw_contract_params)
    params.extend(token_params)

    con = duckdb.connect()
    try:
        if known:
            con.register(
                "known_contracts_input", pd.DataFrame({"address": known})
            )
        frame = con.execute(sql, params).df()
    finally:
        con.close()

    if frame["address"].duplicated().any():
        raise RuntimeError("Feature build produced duplicate addresses")
    if len(frame) != frame["address"].nunique():
        raise RuntimeError("Feature address universe is not unique")
    if "flagged" in frame.columns:
        raise RuntimeError("flagged must never appear in address_features")

    frame["first_seen"] = pd.to_datetime(frame["first_seen"], utc=True)
    frame["last_seen"] = pd.to_datetime(frame["last_seen"], utc=True)
    return frame.sort_values("address").reset_index(drop=True)


def compute_scaler_version(rows: Iterable[dict]) -> str:
    """Content hash of scaler rows, independent of insertion/query order."""
    parts = []
    for row in sorted(rows, key=lambda item: item["feature_name"]):
        mean = Decimal(str(row["mean"]))
        std = Decimal(str(row["std"]))
        parts.append(f"{row['feature_name']}|{mean:.8f}|{std:.8f}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _train_addresses(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT address FROM label_splits WHERE split = 'train'")
        ).scalars()
        return [row for row in rows]


def scaled_columns_from_names(
    feature_names: Iterable[str],
    available_columns: Iterable[str],
    scaler_version: str,
) -> list[str]:
    """Build ``<feature>_scaled`` names and fail loudly on schema drift."""
    available = set(available_columns)
    columns: list[str] = []
    missing: list[str] = []
    for feature_name in sorted(feature_names):
        column = f"{feature_name}_scaled"
        if column not in available:
            missing.append(column)
        else:
            columns.append(column)
    if missing:
        raise RuntimeError(
            "Missing scaled column(s) "
            f"{', '.join(missing)} for scaler_version={scaler_version}. "
            "Re-run `make features` before detection."
        )
    return columns


def get_scaled_feature_columns(engine: Engine) -> tuple[list[str], str]:
    """Read the scaled feature list dynamically from ``feature_scaler``.

    Phase 4 must never hard-code this list: the scaler is the source of truth
    for which features survived variance checks.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT feature_name, std, scaler_version FROM feature_scaler "
                "WHERE std > 0 ORDER BY feature_name"
            )
        ).all()
        if not rows:
            raise RuntimeError("feature_scaler is empty; run `make features` first")
        versions = {row.scaler_version for row in rows}
        if len(versions) != 1:
            raise RuntimeError(
                f"feature_scaler has multiple scaler versions: {sorted(versions)}"
            )
        version = versions.pop()
        available = set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'address_features'"
                )
            ).scalars()
        )
    columns = scaled_columns_from_names(
        [row.feature_name for row in rows], available, version
    )
    return columns, version


def fit_and_apply_scaler(
    df: pd.DataFrame, engine: Engine
) -> tuple[pd.DataFrame, str]:
    """Fit means on train; use train std unless it is zero.

    If train std is zero but the full dataset has variance, use the full-data
    std and mark the row ``train_fallback_full_std``. A feature is dropped only
    when its full-data std is zero. There is deliberately no epsilon floor.
    """
    out = df.copy()
    train = set(_train_addresses(engine))
    if not train:
        raise RuntimeError("No train addresses in label_splits; run Phase 2 first")
    train_mask = out["address"].isin(train)
    train_present = len(train & set(out["address"]))
    if not train_mask.any():
        raise RuntimeError("Train split has no addresses present in address_features")

    dropped: list[dict[str, str]] = []
    scaler_rows: list[dict] = []
    scaled_columns: list[str] = []
    fallbacks: list[str] = []

    for feature_name, transform in SCALER_SPECS.items():
        series = pd.to_numeric(out[feature_name], errors="coerce")
        if series.isna().any():
            raise RuntimeError(f"{feature_name} contains non-numeric values")

        if transform == "log1p_std":
            transformed_train = np.log1p(
                series.loc[train_mask].to_numpy(dtype=float)
            )
            transformed_all = np.log1p(series.to_numpy(dtype=float))
        else:
            transformed_train = series.loc[train_mask].to_numpy(dtype=float)
            transformed_all = series.to_numpy(dtype=float)

        train_mean = float(np.mean(transformed_train))
        train_std = float(np.std(transformed_train, ddof=0))
        full_std = float(np.std(transformed_all, ddof=0))

        if not np.isfinite(full_std) or full_std == 0.0:
            dropped.append({"feature_name": feature_name, "reason": "zero_variance"})
            out[f"{feature_name}_scaled"] = None
            continue

        if not np.isfinite(train_std) or train_std == 0.0:
            std = full_std
            fitted_on_split = "train_fallback_full_std"
            fallbacks.append(feature_name)
        else:
            std = train_std
            fitted_on_split = "train"

        out[f"{feature_name}_scaled"] = (transformed_all - train_mean) / std
        scaled_columns.append(f"{feature_name}_scaled")
        scaler_rows.append(
            {
                "feature_name": feature_name,
                "transform": transform,
                "mean": train_mean,
                "std": std,
                "fitted_on_split": fitted_on_split,
                "row_count": train_present,
            }
        )

    for feature_name in VARIANCE_CHECK_ONLY:
        series = pd.to_numeric(out[feature_name], errors="coerce")
        if series.isna().any():
            raise RuntimeError(f"{feature_name} contains non-numeric values")
        full_std = float(np.std(series.to_numpy(dtype=float), ddof=0))
        if not np.isfinite(full_std) or full_std == 0.0:
            dropped.append({"feature_name": feature_name, "reason": "zero_variance"})

    version = compute_scaler_version(scaler_rows)
    for row in scaler_rows:
        row["scaler_version"] = version
    out["scaler_version"] = version

    print(f"Scaler version: {version} (fit on {train_present} train addresses)")
    print("Scaled columns: " + ", ".join(sorted(scaled_columns)))
    if fallbacks:
        print("Full-std fallback: " + ", ".join(sorted(fallbacks)))
    if dropped:
        print(
            "Dropped features: "
            + ", ".join(f"{item['feature_name']} ({item['reason']})" for item in dropped)
        )
    else:
        print("Dropped features: none")

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM feature_scaler"))
        conn.execute(text("DELETE FROM feature_scaler_dropped"))
        if scaler_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO feature_scaler (
                        feature_name, transform, mean, std,
                        fitted_on_split, row_count, scaler_version
                    ) VALUES (
                        :feature_name, :transform, :mean, :std,
                        :fitted_on_split, :row_count, :scaler_version
                    )
                    """
                ),
                scaler_rows,
            )
        if dropped:
            conn.execute(
                text(
                    "INSERT INTO feature_scaler_dropped (feature_name, reason) "
                    "VALUES (:feature_name, :reason)"
                ),
                dropped,
            )

    return out, version


def write_address_features(df: pd.DataFrame, engine: Engine) -> int:
    """Deterministic DELETE + INSERT (never append to an existing run)."""
    if "flagged" in df.columns:
        raise RuntimeError("flagged must never be written to address_features")

    ordered = df.sort_values("address").reset_index(drop=True)
    ordered["address"] = ordered["address"].str.lower()
    if ordered["address"].duplicated().any():
        raise RuntimeError("Duplicate address before address_features write")

    columns = [column for column in ordered.columns if column in RAW_COLUMNS or column.endswith("_scaled") or column == "scaler_version"]
    ordered = ordered[columns]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM address_features"))
        ordered.to_sql(
            "address_features",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
    return len(ordered)


def ensure_fixture_labels(engine: Engine) -> int:
    """Restore the three fixture-only labels/splits in sample mode only.

    Real mode is a strict no-op: it never touches ``labels`` or
    ``label_splits``. Placements are fixed and asserted because the fixture
    regression gate depends on SANCTIONED_SEED=train and KAGGLE_FRAUD=test.
    """
    if not is_sample_mode():
        return 0

    label_rows = [
        {
            "address": address_for(name),
            "label": label,
            "source": source,
            "confidence": confidence,
            "as_of": date.today(),
        }
        for name, (label, source, confidence) in FIXTURE_LABELS.items()
    ]
    split_rows = [
        {"address": address_for(name), "split": split, "seed": SEED}
        for name, split in FIXTURE_SPLITS.items()
    ]

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO labels (address, label, source, confidence, as_of)
                VALUES (:address, :label, :source, :confidence, :as_of)
                ON CONFLICT (address, source, label) DO NOTHING
                """
            ),
            label_rows,
        )
        conn.execute(
            text(
                """
                INSERT INTO label_splits (address, split, seed)
                VALUES (:address, :split, :seed)
                ON CONFLICT (address) DO NOTHING
                """
            ),
            split_rows,
        )

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM labels WHERE source = 'fixture'")
        ).scalar_one()
        placement = dict(
            conn.execute(
                text(
                    "SELECT address, split FROM label_splits WHERE address = ANY(:addresses)"
                ),
                {"addresses": [address_for(name) for name in FIXTURE_SPLITS]},
            ).all()
        )

    if count != len(label_rows):
        raise RuntimeError(f"Expected 3 fixture labels, found {count}")
    for name, expected in FIXTURE_SPLITS.items():
        actual = placement.get(address_for(name))
        if actual != expected:
            raise RuntimeError(
                f"Fixture split gate failed for {name}: expected {expected}, got {actual}"
            )
    print(f"Fixture backfill: {count} labels, placements verified")
    return count


def ensure_contract_coverage(engine: Engine) -> dict:
    """Compare contract sources with known_addresses and backfill any gap."""
    config = load_data_config()
    clean_path = project_path(config["ingest"]["clean_parquet"])
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}; run Phase 2 ingest first")

    con = duckdb.connect()
    try:
        contracts_path = _optional_path("data/raw/contracts.parquet")
        if contracts_path:
            contracts = {
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT lower(address) FROM read_parquet(?) "
                    "WHERE address IS NOT NULL",
                    [str(contracts_path)],
                ).fetchall()
            }
            source = "data/raw/contracts.parquet"
        else:
            contracts = {
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT lower(receipt_contract_address) "
                    "FROM read_parquet(?) WHERE receipt_contract_address IS NOT NULL",
                    [str(clean_path)],
                ).fetchall()
            }
            source = "transactions_clean.receipt_contract_address"
    finally:
        con.close()

    with engine.connect() as conn:
        known = set(
            conn.execute(
                text("SELECT address FROM known_addresses WHERE category = 'contract'")
            ).scalars()
        )
    missing = sorted(contracts - known)
    if missing:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO known_addresses (address, category, note) "
                    "VALUES (:address, 'contract', :note) "
                    "ON CONFLICT (address) DO NOTHING"
                ),
                [
                    {
                        "address": address,
                        "note": "Phase 3 contract-coverage backfill",
                    }
                    for address in missing
                ],
            )
        print(
            f"Contract coverage: FIXED {len(missing)} missing contract(s) "
            f"from {source}"
        )
    else:
        print(
            f"Contract coverage verified: {len(contracts)} distinct contract(s) "
            f"from {source}, {len(known)} known"
        )
    return {
        "source": source,
        "contracts": len(contracts),
        "known_before": len(known),
        "missing_backfilled": len(missing),
        "verified": not missing,
    }


def expected_evaluable_positives(engine: Engine) -> dict:
    with engine.connect() as conn:
        total = conn.execute(
            text(
                """
                SELECT count(DISTINCT af.address)
                FROM address_features af
                JOIN label_splits ls ON ls.address = af.address
                WHERE ls.split = 'test' AND af.tx_count_total >= 1
                """
            )
        ).scalar_one()
        by_source = conn.execute(
            text(
                """
                SELECT l.source, count(DISTINCT af.address)
                FROM address_features af
                JOIN label_splits ls ON ls.address = af.address
                JOIN labels l ON l.address = af.address
                WHERE ls.split = 'test' AND af.tx_count_total >= 1
                GROUP BY l.source
                ORDER BY l.source
                """
            )
        ).all()
    return {"total": int(total), "by_source": {source: int(count) for source, count in by_source}}


def run_features() -> dict:
    config = load_data_config()
    engine = get_pg_engine()
    apply_schema(engine)

    if is_sample_mode():
        fixture_count = ensure_fixture_labels(engine)
    else:
        with engine.connect() as conn:
            fixture_count = conn.execute(
                text("SELECT count(*) FROM labels WHERE source = 'fixture'")
            ).scalar_one()
        if fixture_count:
            raise RuntimeError(
                "Fixture labels are present in real mode. Re-run Phase 2 ingest "
                "to restore a clean labels/label_splits state before Phase 3."
            )

    coverage = ensure_contract_coverage(engine)
    known_contracts = []
    with engine.connect() as conn:
        known_contracts = list(
            conn.execute(
                text("SELECT address FROM known_addresses WHERE category = 'contract'")
            ).scalars()
        )

    clean_path = project_path(config["ingest"]["clean_parquet"])
    raw = build_raw_features(str(clean_path), known_contracts)
    scaled, version = fit_and_apply_scaler(raw, engine)
    rows = write_address_features(scaled, engine)

    with engine.connect() as conn:
        total, distinct = conn.execute(
            text("SELECT count(*), count(DISTINCT address) FROM address_features")
        ).one()
    if total != distinct or int(total) != rows:
        raise RuntimeError(
            f"address_features verification failed: rows={total}, distinct={distinct}"
        )

    evaluable = expected_evaluable_positives(engine)
    with engine.connect() as conn:
        scaler_rows = [
            {
                "feature_name": row.feature_name,
                "transform": row.transform,
                "mean": str(row.mean),
                "std": str(row.std),
            }
            for row in conn.execute(
                text(
                    "SELECT feature_name, transform, mean, std "
                    "FROM feature_scaler ORDER BY feature_name"
                )
            )
        ]
        dropped = [
            {"feature_name": name, "reason": reason}
            for name, reason in conn.execute(
                text(
                    "SELECT feature_name, reason FROM feature_scaler_dropped "
                    "ORDER BY feature_name"
                )
            )
        ]

    summary = {
        "row_count": int(rows),
        "distinct_addresses": int(distinct),
        "scaler_version": version,
        "scaler_features": scaler_rows,
        "dropped_features": dropped,
        "fixture_labels": int(fixture_count),
        "contract_coverage": coverage,
        "expected_evaluable_positives": evaluable,
        "sample_mode": is_sample_mode(),
    }
    summary_path = project_path(config["ingest"]["features_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    record_file(
        engine,
        summary_path,
        file_path="data/processed/features_summary.json",
        row_count=int(rows),
        metadata={
            "stage": "features",
            "scaler_version": version,
            "dropped_features": [item["feature_name"] for item in dropped],
            "expected_evaluable_positives": evaluable["total"],
            "sample_mode": is_sample_mode(),
        },
    )

    print(f"address_features: {total:,} rows, {distinct:,} distinct addresses")
    print(
        "Expected evaluable positives (test-split addresses present in "
        f"address_features): {evaluable['total']:,}"
    )
    return summary


def print_check() -> int:
    engine = get_pg_engine()
    with engine.connect() as conn:
        total, distinct = conn.execute(
            text("SELECT count(*), count(DISTINCT address) FROM address_features")
        ).one()
        if total == 0:
            print("address_features is empty; run `make features` first")
            return 1
        version = conn.execute(
            text("SELECT DISTINCT scaler_version FROM address_features")
        ).scalar_one()
        scaler_count = conn.execute(
            text("SELECT count(*) FROM feature_scaler")
        ).scalar_one()
        dropped = [
            f"{name} ({reason})"
            for name, reason in conn.execute(
                text(
                    "SELECT feature_name, reason FROM feature_scaler_dropped "
                    "ORDER BY feature_name"
                )
            )
        ]
        test_present = conn.execute(
            text(
                """
                SELECT count(*)
                FROM address_features af
                JOIN label_splits ls ON ls.address = af.address
                WHERE ls.split = 'test'
                """
            )
        ).scalar_one()
    evaluable = expected_evaluable_positives(engine)
    print(f"address_features rows: {total:,} (distinct {distinct:,})")
    print(f"scaler_version: {version}")
    print(f"feature_scaler rows: {scaler_count}")
    print("dropped features: " + (", ".join(dropped) if dropped else "none"))
    print(f"test-split addresses present: {test_present:,}")
    print(f"expected evaluable positives: {evaluable['total']:,}")
    print(f"  by source: {evaluable['by_source']}")
    return 0 if total == distinct else 1
