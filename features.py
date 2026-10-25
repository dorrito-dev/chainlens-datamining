"""Stage 2: cleaning, feature engineering, train-only scaling, warehouse.

All feature accounting stays in DuckDB DECIMAL; wei is never parsed through
pandas int64/float64. The warehouse is one DuckDB file so the dashboard and the
audit trail read the same tables.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

SCALER_SPECS = {
    "total_value_eth": "log1p_std",
    "tx_count_out": "log1p_std",
    "tx_count_in": "log1p_std",
    "unique_counterparties": "log1p_std",
    "fanout_ratio": "std",
}

WEI = 1_000_000_000_000_000_000


def connect(warehouse: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    if warehouse:
        Path(warehouse).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(warehouse))
    return duckdb.connect()


def _value_eth_sql(column: str = "value_wei") -> str:
    wei = f"COALESCE(TRY_CAST({column} AS HUGEINT), 0)"
    return (
        f"CAST(({wei}) // {WEI} AS DECIMAL(38,18)) + "
        f"CAST(({wei}) % {WEI} AS DECIMAL(38,18)) / {WEI}"
    )


def clean_transactions(con: duckdb.DuckDBPyConnection, raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise, de-duplicate, and filter raw transactions.

    The same transaction appears in both the sender's and receiver's cache, so
    `tx_hash` de-duplication is mandatory.
    """
    if raw.empty:
        con.execute("CREATE OR REPLACE TABLE transactions (tx_hash VARCHAR)")
        return pd.DataFrame()

    frame = raw.copy()
    if "value_wei" not in frame.columns and "value_eth" in frame.columns:
        frame["value_wei"] = (
            pd.to_numeric(frame["value_eth"], errors="coerce").fillna(0) * WEI
        ).astype("Int64").astype(str)
    if "is_error" not in frame.columns:
        frame["is_error"] = 0
    if "receipt_status" not in frame.columns:
        frame["receipt_status"] = 1
    if "receipt_contract_address" not in frame.columns:
        frame["receipt_contract_address"] = None
    if "source" not in frame.columns:
        frame["source"] = "unknown"
    if "fetched_at" not in frame.columns:
        frame["fetched_at"] = None

    con.register("raw_tx", frame)
    con.execute(f"""
        CREATE OR REPLACE TABLE transactions AS
        WITH src AS (
            SELECT
                lower(trim(tx_hash)) AS tx_hash,
                TRY_CAST(block_number AS BIGINT) AS block_number,
                TRY_CAST(block_timestamp AS TIMESTAMPTZ) AS block_timestamp,
                lower(trim(from_address)) AS from_address,
                lower(trim(to_address)) AS to_address,
                CAST(value_wei AS VARCHAR) AS value_wei,
                {_value_eth_sql()} AS value_eth,
                TRY_CAST(gas AS BIGINT) AS gas,
                TRY_CAST(gas_price AS BIGINT) AS gas_price,
                COALESCE(TRY_CAST(receipt_status AS INTEGER), 1) AS receipt_status,
                lower(trim(receipt_contract_address)) AS receipt_contract_address,
                COALESCE(TRY_CAST(is_error AS INTEGER), 0) AS is_error,
                CAST(source AS VARCHAR) AS source,
                TRY_CAST(fetched_at AS TIMESTAMPTZ) AS fetched_at
            FROM raw_tx
        )
        SELECT * FROM src
        WHERE receipt_status = 1
          AND is_error = 0
          AND from_address IS NOT NULL
          AND to_address IS NOT NULL
          AND from_address <> to_address
          AND regexp_matches(from_address, '^0x[0-9a-f]{{40}}$')
          AND regexp_matches(to_address, '^0x[0-9a-f]{{40}}$')
          AND block_timestamp IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY tx_hash ORDER BY block_number) = 1
    """)
    con.unregister("raw_tx")
    return con.execute("SELECT * FROM transactions ORDER BY block_number, tx_hash").df()


def compute_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One DuckDB statement, one row per address in the sender/receiver universe."""
    con.execute("""
        CREATE OR REPLACE TABLE wallet_features AS
        WITH tx AS (
            SELECT from_address, to_address, value_eth, block_timestamp,
                   receipt_contract_address
            FROM transactions
        ),
        universe AS (
            SELECT from_address AS address FROM tx
            UNION
            SELECT to_address FROM tx
        ),
        outbound AS (
            SELECT from_address AS address,
                   COUNT(*)::BIGINT AS tx_count_out,
                   COUNT(DISTINCT to_address)::BIGINT AS unique_counterparties,
                   SUM(value_eth) AS total_value_eth,
                   CAST(AVG(value_eth) AS DECIMAL(38,18)) AS avg_value_eth,
                   CAST(MEDIAN(value_eth) AS DECIMAL(38,18)) AS median_value_eth,
                   COUNT(*) FILTER (WHERE value_eth = FLOOR(value_eth))::BIGINT AS whole_eth_count
            FROM tx GROUP BY 1
        ),
        inbound AS (
            SELECT to_address AS address,
                   COUNT(*)::BIGINT AS tx_count_in,
                   COUNT(DISTINCT from_address)::BIGINT AS unique_senders,
                   SUM(value_eth) AS total_value_in
            FROM tx GROUP BY 1
        ),
        activity AS (
            SELECT address, MIN(block_timestamp) AS first_seen,
                   MAX(block_timestamp) AS last_seen,
                   COUNT(DISTINCT CAST(block_timestamp AS DATE))::BIGINT AS active_days
            FROM (
                SELECT from_address AS address, block_timestamp FROM tx
                UNION ALL
                SELECT to_address, block_timestamp FROM tx
            ) GROUP BY 1
        ),
        contracts AS (
            SELECT DISTINCT receipt_contract_address AS address FROM tx
            WHERE receipt_contract_address IS NOT NULL
        )
        SELECT
            u.address,
            COALESCE(o.tx_count_out, 0)::BIGINT AS tx_count_out,
            COALESCE(i.tx_count_in, 0)::BIGINT AS tx_count_in,
            (COALESCE(o.tx_count_out, 0) + COALESCE(i.tx_count_in, 0))::BIGINT AS tx_count_total,
            COALESCE(o.unique_counterparties, 0)::BIGINT AS unique_counterparties,
            COALESCE(i.unique_senders, 0)::BIGINT AS unique_senders,
            CAST(CASE WHEN COALESCE(o.tx_count_out, 0) = 0 THEN 0
                 ELSE CAST(COALESCE(o.unique_counterparties, 0) AS DOUBLE) / o.tx_count_out
                 END AS DECIMAL(18,10)) AS fanout_ratio,
            CAST(CASE WHEN COALESCE(i.tx_count_in, 0) = 0 THEN NULL
                 ELSE CAST(COALESCE(o.tx_count_out, 0) AS DOUBLE) / i.tx_count_in
                 END AS DECIMAL(18,10)) AS out_in_ratio,
            CAST(COALESCE(o.total_value_eth, 0) AS DECIMAL(38,18)) AS total_value_eth,
            CAST(COALESCE(o.avg_value_eth, 0) AS DECIMAL(38,18)) AS avg_value_eth,
            CAST(COALESCE(o.median_value_eth, 0) AS DECIMAL(38,18)) AS median_value_eth,
            CAST(COALESCE(i.total_value_in, 0) - COALESCE(o.total_value_eth, 0)
                 AS DECIMAL(38,18)) AS net_flow_eth,
            (u.address IN (SELECT address FROM contracts)) AS is_contract,
            a.first_seen,
            a.last_seen,
            COALESCE(a.active_days, 0)::BIGINT AS active_days,
            CAST(CASE WHEN COALESCE(o.tx_count_out, 0) = 0 THEN 0
                 ELSE CAST(o.whole_eth_count AS DOUBLE) / o.tx_count_out
                 END AS DECIMAL(18,10)) AS round_number_ratio
        FROM universe u
        LEFT JOIN outbound o ON o.address = u.address
        LEFT JOIN inbound i ON i.address = u.address
        LEFT JOIN activity a ON a.address = u.address
        ORDER BY u.address
    """)
    frame = con.execute("SELECT * FROM wallet_features").df()
    frame["first_seen"] = pd.to_datetime(frame["first_seen"], utc=True)
    frame["last_seen"] = pd.to_datetime(frame["last_seen"], utc=True)
    return frame.sort_values("address").reset_index(drop=True)


def scale_features(
    features: pd.DataFrame,
    train_addresses: set[str],
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    """Fit mean/std on train addresses only, then transform every address."""
    out = features.copy()
    train_mask = out["address"].isin(train_addresses)
    if not train_mask.any():
        train_mask = pd.Series(True, index=out.index)

    scaler_rows: list[dict[str, Any]] = []
    for feat, transform in SCALER_SPECS.items():
        if feat not in out.columns:
            continue
        series = pd.to_numeric(out[feat], errors="coerce").astype(float)
        if transform == "log1p_std":
            t_train = np.log1p(series.loc[train_mask].to_numpy(dtype=float))
            t_all = np.log1p(series.to_numpy(dtype=float))
        else:
            t_train = series.loc[train_mask].to_numpy(dtype=float)
            t_all = series.to_numpy(dtype=float)

        train_mean = float(np.nanmean(t_train)) if len(t_train) else 0.0
        train_std = float(np.nanstd(t_train, ddof=0)) if len(t_train) else 0.0
        full_std = float(np.nanstd(t_all, ddof=0))

        if full_std == 0.0 or not np.isfinite(full_std):
            out[f"{feat}_scaled"] = 0.0
            continue

        std = train_std if train_std > 0 else full_std
        fitted_on = "train" if train_std > 0 else "train_fallback_full_std"
        out[f"{feat}_scaled"] = (t_all - train_mean) / std
        scaler_rows.append({
            "feature_name": feat,
            "transform": transform,
            "mean": train_mean,
            "std": std,
            "fitted_on_split": fitted_on,
            "row_count": int(train_mask.sum()),
        })

    parts = [f"{r['feature_name']}|{r['mean']:.8f}|{r['std']:.8f}"
             for r in sorted(scaler_rows, key=lambda x: x["feature_name"])]
    version = hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]
    out["scaler_version"] = version
    return out, version, scaler_rows


def write_feature_tables(
    con: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    scaler_rows: list[dict[str, Any]],
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    known: pd.DataFrame,
) -> None:
    con.register("features_df", features)
    con.execute("CREATE OR REPLACE TABLE wallet_features AS SELECT * FROM features_df")
    con.unregister("features_df")

    scaler_columns = ["feature_name", "transform", "mean", "std", "fitted_on_split", "row_count"]
    scaler_frame = pd.DataFrame(scaler_rows, columns=scaler_columns) if scaler_rows else pd.DataFrame(columns=scaler_columns)
    con.register("scaler_df", scaler_frame)
    con.execute("CREATE OR REPLACE TABLE feature_scaler AS SELECT * FROM scaler_df")
    con.unregister("scaler_df")

    labels_frame = labels if not labels.empty else pd.DataFrame(
        columns=["address", "label", "source", "confidence", "as_of"]
    )
    con.register("labels_df", labels_frame)
    con.execute("CREATE OR REPLACE TABLE labels AS SELECT * FROM labels_df")
    con.unregister("labels_df")

    splits_frame = splits if not splits.empty else pd.DataFrame(columns=["address", "split", "seed"])
    con.register("splits_df", splits_frame)
    con.execute("CREATE OR REPLACE TABLE splits AS SELECT * FROM splits_df")
    con.unregister("splits_df")

    known_frame = known if not known.empty else pd.DataFrame(columns=["address", "category", "note"])
    con.register("known_df", known_frame)
    con.execute("CREATE OR REPLACE TABLE known_addresses AS SELECT * FROM known_df")
    con.unregister("known_df")


def write_table(con: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    con.register("_tmp_frame", frame)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_frame")
    con.unregister("_tmp_frame")


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return p


def sample_schema_guard(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept both committed sample (`value_wei`) and processed (`value_eth`)."""
    frame = raw.copy()
    if "value_wei" not in frame.columns:
        if "value_eth" not in frame.columns:
            raise ValueError("transactions need value_wei or value_eth")
        frame["value_wei"] = (
            pd.to_numeric(frame["value_eth"], errors="coerce").fillna(0) * WEI
        ).astype("Int64").astype(str)
    return frame
