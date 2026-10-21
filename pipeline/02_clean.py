#!/usr/bin/env python3
"""Canonicalise raw transactions into data/processed/transactions_clean.parquet.

All wei arithmetic happens in DuckDB with HUGEINT/DECIMAL(38,18); pandas never
sees an int64/float64 amount.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.config import is_sample_mode, load_data_config, project_path  # noqa: E402
from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.manifest import record_file  # noqa: E402

WEI_PER_ETH = 10**18


def resolve_transactions(config: dict) -> Path:
    raw = project_path(config["ingest"]["raw_transactions"])
    if raw.exists():
        return raw
    if is_sample_mode():
        sample = project_path(config["paths"]["sample_parquet"])
        if not sample.exists():
            raise FileNotFoundError(f"Missing sample Parquet: {sample}")
        print(f"Sample mode: reading {sample} instead of {raw}")
        return sample
    raise FileNotFoundError(
        f"Missing {raw}. Run the BigQuery export first, or set "
        "CHAINLENS_SAMPLE_MODE=1 for the synthetic fixture path."
    )


def value_eth_sql(value_column: str = "value_wei") -> str:
    """Exact wei -> ETH conversion using HUGEINT integer division, no DOUBLE."""
    huge = f"CAST({value_column} AS HUGEINT)"
    return (
        "CAST("
        f"({huge} // {WEI_PER_ETH})::VARCHAR || '.' || "
        f"lpad(({huge} % {WEI_PER_ETH})::VARCHAR, 18, '0') "
        "AS DECIMAL(38,18))"
    )


def run_clean() -> Path:
    config = load_data_config()
    source = resolve_transactions(config)
    output = project_path(config["ingest"]["clean_parquet"])
    output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
            ).fetchall()
        }
        has_tx_index = "transaction_index" in columns
        tx_index_sql = (
            "CAST(transaction_index AS INTEGER) AS transaction_index"
            if has_tx_index
            else "CAST(NULL AS INTEGER) AS transaction_index"
        )

        contracts_path = project_path(config["ingest"]["raw_contracts"])
        if contracts_path.exists():
            con.execute(
                "CREATE TEMP TABLE contract_addresses AS "
                "SELECT DISTINCT lower(address) AS address FROM read_parquet(?) "
                "WHERE address IS NOT NULL",
                [str(contracts_path)],
            )
            contract_source = str(contracts_path.relative_to(ROOT))
        else:
            con.execute(
                """
                CREATE TEMP TABLE contract_addresses AS
                SELECT DISTINCT lower(receipt_contract_address) AS address
                FROM read_parquet(?)
                WHERE receipt_contract_address IS NOT NULL
                """,
                [str(source)],
            )
            contract_source = "receipt_contract_address (derived)"

        raw_count = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(source)]
        ).fetchone()[0]
        contract_count = con.execute(
            "SELECT count(*) FROM contract_addresses"
        ).fetchone()[0]

        query = f"""
        COPY (
            SELECT
                tx_hash,
                CAST(block_number AS BIGINT) AS block_number,
                CAST(block_timestamp AS TIMESTAMPTZ) AS block_timestamp,
                lower(from_address) AS from_address,
                lower(to_address) AS to_address,
                CAST(CAST(value_wei AS DECIMAL(38,0)) AS VARCHAR) AS value_wei,
                {value_eth_sql()} AS value_eth,
                CAST(gas AS BIGINT) AS gas,
                CAST(gas_price AS BIGINT) AS gas_price,
                CAST(receipt_status AS INTEGER) AS receipt_status,
                lower(receipt_contract_address) AS receipt_contract_address,
                {tx_index_sql},
                (lower(to_address) IN (SELECT address FROM contract_addresses)) AS is_contract
            FROM read_parquet(?)
            WHERE CAST(receipt_status AS INTEGER) = 1
              AND lower(from_address) <> lower(to_address)
              AND regexp_full_match(lower(from_address), '0x[0-9a-f]{{40}}')
              AND regexp_full_match(lower(to_address), '0x[0-9a-f]{{40}}')
        ) TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query, [str(source)])

        clean_count = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(output)]
        ).fetchone()[0]
        bounds = con.execute(
            "SELECT min(block_number), max(block_number) FROM read_parquet(?)",
            [str(output)],
        ).fetchone()
        failed = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE receipt_status <> 1",
            [str(output)],
        ).fetchone()[0]
        null_contracts = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE is_contract IS NULL",
            [str(output)],
        ).fetchone()[0]
        max_row = con.execute(
            """
            SELECT value_wei, value_eth FROM read_parquet(?)
            ORDER BY CAST(value_wei AS HUGEINT) DESC LIMIT 1
            """,
            [str(output)],
        ).fetchone()
    finally:
        con.close()

    if clean_count > raw_count:
        raise RuntimeError(f"Clean row count {clean_count} exceeds raw {raw_count}")
    if failed:
        raise RuntimeError(f"{failed} failed transactions survived the clean stage")
    if null_contracts:
        raise RuntimeError("is_contract contains NULLs")
    if clean_count:
        expected = Decimal(max_row[0]) / Decimal(WEI_PER_ETH)
        actual = Decimal(max_row[1])
        if actual != expected.quantize(Decimal("0.000000000000000001")):
            raise RuntimeError(f"value_eth precision mismatch: {actual} != {expected}")

    engine = get_pg_engine()
    apply_schema(engine)
    record_file(
        engine,
        output,
        file_path="data/processed/transactions_clean.parquet",
        row_count=clean_count,
        block_start=int(bounds[0]) if bounds[0] is not None else None,
        block_end=int(bounds[1]) if bounds[1] is not None else None,
        metadata={
            "stage": "clean",
            "source": str(source.relative_to(ROOT)),
            "sample_mode": is_sample_mode(),
            "raw_rows": raw_count,
            "dropped_rows": raw_count - clean_count,
            "contract_addresses": contract_count,
            "contract_source": contract_source,
            "value_eth_conversion": "duckdb HUGEINT // and % -> DECIMAL(38,18)",
        },
    )

    print(
        f"Cleaned {clean_count:,}/{raw_count:,} transactions -> "
        f"{output.relative_to(ROOT)}"
    )
    print(
        f"Blocks {bounds[0]}-{bounds[1]}; contracts={contract_count:,}; "
        f"failed={failed}; null is_contract={null_contracts}"
    )
    return output


if __name__ == "__main__":
    run_clean()
