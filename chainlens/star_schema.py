"""Phase 6 star-schema population (one-off reporting artifact).

Nothing in ``chainlens/`` imports these tables as a data-access dependency. The
module contains SQL strings because the star schema is a presentation layer,
not the pipeline's working schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.config import load_data_config, project_path
from chainlens.db import apply_schema, get_pg_engine
from chainlens.manifest import record_manifest, sha256_file

STAR_SCHEMA_PATH = project_path("sql/star_schema.sql")
FACT_TABLE = "fact_transactions"


def apply_star_schema(engine: Engine) -> None:
    apply_schema(engine, str(STAR_SCHEMA_PATH))


def _clean_parquet_path() -> Path:
    return project_path(load_data_config()["ingest"]["clean_parquet"])


def _fact_frame(parquet_path: Path) -> pd.DataFrame:
    with duckdb.connect() as conn:
        return conn.execute(
            """
            SELECT
                tx_hash,
                block_number,
                block_timestamp,
                from_address,
                to_address,
                value_eth,
                gas,
                gas_price,
                is_contract AS is_contract_to,
                receipt_status
            FROM read_parquet(?)
            ORDER BY block_number, tx_hash
            """,
            [str(parquet_path)],
        ).df()


def populate_star_schema(
    engine: Engine | None = None,
    parquet_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply and populate the star schema; idempotent on re-run."""
    engine = engine or get_pg_engine()
    apply_star_schema(engine)
    path = Path(parquet_path) if parquet_path else _clean_parquet_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing clean parquet: {path}; run `make ingest`")

    fact = _fact_frame(path)
    rows = []
    for row in fact.itertuples(index=False):
        rows.append(
            {
                "tx_hash": row.tx_hash,
                "block_number": int(row.block_number),
                "block_timestamp": row.block_timestamp,
                "from_address": str(row.from_address).lower(),
                "to_address": str(row.to_address).lower(),
                "value_eth": row.value_eth,
                "gas": None if pd.isna(row.gas) else int(row.gas),
                "gas_price": None if pd.isna(row.gas_price) else int(row.gas_price),
                "is_contract_to": (
                    None if pd.isna(row.is_contract_to) else bool(row.is_contract_to)
                ),
                "receipt_status": (
                    None if pd.isna(row.receipt_status) else int(row.receipt_status)
                ),
            }
        )

    statement = text(
        """
        INSERT INTO fact_transactions (
            tx_hash, block_number, block_timestamp, from_address, to_address,
            value_eth, gas, gas_price, is_contract_to, receipt_status
        ) VALUES (
            :tx_hash, :block_number, :block_timestamp, :from_address, :to_address,
            :value_eth, :gas, :gas_price, :is_contract_to, :receipt_status
        )
        ON CONFLICT (tx_hash) DO UPDATE SET
            block_number = EXCLUDED.block_number,
            block_timestamp = EXCLUDED.block_timestamp,
            from_address = EXCLUDED.from_address,
            to_address = EXCLUDED.to_address,
            value_eth = EXCLUDED.value_eth,
            gas = EXCLUDED.gas,
            gas_price = EXCLUDED.gas_price,
            is_contract_to = EXCLUDED.is_contract_to,
            receipt_status = EXCLUDED.receipt_status
        """
    )
    with engine.begin() as conn:
        if rows:
            conn.execute(statement, rows)
        conn.execute(
            text(
                """
                INSERT INTO dim_address (
                    address, is_contract, is_known, known_category,
                    first_seen, last_seen
                )
                SELECT
                    af.address,
                    af.is_contract,
                    (ka.address IS NOT NULL) AS is_known,
                    ka.category AS known_category,
                    af.first_seen,
                    af.last_seen
                FROM address_features af
                LEFT JOIN known_addresses ka ON ka.address = af.address
                ON CONFLICT (address) DO UPDATE SET
                    is_contract = EXCLUDED.is_contract,
                    is_known = EXCLUDED.is_known,
                    known_category = EXCLUDED.known_category,
                    first_seen = EXCLUDED.first_seen,
                    last_seen = EXCLUDED.last_seen
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO dim_label (address, label, source, confidence, as_of)
                SELECT address, label, source, confidence, as_of
                FROM labels
                ON CONFLICT (address, source, label) DO NOTHING
                """
            )
        )
        fact_count = int(
            conn.execute(text("SELECT count(*) FROM fact_transactions")).scalar_one()
        )
        dim_address_count = int(
            conn.execute(text("SELECT count(*) FROM dim_address")).scalar_one()
        )
        dim_label_count = int(
            conn.execute(text("SELECT count(*) FROM dim_label")).scalar_one()
        )

    source_sha256 = sha256_file(path)
    record_manifest(
        engine,
        "star_schema:fact_transactions",
        source_sha256,
        row_count=fact_count,
        metadata={
            "source_parquet": str(path),
            "schema": str(STAR_SCHEMA_PATH),
            "dim_address_rows": dim_address_count,
            "dim_label_rows": dim_label_count,
            "fact_rows": fact_count,
            "source_sha256": source_sha256,
        },
    )
    return {
        "fact_transactions": fact_count,
        "dim_address": dim_address_count,
        "dim_label": dim_label_count,
        "source_rows": int(len(fact)),
        "parquet_path": str(path),
        "schema_path": str(STAR_SCHEMA_PATH),
    }


def _main() -> int:
    result = populate_star_schema()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
