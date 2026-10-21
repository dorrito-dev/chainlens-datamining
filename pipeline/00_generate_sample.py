#!/usr/bin/env python3
"""Generate ChainLens' deterministic synthetic sample and seed PostgreSQL.

The parquet file is written to ``data/sample/transactions.parquet``. The same
run also applies ``sql/schema.sql`` and replaces the Phase 1 rows in ``labels``,
``label_splits`` and ``known_addresses`` so ``make sample`` is idempotent.

Run from the repository root::

    python pipeline/00_generate_sample.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.fixtures import SEED, address_for  # noqa: E402

BLOCK_START = 19_000_000
BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
GAS_PRICE = 20_000_000_000
PARQUET_PATH = ROOT / "data" / "sample" / "transactions.parquet"


def wei(eth: str | float | Decimal) -> str:
    return str(int(Decimal(str(eth)) * Decimal(10**18)))


class TxBuilder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(
        self,
        from_name: str,
        to_name: str,
        eth: str | float,
        offset_seconds: int,
        *,
        gas: int = 21_000,
        contract_address: str | None = None,
    ) -> None:
        self.rows.append(
            {
                "_offset": int(offset_seconds),
                "_seq": len(self.rows),
                "from_address": address_for(from_name),
                "to_address": address_for(to_name),
                "value_wei": wei(eth),
                "gas": gas,
                "gas_price": GAS_PRICE,
                "receipt_status": 1,
                "receipt_contract_address": (
                    address_for(contract_address) if contract_address else None
                ),
            }
        )


def build_fixture_rows(builder: TxBuilder, rng: np.random.Generator) -> None:
    # R2: one whale sends to 40+ unique addresses.
    for i in range(40):
        builder.add(
            "FANOUT_WHALE",
            f"FANOUT_RECIPIENT_{i:02d}",
            f"0.{5 + (i % 5):02d}",
            i * 300,
        )
    for i in range(5):
        builder.add(f"AUX_FANIN_{i}", "FANOUT_WHALE", "0.5", 18_000 + i * 60)

    # R1: a 3-hop cycle with near-identical values inside 24 hours.
    builder.add("CYCLE_A", "CYCLE_B", "1.00", 0)
    builder.add("CYCLE_B", "CYCLE_C", "1.02", 3_600)
    builder.add("CYCLE_C", "CYCLE_A", "0.99", 7_200)

    # R3: OFAC seed and a single-hop neighbour.
    builder.add("AUX_SEED_FUNDER", "SANCTIONED_SEED", "3.0", 200)
    builder.add("SANCTIONED_SEED", "SANCTIONED_1HOP", "0.25", 1_800)
    builder.add("SANCTIONED_1HOP", "AUX_1HOP_DEST", "0.10", 4_000)

    # Down-weighting: high fan-in and high fan-out exchange.
    for i in range(25):
        builder.add(
            f"EXCHANGE_DEPOSITOR_{i:02d}",
            "EXCHANGE_HOT",
            f"{1 + (i % 5)}.{i:02d}",
            10_000 + i * 90,
        )
    for i in range(30):
        builder.add(
            "EXCHANGE_HOT",
            f"EXCHANGE_WITHDRAWER_{i:02d}",
            f"0.{10 + (i % 8):02d}",
            20_000 + i * 120,
        )

    # Extensibility: mixer-tagged source adjacent to a wallet.
    builder.add("MIXER_SOURCE", "MIXER_NEIGHBOR", "2.0", 30_000)
    builder.add("MIXER_NEIGHBOR", "AUX_MIX_DEST_0", "0.9", 33_000)
    builder.add("MIXER_NEIGHBOR", "AUX_MIX_DEST_1", "0.8", 36_000)

    # Contract separation.
    builder.add(
        "AUX_CONTRACT_DEPLOYER",
        "CONTRACT_ADDR",
        "0",
        40_000,
        gas=200_000,
        contract_address="CONTRACT_ADDR",
    )
    for i in range(4):
        builder.add(f"AUX_CONTRACT_CALLER_{i}", "CONTRACT_ADDR", "0.1", 42_000 + i * 600, gas=90_000)

    # True negatives: ordinary wallets with exactly 4 transfers each.
    for i in range(1, 11):
        builder.add(
            f"CLEAN_USER_{i}",
            f"CLEAN_COUNTERPARTY_{i}_A",
            "0.50",
            50_000 + i * 1_000,
        )
        builder.add(
            f"CLEAN_COUNTERPARTY_{i}_A",
            f"CLEAN_USER_{i}",
            "0.25",
            51_000 + i * 1_000,
        )
        builder.add(
            f"CLEAN_USER_{i}",
            f"CLEAN_COUNTERPARTY_{i}_B",
            "0.10",
            52_000 + i * 1_000,
        )
        builder.add(
            f"CLEAN_COUNTERPARTY_{i}_B",
            f"CLEAN_USER_{i}",
            "0.05",
            53_000 + i * 1_000,
        )

    # DBSCAN noise guard: exactly one transaction each.
    for i in range(1, 11):
        builder.add(f"LOW_ACTIVITY_{i}", f"LOW_RECIPIENT_{i}", "0.01", 70_000 + i * 300)

    # Labeled non-required wallets still present in the graph.
    builder.add("AUX_FRAUD_FUNDER", "KAGGLE_FRAUD", "1.0", 80_000)
    builder.add("KAGGLE_FRAUD", "AUX_FRAUD_DEST_0", "0.4", 81_000)
    builder.add("KAGGLE_FRAUD", "AUX_FRAUD_DEST_1", "0.3", 82_000)
    builder.add("AUX_DARK_FUNDER", "DARKLIST_EXTRA", "0.7", 84_000)
    builder.add("DARKLIST_EXTRA", "AUX_DARK_DEST_0", "0.2", 85_000)
    builder.add("DARKLIST_EXTRA", "AUX_DARK_DEST_1", "0.1", 86_000)

    # Background traffic: 200 wallets x 4 transfers = 800 rows.
    background = [f"BG_{i:03d}" for i in range(200)]
    for name in background:
        for _ in range(4):
            counterparty = background[int(rng.integers(0, len(background)))]
            while counterparty == name:
                counterparty = background[int(rng.integers(0, len(background)))]
            value = f"{rng.uniform(0.01, 2.0):.6f}"
            offset = int(rng.integers(0, 30 * 24 * 3600))
            builder.add(name, counterparty, value, offset)


def build_dataframe(builder: TxBuilder) -> pd.DataFrame:
    ordered = sorted(builder.rows, key=lambda row: (row["_offset"], row["_seq"]))
    records = []
    for i, row in enumerate(ordered):
        records.append(
            {
                "tx_hash": "0x"
                + hashlib.sha256(f"chainlens-tx:{SEED}:{i}".encode()).hexdigest(),
                "block_number": BLOCK_START + i,
                "block_timestamp": BASE_TS + timedelta(seconds=row["_offset"]),
                "from_address": row["from_address"],
                "to_address": row["to_address"],
                "value_wei": row["value_wei"],
                "gas": row["gas"],
                "gas_price": row["gas_price"],
                "receipt_status": row["receipt_status"],
                "receipt_contract_address": row["receipt_contract_address"],
            }
        )
    return pd.DataFrame.from_records(records).astype(
        {
            "block_number": "int64",
            "gas": "int64",
            "gas_price": "int64",
            "receipt_status": "int64",
        }
    )


def seed_postgres() -> tuple[int, int, int]:
    """Apply schema and replace Phase 1 label/known-address rows idempotently."""
    engine = get_pg_engine()
    apply_schema(engine)

    labels = [
        {
            "address": address_for("SANCTIONED_SEED"),
            "label": "sanctioned",
            "source": "ofac",
            "confidence": 1.0,
            "as_of": date(2025, 1, 1),
        },
        {
            "address": address_for("KAGGLE_FRAUD"),
            "label": "fraud",
            "source": "kaggle",
            "confidence": 0.9,
            "as_of": date(2025, 1, 1),
        },
        {
            "address": address_for("MIXER_SOURCE"),
            "label": "darklist",
            "source": "darklist",
            "confidence": 0.8,
            "as_of": date(2025, 1, 1),
        },
        {
            "address": address_for("DARKLIST_EXTRA"),
            "label": "darklist",
            "source": "darklist",
            "confidence": 0.8,
            "as_of": date(2025, 1, 1),
        },
    ]
    splits = [
        {"address": address_for("SANCTIONED_SEED"), "split": "train", "seed": SEED},
        {"address": address_for("KAGGLE_FRAUD"), "split": "test", "seed": SEED},
        {"address": address_for("MIXER_SOURCE"), "split": "train", "seed": SEED},
        {"address": address_for("DARKLIST_EXTRA"), "split": "train", "seed": SEED},
    ]
    known = [
        {
            "address": address_for("EXCHANGE_HOT"),
            "category": "exchange",
            "note": "Synthetic exchange hot wallet for down-weighting.",
        },
        {
            "address": address_for("CONTRACT_ADDR"),
            "category": "contract",
            "note": "Synthetic contract address.",
        },
        {
            "address": address_for("MIXER_SOURCE"),
            "category": "mixer",
            "note": "Synthetic mixer-tagged source.",
        },
    ]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM labels"))
        conn.execute(text("DELETE FROM label_splits"))
        conn.execute(text("DELETE FROM known_addresses"))
        conn.execute(
            text(
                """
                INSERT INTO labels (address, label, source, confidence, as_of)
                VALUES (:address, :label, :source, :confidence, :as_of)
                """
            ),
            labels,
        )
        conn.execute(
            text(
                "INSERT INTO label_splits (address, split, seed) "
                "VALUES (:address, :split, :seed)"
            ),
            splits,
        )
        conn.execute(
            text(
                "INSERT INTO known_addresses (address, category, note) "
                "VALUES (:address, :category, :note)"
            ),
            known,
        )

    return len(labels), len(splits), len(known)


def fixture_summary(df: pd.DataFrame) -> pd.DataFrame:
    fixtures = [
        ("FANOUT_WHALE", "R2 fan-out (>= 40 unique outbound)"),
        ("CYCLE_A", "R1 cycle A -> B -> C -> A"),
        ("CYCLE_B", "R1 cycle A -> B -> C -> A"),
        ("CYCLE_C", "R1 cycle A -> B -> C -> A"),
        ("SANCTIONED_SEED", "R3 seed (ofac label, train split)"),
        ("SANCTIONED_1HOP", "R3 one-hop neighbour"),
        ("EXCHANGE_HOT", "Exchange down-weighting"),
        ("MIXER_SOURCE", "Mixer-tagged source (R6+ extensibility)"),
        ("MIXER_NEIGHBOR", "Mixer adjacency (R6+ extensibility)"),
        ("CONTRACT_ADDR", "Contract separation (is_contract=true)"),
        *[(f"CLEAN_USER_{i}", "True negative") for i in range(1, 11)],
        *[(f"LOW_ACTIVITY_{i}", "DBSCAN noise guard (exactly 1 tx)") for i in range(1, 11)],
        ("KAGGLE_FRAUD", "Ground-truth fraud label (test split)"),
        ("DARKLIST_EXTRA", "Ground-truth darklist label"),
    ]
    rows = []
    for name, expected in fixtures:
        address = address_for(name)
        count = int(
            ((df["from_address"] == address) | (df["to_address"] == address)).sum()
        )
        rows.append(
            {
                "fixture": name,
                "address": address,
                "expected_signal": expected,
                "tx_count": count,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(seed=SEED)
    builder = TxBuilder()
    build_fixture_rows(builder, rng)
    df = build_dataframe(builder)

    if not 800 <= len(df) <= 1200:
        raise SystemExit(f"Sample size {len(df)} outside required 800-1200 rows")

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)

    summary = fixture_summary(df)
    print("ChainLens synthetic sample fixtures")
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {len(df):,} transactions to {PARQUET_PATH.relative_to(ROOT)}")
    print(
        f"Blocks {int(df['block_number'].min()):,}-{int(df['block_number'].max()):,}; "
        f"timestamps {df['block_timestamp'].min()} to {df['block_timestamp'].max()}"
    )

    n_labels, n_splits, n_known = seed_postgres()
    print(
        f"Seeded PostgreSQL: labels={n_labels}, label_splits={n_splits}, "
        f"known_addresses={n_known} (truncate + reload)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
