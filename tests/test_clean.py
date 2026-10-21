from __future__ import annotations

from decimal import Decimal

import pandas as pd

from chainlens.config import project_path

CLEAN_PATH = project_path("data/processed/transactions_clean.parquet")


def test_clean_parquet_exists_and_is_smaller(pg_engine):
    assert CLEAN_PATH.exists()
    clean = pd.read_parquet(CLEAN_PATH)
    assert not clean.empty

    raw_path = project_path("data/raw/transactions.parquet")
    if not raw_path.exists():
        raw_path = project_path("data/sample/transactions.parquet")
    raw = pd.read_parquet(raw_path)
    assert len(clean) <= len(raw)


def test_failed_transactions_are_absent():
    clean = pd.read_parquet(CLEAN_PATH)
    assert (clean["receipt_status"] == 1).all()
    assert (clean["from_address"] != clean["to_address"]).all()
    assert clean["is_contract"].notna().all()


def test_clean_output_is_recorded_in_manifest(pg_engine):
    manifest = pd.read_sql(
        "SELECT row_count, block_start, block_end FROM data_manifest "
        "WHERE file_path = 'data/processed/transactions_clean.parquet'",
        pg_engine,
    )
    assert len(manifest) == 1
    assert manifest.iloc[0]["row_count"] == len(pd.read_parquet(CLEAN_PATH))
    assert manifest.iloc[0]["block_start"] <= manifest.iloc[0]["block_end"]


def test_value_eth_matches_wei_without_float_drift():
    clean = pd.read_parquet(CLEAN_PATH)
    values = clean.assign(wei=clean["value_wei"].map(lambda v: Decimal(str(v))))
    row = values.sort_values("wei", ascending=False).iloc[0]
    expected = (row["wei"] / Decimal(10**18)).quantize(Decimal("0.000000000000000001"))
    actual = Decimal(str(row["value_eth"]))
    assert actual == expected

    # A value large enough to exceed int64 wei also survives exactly.
    large = clean.loc[clean["value_eth"] > Decimal("1")].iloc[0]
    assert Decimal(str(large["value_eth"])) == (
        Decimal(str(large["value_wei"])) / Decimal(10**18)
    ).quantize(Decimal("0.000000000000000001"))
