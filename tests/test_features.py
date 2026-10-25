"""Cleaning, DECIMAL value handling, features, and train-only scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features as features_mod


def addr(n: int) -> str:
    return f"0x{n:040x}"


def raw_rows() -> pd.DataFrame:
    A, B, C, D, E, F = (addr(i) for i in range(1, 7))

    def row(tx_hash: str, block: int, ts: str, frm: str, to: str, value: str,
            status: int = 1, contract: str | None = None) -> dict:
        return {
            "tx_hash": tx_hash,
            "block_number": block,
            "block_timestamp": pd.Timestamp(ts),
            "from_address": frm,
            "to_address": to,
            "value_wei": value,
            "gas": 21_000,
            "gas_price": 1_000_000_000,
            "receipt_status": status,
            "receipt_contract_address": contract,
            "is_error": 0,
            "source": "test",
            "fetched_at": "2024-01-01T00:00:00Z",
        }

    return pd.DataFrame([
        row("0x1", 1, "2024-01-01T00:00:00Z", A, B, str(1 * 10**18)),
        row("0x2", 2, "2024-01-01T01:00:00Z", A, C, str(2 * 10**18)),
        row("0x3", 3, "2024-01-01T02:00:00Z", B, A, str(1 * 10**18)),
        row("0x1", 1, "2024-01-01T00:00:00Z", A, B, str(1 * 10**18)),  # duplicate
        row("0x4", 4, "2024-01-01T02:30:00Z", A, D, str(1 * 10**18), status=0),  # failed
        row("0x5", 5, "2024-01-01T02:45:00Z", A, A, str(1 * 10**18)),  # self-transfer
        row("0x6", 6, "2024-01-01T03:00:00Z", A, E, str(5 * 10**17), contract=E),  # contract
        row("0x8", 7, "2024-01-01T03:15:00Z", F, B, str(25 * 10**16)),  # outbound-only
    ])


def test_clean_deduplicates_filters_and_keeps_decimal_eth() -> None:
    con = features_mod.connect()
    try:
        clean = features_mod.clean_transactions(con, raw_rows())
        assert len(clean) == 5
        assert set(clean["tx_hash"]) == {"0x1", "0x2", "0x3", "0x6", "0x8"}
        assert addr(4) not in set(clean["from_address"]) | set(clean["to_address"])
        assert clean["value_eth"].sum() == pytest.approx(4.75)
        assert float(clean.loc[clean["tx_hash"] == "0x6", "value_eth"].iloc[0]) == pytest.approx(0.5)
    finally:
        con.close()


def test_features_match_hand_computed_values() -> None:
    con = features_mod.connect()
    try:
        features_mod.clean_transactions(con, raw_rows())
        frame = features_mod.compute_features(con)
        by_addr = frame.set_index("address")
        A, B, C, E, F = addr(1), addr(2), addr(3), addr(5), addr(6)

        assert set(frame["address"]) == {A, B, C, E, F}
        assert by_addr.loc[A, "tx_count_out"] == 3
        assert by_addr.loc[A, "tx_count_in"] == 1
        assert by_addr.loc[A, "tx_count_total"] == 4
        assert by_addr.loc[A, "unique_counterparties"] == 3
        assert by_addr.loc[A, "fanout_ratio"] == pytest.approx(1.0)
        assert by_addr.loc[A, "out_in_ratio"] == pytest.approx(3.0)
        assert by_addr.loc[A, "total_value_eth"] == pytest.approx(3.5)
        assert by_addr.loc[A, "median_value_eth"] == pytest.approx(1.0)
        assert by_addr.loc[A, "net_flow_eth"] == pytest.approx(-2.5)
        assert by_addr.loc[A, "round_number_ratio"] == pytest.approx(2 / 3)
        assert bool(by_addr.loc[A, "is_contract"]) is False

        assert by_addr.loc[B, "tx_count_out"] == 1
        assert by_addr.loc[B, "tx_count_in"] == 2
        assert by_addr.loc[B, "tx_count_total"] == 3
        assert by_addr.loc[B, "unique_counterparties"] == 1
        assert by_addr.loc[B, "fanout_ratio"] == pytest.approx(1.0)
        assert by_addr.loc[B, "out_in_ratio"] == pytest.approx(0.5)

        assert by_addr.loc[C, "tx_count_out"] == 0
        assert by_addr.loc[C, "tx_count_in"] == 1
        assert by_addr.loc[C, "out_in_ratio"] == pytest.approx(0.0)

        assert bool(by_addr.loc[E, "is_contract"]) is True
        assert by_addr.loc[E, "tx_count_in"] == 1

        assert by_addr.loc[F, "tx_count_out"] == 1
        assert by_addr.loc[F, "tx_count_in"] == 0
        assert pd.isna(by_addr.loc[F, "out_in_ratio"])
    finally:
        con.close()


def test_scaler_is_fitted_on_train_only() -> None:
    con = features_mod.connect()
    try:
        features_mod.clean_transactions(con, raw_rows())
        frame = features_mod.compute_features(con)
    finally:
        con.close()

    A, B, C = addr(1), addr(2), addr(3)
    scaled, version, rows = features_mod.scale_features(frame, {A, B})
    assert len(version) == 12
    assert {row["feature_name"] for row in rows} == set(features_mod.SCALER_SPECS)
    total = next(row for row in rows if row["feature_name"] == "total_value_eth")
    assert total["fitted_on_split"] == "train"

    train_values = frame.loc[frame["address"].isin({A, B}), "total_value_eth"].astype(float)
    mean = float(np.log1p(train_values).mean())
    std = float(np.log1p(train_values).std(ddof=0))
    expected_c = (np.log1p(0.0) - mean) / std
    actual_c = scaled.set_index("address").loc[C, "total_value_eth_scaled"]
    assert actual_c == pytest.approx(expected_c)

    # A train-only mean is not the full-data mean.
    full_mean = float(np.log1p(frame["total_value_eth"].astype(float)).mean())
    assert mean != pytest.approx(full_mean)


def test_schema_guard_accepts_value_eth_only() -> None:
    frame = pd.DataFrame({"value_eth": [1.5, 0.0]})
    guarded = features_mod.sample_schema_guard(frame)
    assert "value_wei" in guarded.columns
    assert guarded.loc[0, "value_wei"] == "1500000000000000000"
    assert guarded.loc[1, "value_wei"] == "0"
