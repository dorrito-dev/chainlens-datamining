from __future__ import annotations

import pandas as pd

from chainlens.features import run_features


def test_feature_scaler_was_fitted_on_train_only(pg_engine):
    scaler = pd.read_sql(
        "SELECT feature_name, std, fitted_on_split, row_count, scaler_version "
        "FROM feature_scaler",
        pg_engine,
    )
    assert not scaler.empty
    allowed = {"train", "train_fallback_full_std"}
    assert set(scaler["fitted_on_split"]).issubset(allowed)
    assert (scaler["std"] > 0).all()
    assert scaler["row_count"].notna().all()
    assert scaler["scaler_version"].nunique() == 1


def test_tx_count_in_uses_full_std_fallback(pg_engine):
    row = pd.read_sql(
        "SELECT std, fitted_on_split FROM feature_scaler WHERE feature_name = 'tx_count_in'",
        pg_engine,
    )
    assert len(row) == 1
    assert float(row.iloc[0]["std"]) > 0
    assert row.iloc[0]["fitted_on_split"] == "train_fallback_full_std"


def test_feature_scaler_row_count_matches_train_intersection(pg_engine):
    expected = pd.read_sql(
        """
        SELECT count(*) AS n
        FROM address_features af
        JOIN label_splits ls ON ls.address = af.address
        WHERE ls.split = 'train'
        """,
        pg_engine,
    ).iloc[0]["n"]
    scaler = pd.read_sql("SELECT row_count FROM feature_scaler", pg_engine)
    assert (scaler["row_count"] == expected).all()


def test_two_runs_are_identical(pg_engine, monkeypatch):
    fixture_count = pd.read_sql(
        "SELECT count(*) AS n FROM labels WHERE source = 'fixture'", pg_engine
    ).iloc[0]["n"]
    if int(fixture_count):
        monkeypatch.setenv("CHAINLENS_SAMPLE_MODE", "1")
    first_summary = run_features()
    first = pd.read_sql(
        "SELECT * FROM address_features ORDER BY address", pg_engine
    )
    first_version = first_summary["scaler_version"]

    second_summary = run_features()
    second = pd.read_sql(
        "SELECT * FROM address_features ORDER BY address", pg_engine
    )
    second_version = second_summary["scaler_version"]

    assert first_version == second_version
    pd.testing.assert_frame_equal(first, second)
