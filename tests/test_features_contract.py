from __future__ import annotations

import numpy as np
import pandas as pd

from chainlens.features import compute_scaler_version


def test_address_features_has_no_flagged_column(pg_engine):
    features = pd.read_sql("SELECT * FROM address_features", pg_engine)
    assert "flagged" not in features.columns


def test_tx_count_total_is_the_sum(pg_engine):
    features = pd.read_sql(
        "SELECT tx_count_out, tx_count_in, tx_count_total FROM address_features",
        pg_engine,
    )
    assert not features.empty
    assert (features["tx_count_total"] == features["tx_count_out"] + features["tx_count_in"]).all()


def test_fanout_ratio_is_bounded(pg_engine):
    features = pd.read_sql("SELECT fanout_ratio FROM address_features", pg_engine)
    assert features["fanout_ratio"].between(0, 1).all()


def test_scaler_version_matches_feature_scaler_rows(pg_engine):
    versions = pd.read_sql(
        "SELECT DISTINCT scaler_version FROM address_features", pg_engine
    )
    assert len(versions) == 1
    rows = pd.read_sql(
        "SELECT feature_name, mean, std FROM feature_scaler", pg_engine
    ).to_dict("records")
    assert versions.iloc[0]["scaler_version"] == compute_scaler_version(rows)


def test_surviving_scaled_columns_are_finite_for_active_addresses(pg_engine):
    features = pd.read_sql("SELECT * FROM address_features", pg_engine)
    scaled_features = pd.read_sql(
        "SELECT feature_name FROM feature_scaler", pg_engine
    )["feature_name"]
    active = features[features["tx_count_total"] >= 3]
    assert not active.empty
    for name in scaled_features:
        column = f"{name}_scaled"
        assert column in active.columns
        values = pd.to_numeric(active[column], errors="coerce")
        assert np.isfinite(values).all(), f"{column} has NaN/inf"
