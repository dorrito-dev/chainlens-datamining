from __future__ import annotations

import json

import pandas as pd
import pytest
from sqlalchemy import text

from chainlens.detection import CLUSTER_SIGNALS, load_detection_config
from chainlens.features import get_scaled_feature_columns, scaled_columns_from_names


def _as_dict(value):
    return value if isinstance(value, dict) else json.loads(value)


def test_every_eligible_address_has_a_cluster_row(pg_engine):
    eligible = set(
        pd.read_sql(
            "SELECT address FROM address_features WHERE tx_count_total >= 3",
            pg_engine,
        )["address"]
    )
    flagged = set(
        pd.read_sql(
            "SELECT address FROM address_flags WHERE signal_name = 'dbscan_cluster'",
            pg_engine,
        )["address"]
    )
    assert eligible
    assert flagged == eligible


def test_ineligible_addresses_are_not_clustered(pg_engine):
    ineligible = set(
        pd.read_sql(
            "SELECT address FROM address_features WHERE tx_count_total < 3",
            pg_engine,
        )["address"]
    )
    flagged = set(
        pd.read_sql(
            "SELECT address FROM address_flags WHERE signal_name = 'dbscan_cluster'",
            pg_engine,
        )["address"]
    )
    assert not (ineligible & flagged)


def test_noise_rows_equal_negative_cluster_labels(pg_engine):
    with pg_engine.connect() as conn:
        noise = conn.execute(
            text(
                "SELECT count(*) FROM address_flags "
                "WHERE signal_name = 'dbscan_noise'"
            )
        ).scalar_one()
        negatives = conn.execute(
            text(
                "SELECT count(*) FROM address_flags "
                "WHERE signal_name = 'dbscan_cluster' AND signal_value::int = -1"
            )
        ).scalar_one()
    assert int(noise) == int(negatives)


def test_cluster_evidence_records_config(pg_engine):
    config = load_detection_config()["dbscan"]
    row = pd.read_sql(
        "SELECT evidence FROM address_flags WHERE signal_name = 'dbscan_cluster' LIMIT 1",
        pg_engine,
    )
    assert len(row) == 1
    evidence = _as_dict(row.iloc[0]["evidence"])
    assert float(evidence["eps"]) == pytest.approx(float(config["eps"]))
    assert int(evidence["min_samples"]) == int(config["min_samples"])
    assert evidence["scaler_version"]


def test_scaled_columns_are_discovered_from_feature_scaler(pg_engine):
    columns, version = get_scaled_feature_columns(pg_engine)
    expected = [
        f"{name}_scaled"
        for name in pd.read_sql(
            "SELECT feature_name FROM feature_scaler WHERE std > 0 ORDER BY feature_name",
            pg_engine,
        )["feature_name"]
    ]
    assert columns == expected
    assert version


def test_missing_scaled_column_raises_named_error():
    with pytest.raises(RuntimeError, match="tx_count_in_scaled.*scaler_version=v-test"):
        scaled_columns_from_names(
            ["tx_count_in"],
            ["tx_count_out_scaled"],
            "v-test",
        )


def test_cluster_signal_set_is_exact():
    assert CLUSTER_SIGNALS == {"dbscan_cluster", "dbscan_noise"}
