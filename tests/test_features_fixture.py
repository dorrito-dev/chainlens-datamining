from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import text

from chainlens.features import ensure_fixture_labels
from chainlens.fixtures import address_for

pytestmark = pytest.mark.skipif(
    os.getenv("CHAINLENS_SAMPLE_MODE", "").lower() not in {"1", "true", "yes", "on"},
    reason="fixture feature assertions require CHAINLENS_SAMPLE_MODE=1",
)


def _feature(pg_engine, name: str) -> pd.Series:
    frame = pd.read_sql(
        text("SELECT * FROM address_features WHERE address = :address"),
        pg_engine,
        params={"address": address_for(name)},
    )
    assert len(frame) == 1, f"{name} missing from address_features"
    return frame.iloc[0]


def test_fanout_whale_matches_phase1_fixture(pg_engine):
    whale = _feature(pg_engine, "FANOUT_WHALE")
    assert int(whale["tx_count_out"]) >= 40
    assert float(whale["fanout_ratio"]) == pytest.approx(1.0, abs=1e-6)


def test_cycle_fixture_addresses_have_enough_activity(pg_engine):
    for name in ("CYCLE_A", "CYCLE_B", "CYCLE_C"):
        feature = _feature(pg_engine, name)
        assert int(feature["tx_count_total"]) >= 2


def test_fixture_split_and_label_placement(pg_engine):
    rows = pd.read_sql(
        text("SELECT address, split FROM label_splits WHERE address = ANY(:addresses)"),
        pg_engine,
        params={
            "addresses": [
                address_for("SANCTIONED_SEED"),
                address_for("KAGGLE_FRAUD"),
                address_for("DARKLIST_EXTRA"),
            ]
        },
    )
    placement = dict(zip(rows["address"], rows["split"], strict=True))
    assert placement[address_for("SANCTIONED_SEED")] == "train"
    assert placement[address_for("KAGGLE_FRAUD")] == "test"
    assert placement[address_for("DARKLIST_EXTRA")] == "train"

    labels = pd.read_sql(
        "SELECT address, source FROM labels WHERE source = 'fixture'", pg_engine
    )
    assert len(labels) == 3
    assert set(labels["address"]) == {
        address_for("SANCTIONED_SEED"),
        address_for("KAGGLE_FRAUD"),
        address_for("DARKLIST_EXTRA"),
    }


def test_fixture_backfill_is_idempotent(pg_engine):
    ensure_fixture_labels(pg_engine)
    ensure_fixture_labels(pg_engine)
    count = pd.read_sql(
        "SELECT count(*) AS n FROM labels WHERE source = 'fixture'", pg_engine
    ).iloc[0]["n"]
    assert count == 3
