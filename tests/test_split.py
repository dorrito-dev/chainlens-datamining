from __future__ import annotations

import json

import pandas as pd

from chainlens.config import project_path


def test_split_is_disjoint_and_seeded(pg_engine):
    splits = pd.read_sql("SELECT * FROM label_splits", pg_engine)
    assert not splits.empty
    assert set(splits["split"]) == {"train", "test"}
    assert set(splits["seed"]) == {42}
    assert not splits["address"].duplicated().any()
    assert splits["address"].str.fullmatch(r"^0x[0-9a-f]{40}$").all()


def test_every_labeled_address_has_a_split(pg_engine):
    labels = pd.read_sql("SELECT DISTINCT address FROM labels", pg_engine)
    splits = pd.read_sql("SELECT address FROM label_splits", pg_engine)
    missing = set(labels["address"]) - set(splits["address"])
    assert not missing, f"{len(missing)} labeled addresses have no split"


def test_each_source_with_two_positives_has_test_positives(pg_engine):
    labels = pd.read_sql(
        "SELECT address, source FROM labels GROUP BY address, source", pg_engine
    )
    splits = pd.read_sql("SELECT address, split FROM label_splits", pg_engine)
    merged = labels.merge(splits, on="address", how="inner")
    by_source = merged.groupby("source")["split"]
    for source, values in by_source:
        total = len(values)
        if total >= 2:
            assert (values == "test").any(), f"no test positives for source {source}"


def test_window_decision_is_recorded_in_manifest(pg_engine):
    manifest = pd.read_sql(
        "SELECT row_count, metadata FROM data_manifest "
        "WHERE file_path = 'data/processed/window_decision.json'",
        pg_engine,
    )
    assert len(manifest) == 1
    assert manifest.iloc[0]["metadata"]["stage"] == "overlap_gate"
    assert manifest.iloc[0]["metadata"]["decision"] in {"time", "label_anchored"}


def test_split_summary_is_recorded():
    summary = json.loads(
        project_path("data/processed/split_summary.json").read_text(encoding="utf-8")
    )
    assert summary["seed"] == 42
    assert summary["train_test_disjoint"] is True
    assert summary["train_addresses"] > 0
    assert summary["test_addresses"] > 0
