from __future__ import annotations

import pandas as pd

ADDRESS_PATTERN = r"^0x[0-9a-f]{40}$"


def test_labels_have_all_sources_and_valid_addresses(pg_engine):
    labels = pd.read_sql("SELECT * FROM labels", pg_engine)
    assert not labels.empty
    assert {"ofac", "darklist", "kaggle"} <= set(labels["source"])
    assert labels["address"].str.fullmatch(ADDRESS_PATTERN).all()
    assert labels["as_of"].notna().all()
    assert labels["confidence"].between(0, 1).all()


def test_labels_are_unique_per_address_source_label(pg_engine):
    labels = pd.read_sql("SELECT address, source, label FROM labels", pg_engine)
    assert not labels.duplicated(subset=["address", "source", "label"]).any()


def test_known_addresses_cover_exchange_mixer_and_contract(pg_engine):
    known = pd.read_sql("SELECT address, category FROM known_addresses", pg_engine)
    assert {"exchange", "mixer", "contract"} <= set(known["category"])
    assert known["address"].str.fullmatch(ADDRESS_PATTERN).all()
