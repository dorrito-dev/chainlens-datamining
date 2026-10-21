from __future__ import annotations

import pandas as pd
import pytest

from chainlens.apriori import (
    ANTECEDENT_ITEMS,
    apply_rules,
    build_apriori_table,
    build_scoring_table,
    mine_rules,
)


def _synthetic_table(rows: int = 100) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "address": [f"0x{index:040x}" for index in range(rows)],
        }
    )
    for column in ANTECEDENT_ITEMS:
        table[column] = False
    table["flagged"] = False
    table.loc[:29, "high_fanout"] = True
    table.loc[:19, "flagged"] = True
    return table


def test_synthetic_frequent_itemset_yields_known_rule():
    table = _synthetic_table()
    rules = mine_rules(table, min_support=0.05, min_lift=2.0)
    assert not rules.empty
    assert ("high_fanout",) in set(rules["antecedents"])
    rule = rules[rules["antecedents"] == ("high_fanout",)].iloc[0]
    assert float(rule["support"]) == pytest.approx(0.20)
    assert float(rule["confidence"]) == pytest.approx(20 / 30)
    assert float(rule["lift"]) > 2.0

    flags = apply_rules(table, rules)
    assert len(flags) == 30
    assert {row["address"] for row in flags} == set(table.loc[:29, "address"])
    assert all(row["signal_name"] == "apriori_match" for row in flags)


def test_apriori_consequent_is_only_flagged():
    table = _synthetic_table()
    rules = mine_rules(table, min_support=0.05, min_lift=1.0)
    assert not rules.empty
    for antecedent in rules["antecedents"]:
        assert "flagged" not in antecedent


def test_mining_table_excludes_every_test_split_address(pg_engine):
    table = build_apriori_table(pg_engine)
    test_addresses = set(
        pd.read_sql(
            "SELECT address FROM label_splits WHERE split = 'test'",
            pg_engine,
        )["address"]
    )
    assert not (set(table["address"]) & test_addresses)

    feature_addresses = set(
        pd.read_sql("SELECT address FROM address_features", pg_engine)["address"]
    )
    assert set(table["address"]) <= feature_addresses
    assert len(table) == len(feature_addresses - test_addresses)


def test_mining_table_has_required_boolean_items(pg_engine):
    table = build_apriori_table(pg_engine)
    assert set(ANTECEDENT_ITEMS + ["flagged"]) <= set(table.columns)
    for column in ANTECEDENT_ITEMS + ["flagged"]:
        assert table[column].dtype == bool


def test_scoring_table_includes_test_addresses_without_test_labels(pg_engine):
    table = build_scoring_table(pg_engine)
    features = set(
        pd.read_sql("SELECT address FROM address_features", pg_engine)["address"]
    )
    test_addresses = set(
        pd.read_sql(
            "SELECT address FROM label_splits WHERE split = 'test'",
            pg_engine,
        )["address"]
    )
    assert set(table["address"]) == features
    test_rows = table[table["address"].isin(test_addresses)]
    assert not test_rows.empty
    assert not test_rows["flagged"].any()
