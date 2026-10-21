"""Train-only Apriori mining and profile matching for R5."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.detection import (
    APRIORI_SIGNALS,
    load_rules_config,
    write_flags,
)

BOOLEAN_ITEMS = [
    "high_fanout",
    "high_tx_count",
    "low_value",
    "high_value",
    "many_round",
    "is_contract",
    "recent_account",
    "high_pagerank",
    "flagged",
]

ANTECEDENT_ITEMS = [item for item in BOOLEAN_ITEMS if item != "flagged"]


def _build_table(engine: Engine, *, include_test: bool) -> pd.DataFrame:
    """Build boolean profiles; the mining path excludes test addresses."""
    split_join = (
        ""
        if include_test
        else "LEFT JOIN label_splits ls ON ls.address = af.address"
    )
    split_filter = (
        ""
        if include_test
        else "WHERE ls.split IS NULL OR ls.split = 'train'"
    )
    query = text(
        f"""
        SELECT
            af.address,
            af.unique_counterparties,
            af.tx_count_total,
            af.median_value_eth,
            af.total_value_eth,
            af.round_number_ratio,
            af.is_contract,
            af.first_seen,
            af.last_seen,
            (train_label.address IS NOT NULL) AS flagged,
            (pf.address IS NOT NULL) AS high_pagerank
        FROM address_features af
        {split_join}
        LEFT JOIN (
            SELECT DISTINCT l.address
            FROM labels l
            JOIN label_splits s ON s.address = l.address
            WHERE s.split = 'train'
        ) AS train_label ON train_label.address = af.address
        LEFT JOIN (
            SELECT address FROM address_flags
            WHERE signal_name = 'pagerank_top5'
        ) pf ON pf.address = af.address
        {split_filter}
        ORDER BY af.address
        """
    )
    with engine.connect() as conn:
        frame = pd.read_sql(query, conn)
        window_end = pd.to_datetime(
            conn.execute(
                text("SELECT max(last_seen) FROM address_features")
            ).scalar_one(),
            utc=True,
        )
        test_addresses = set(
            conn.execute(
                text("SELECT address FROM label_splits WHERE split = 'test'")
            ).scalars()
        )

    if not include_test:
        leaked = set(frame["address"]) & test_addresses
        if leaked:
            raise RuntimeError(
                f"Apriori table contains {len(leaked)} test-split address(es); "
                "refusing to mine"
            )

    table = pd.DataFrame({"address": frame["address"]})
    table["high_fanout"] = frame["unique_counterparties"].fillna(0) >= 20
    table["high_tx_count"] = frame["tx_count_total"].fillna(0) >= 10
    table["low_value"] = frame["median_value_eth"].fillna(0) < 0.1
    table["high_value"] = frame["total_value_eth"].fillna(0) >= 10
    table["many_round"] = frame["round_number_ratio"].fillna(0) >= 0.5
    table["is_contract"] = frame["is_contract"].fillna(False).astype(bool)
    first_seen = pd.to_datetime(frame["first_seen"], utc=True)
    table["recent_account"] = (
        first_seen >= window_end - pd.Timedelta(days=7)
        if not pd.isna(window_end)
        else False
    )
    table["high_pagerank"] = frame["high_pagerank"].fillna(False).astype(bool)
    table["flagged"] = frame["flagged"].fillna(False).astype(bool)
    for column in BOOLEAN_ITEMS:
        table[column] = table[column].astype(bool)
    return table


def build_apriori_table(engine: Engine) -> pd.DataFrame:
    """Build the train-only mining table; test addresses are absent entirely."""
    return _build_table(engine, include_test=False)


def build_scoring_table(engine: Engine) -> pd.DataFrame:
    """Build matching profiles for every address, including test commitments.

    The ``flagged`` column is still populated only from train labels. Mining
    must use :func:`build_apriori_table`, never this table.
    """
    return _build_table(engine, include_test=True)


def mine_rules(
    table: pd.DataFrame,
    min_support: float = 0.05,
    min_lift: float = 2.0,
) -> pd.DataFrame:
    """Return rules whose consequent is exactly ``flagged``."""
    columns = ["address", *BOOLEAN_ITEMS]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise RuntimeError(f"Apriori table missing columns: {missing}")
    items = table[BOOLEAN_ITEMS].astype(bool)
    if items.empty or not items["flagged"].any():
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift"]
        )

    frequent = apriori(items, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift"]
        )
    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)
    if rules.empty:
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift"]
        )

    keep: list[dict] = []
    for row in rules.itertuples(index=False):
        antecedent = set(row.antecedents)
        consequent = set(row.consequents)
        if consequent != {"flagged"}:
            continue
        if "flagged" in antecedent:
            continue
        keep.append(
            {
                "antecedents": tuple(sorted(antecedent)),
                "support": float(row.support),
                "confidence": float(row.confidence),
                "lift": float(row.lift),
            }
        )
    if not keep:
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift"]
        )
    return (
        pd.DataFrame(keep)
        .sort_values(
            ["confidence", "lift", "antecedents"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def apply_rules(table: pd.DataFrame, rules: pd.DataFrame) -> list[dict]:
    """Flag every address whose profile is a superset of a rule antecedent."""
    if rules.empty:
        return []
    best: dict[str, dict] = {}
    for rule in rules.itertuples(index=False):
        antecedent = tuple(rule.antecedents)
        if not antecedent:
            continue
        mask = table[list(antecedent)].all(axis=1)
        for address in table.loc[mask, "address"]:
            current = best.get(address)
            candidate = {
                "address": address,
                "signal_name": "apriori_match",
                "signal_value": float(rule.confidence),
                "evidence": {
                    "antecedent": list(antecedent),
                    "support": float(rule.support),
                    "confidence": float(rule.confidence),
                    "lift": float(rule.lift),
                },
            }
            if current is None:
                best[address] = candidate
                continue
            key = (candidate["signal_value"], candidate["evidence"]["lift"])
            current_key = (
                current["signal_value"],
                current["evidence"]["lift"],
            )
            if key > current_key:
                best[address] = candidate
    return [best[address] for address in sorted(best)]


def run_apriori(engine: Engine) -> dict:
    params = next(
        rule["params"]
        for rule in load_rules_config()["rules"]
        if rule["id"] == "R5"
    )
    min_support = float(params.get("min_support", 0.05))
    min_lift = float(params.get("min_lift", 2.0))
    mining_table = build_apriori_table(engine)
    scoring_table = build_scoring_table(engine)
    rules = mine_rules(mining_table, min_support=min_support, min_lift=min_lift)
    flags = apply_rules(scoring_table, rules)
    written = write_flags(engine, flags, reset_signals=APRIORI_SIGNALS)
    print(
        f"Apriori: {len(mining_table)} mining rows / {len(scoring_table)} scoring rows, "
        f"{len(rules)} rule(s), {len(flags)} flagged address(es), "
        f"min_support={min_support}, min_lift={min_lift}"
    )
    print("Apriori test-split overlap: 0 (asserted in build)")
    return {
        "mining_rows": len(mining_table),
        "scoring_rows": len(scoring_table),
        "rules": len(rules),
        "flagged_addresses": len(flags),
        "flags_written": written,
        "min_support": min_support,
        "min_lift": min_lift,
    }
