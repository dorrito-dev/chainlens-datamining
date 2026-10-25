"""Graph rules, DBSCAN, Apriori, and Louvain."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from patterns import (
    APRIORI_ITEMS,
    _boolean_table,
    compute_pagerank,
    detect_cycles,
    detect_fanout,
    detect_louvain,
    detect_sanctioned_1hop,
    mine_apriori,
    run_dbscan,
)


def addr(n: int) -> str:
    return f"0x{n:040x}"


def add_edge(graph: nx.DiGraph, u: str, v: str, value: float, start: str, end: str) -> None:
    graph.add_edge(
        u, v, total_value_eth=value, tx_count=1,
        first_tx_ts=pd.Timestamp(start, tz="UTC"),
        last_tx_ts=pd.Timestamp(end, tz="UTC"),
    )


def test_cycle_rule_filters_value_time_and_length() -> None:
    g = nx.DiGraph()
    A, B, C = addr(1), addr(2), addr(3)
    add_edge(g, A, B, 10, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, B, C, 10, "2024-01-01T00:20:00Z", "2024-01-01T00:30:00Z")
    add_edge(g, C, A, 10, "2024-01-01T00:40:00Z", "2024-01-01T00:50:00Z")

    D, E, F = addr(4), addr(5), addr(6)
    add_edge(g, D, E, 10, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, E, F, 10, "2024-01-01T00:20:00Z", "2024-01-01T00:30:00Z")
    add_edge(g, F, D, 20, "2024-01-01T00:40:00Z", "2024-01-01T00:50:00Z")  # value mismatch

    X, Y, Z = addr(9), addr(10), addr(11)
    add_edge(g, X, Y, 10, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, Y, Z, 10, "2024-01-02T00:00:00Z", "2024-01-02T00:10:00Z")
    add_edge(g, Z, X, 10, "2024-01-04T00:00:00Z", "2024-01-04T00:10:00Z")  # too long

    G, H = addr(7), addr(8)
    add_edge(g, G, H, 10, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, H, G, 10, "2024-01-01T00:20:00Z", "2024-01-01T00:30:00Z")  # length 2

    L, M, N = addr(12), addr(13), addr(14)
    add_edge(g, L, M, 0, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, M, N, 0, "2024-01-01T00:20:00Z", "2024-01-01T00:30:00Z")
    add_edge(g, N, L, 0, "2024-01-01T00:40:00Z", "2024-01-01T00:50:00Z")  # zero value

    flags, stats = detect_cycles(g)
    assert set(a for a, signal in flags if signal == "cycle_member") == {A, B, C}
    assert stats["qualifying_cycles"] == 1
    assert stats["flagged_addresses"] == 3
    assert flags[(A, "cycle_member")]["evidence"]["cycle_count"] == 1

    for bad in (D, E, F, G, H, X, Y, Z, L, M, N):
        assert (bad, "cycle_member") not in flags


def test_fanout_rule_and_deck_equivalent_diagnostic() -> None:
    a, b, c, d = addr(1), addr(2), addr(3), addr(4)
    features = pd.DataFrame([
        {"address": a, "unique_counterparties": 60, "tx_count_out": 60,
         "fanout_ratio": 1.0, "out_in_ratio": 6.0},
        {"address": b, "unique_counterparties": 10, "tx_count_out": 30,
         "fanout_ratio": 0.33, "out_in_ratio": 0.5},
        {"address": c, "unique_counterparties": 25, "tx_count_out": 25,
         "fanout_ratio": 1.0, "out_in_ratio": 1.0},
        {"address": d, "unique_counterparties": 60, "tx_count_out": 60,
         "fanout_ratio": 1.0, "out_in_ratio": None},
    ])
    flags = detect_fanout(features)
    assert (a, "fanout") in flags
    assert (b, "fanout") not in flags
    assert (c, "fanout") in flags
    assert (d, "fanout") in flags
    assert flags[(a, "fanout")]["evidence"]["r2_deck_equivalent"] is True
    assert flags[(c, "fanout")]["evidence"]["r2_deck_equivalent"] is False
    assert flags[(d, "fanout")]["evidence"]["out_in_ratio"] is None


def test_pagerank_and_zero_value_floor() -> None:
    g = nx.DiGraph()
    A, B, C = addr(1), addr(2), addr(3)
    add_edge(g, A, B, 100, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, A, C, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, C, A, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(g, B, A, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")

    flags, scores = compute_pagerank(g, percentile=50)
    assert sum(scores.values()) == pytest.approx(1.0)
    top = max(scores, key=scores.get)
    assert (top, "pagerank_top5") in flags
    assert all(v["signal_value"] == 1 for v in flags.values())

    zero = nx.DiGraph()
    add_edge(zero, A, B, 0, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    add_edge(zero, B, A, 0, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    _, zero_scores = compute_pagerank(zero, percentile=95)
    assert sum(zero_scores.values()) == pytest.approx(1.0)
    assert all(value > 0 for value in zero_scores.values())


def test_dbscan_flags_outlier() -> None:
    addresses = [addr(i) for i in range(1, 7)]
    features = pd.DataFrame({
        "address": addresses,
        "tx_count_total": [5] * 6,
        "a_scaled": [0.0, 0.01, -0.01, 0.02, 0.0, 10.0],
        "b_scaled": [0.0, 0.01, 0.0, -0.01, 0.02, 10.0],
    })
    mapping, summary = run_dbscan(features, eps=0.5, min_samples=2, min_tx_total=3)
    assert summary["eligible"] == 6
    assert summary["clusters"] == 1
    assert summary["noise"] == 1
    assert mapping[addresses[-1]] == -1
    assert all(mapping[a] == 0 for a in addresses[:-1])


def _apriori_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], str, str]:
    rows = []
    labels = []
    splits = []
    for i in range(10):
        address = addr(100 + i)
        rows.append({
            "address": address, "unique_counterparties": 30, "tx_count_total": 30,
            "median_value_eth": 0.01, "total_value_eth": 1.0,
            "round_number_ratio": 0.9, "is_contract": False,
            "first_seen": pd.Timestamp("2023-12-31T00:00:00Z"),
            "last_seen": pd.Timestamp("2024-01-01T00:00:00Z"),
        })
        labels.append({"address": address, "label": "fraud", "source": "test"})
        splits.append({"address": address, "split": "train"})
    for i in range(10):
        address = addr(200 + i)
        rows.append({
            "address": address, "unique_counterparties": 1, "tx_count_total": 30,
            "median_value_eth": 1.0, "total_value_eth": 100.0,
            "round_number_ratio": 0.1, "is_contract": False,
            "first_seen": pd.Timestamp("2023-12-31T00:00:00Z"),
            "last_seen": pd.Timestamp("2024-01-01T00:00:00Z"),
        })
        labels.append({"address": address, "label": "negative", "source": "test"})
        splits.append({"address": address, "split": "train"})

    test_positive, test_negative = addr(300), addr(301)
    for address, unique, label in ((test_positive, 30, "fraud"), (test_negative, 1, "negative")):
        rows.append({
            "address": address, "unique_counterparties": unique, "tx_count_total": 30,
            "median_value_eth": 0.01 if label == "fraud" else 1.0,
            "total_value_eth": 1.0 if label == "fraud" else 100.0,
            "round_number_ratio": 0.9 if label == "fraud" else 0.1,
            "is_contract": False,
            "first_seen": pd.Timestamp("2023-12-31T00:00:00Z"),
            "last_seen": pd.Timestamp("2024-01-01T00:00:00Z"),
        })
        labels.append({"address": address, "label": label, "source": "test"})
        splits.append({"address": address, "split": "test"})

    features = pd.DataFrame(rows)
    label_frame = pd.DataFrame(labels)
    split_frame = pd.DataFrame(splits)
    dbscan = {addr(100): -1}
    return features, label_frame, split_frame, dbscan, test_positive, test_negative


def test_apriori_uses_dbscan_noise_and_is_leakage_safe() -> None:
    features, labels, splits, dbscan, test_positive, test_negative = _apriori_features()
    assert "dbscan_noise" in APRIORI_ITEMS

    train_positive = set(labels.loc[labels["label"] == "fraud", "address"]) & set(
        splits.loc[splits["split"] == "train", "address"]
    )
    test_addresses = set(splits.loc[splits["split"] == "test", "address"])
    table = _boolean_table(
        features, {}, dbscan, train_positive,
        exclude_test=True, test_addresses=test_addresses,
    )
    assert addr(100) in set(table["address"])
    assert test_positive not in set(table["address"])
    assert bool(table.loc[table["address"] == addr(100), "dbscan_noise"].iloc[0]) is True
    assert bool(table.loc[table["address"] == addr(200), "dbscan_noise"].iloc[0]) is False

    flags, rules, status = mine_apriori(
        features, {}, dbscan, labels, splits,
        min_support=0.1, min_lift=1.5, min_confidence=0.5, max_antecedent_size=2,
    )
    assert status and "no " not in status.lower()
    assert rules
    assert all(len(rule["antecedents"]) <= 2 for rule in rules)
    assert all(rule["confidence"] >= 0.5 for rule in rules)
    flagged = {address for address, signal in flags if signal == "apriori_match"}
    assert addr(100) in flagged
    assert addr(200) not in flagged
    assert test_positive in flagged  # behaviour match, not label leakage
    assert test_negative not in flagged


def test_louvain_finds_two_communities() -> None:
    g = nx.DiGraph()
    triangles = [(addr(1), addr(2), addr(3)), (addr(4), addr(5), addr(6))]
    for tri in triangles:
        for i in range(3):
            u, v = tri[i], tri[(i + 1) % 3]
            add_edge(g, u, v, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
            add_edge(g, v, u, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    flags = detect_louvain(g)
    assert len(flags) == 6
    community_ids = {value["signal_value"] for value in flags.values()}
    assert len(community_ids) == 2
    sizes = {
        community_id: sum(1 for value in flags.values() if value["signal_value"] == community_id)
        for community_id in community_ids
    }
    assert sorted(sizes.values()) == [3, 3]


def test_sanctioned_one_hop_is_train_only_and_directional() -> None:
    g = nx.DiGraph()
    S, A, B, C, T, D = addr(1), addr(2), addr(3), addr(4), addr(5), addr(6)
    for u, v in ((S, A), (B, S), (A, C), (T, D)):
        add_edge(g, u, v, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z")
    labels = pd.DataFrame({
        "address": [S, T], "label": ["sanctioned", "sanctioned"],
    })
    splits = pd.DataFrame({
        "address": [S, T, A, B, C, D],
        "split": ["train", "test", "train", "train", "train", "train"],
    })
    flags = detect_sanctioned_1hop(g, labels, splits)
    assert set(address for address, signal in flags if signal == "sanctioned_1hop") == {A, B}
    assert flags[(A, "sanctioned_1hop")]["evidence"]["direction"] == "out"
    assert flags[(B, "sanctioned_1hop")]["evidence"]["direction"] == "in"
    assert (C, "sanctioned_1hop") not in flags
    assert (D, "sanctioned_1hop") not in flags
    assert (S, "sanctioned_1hop") not in flags
