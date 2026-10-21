from __future__ import annotations

import pytest

from chainlens.ablation import (
    assert_baseline_restored,
    baseline_path,
    load_baseline,
    run_ablation,
)
from chainlens.manifest import sha256_file
from chainlens.scoring import RULES_PATH


@pytest.fixture(scope="module")
def baseline():
    try:
        path = baseline_path()
    except FileNotFoundError as exc:
        pytest.skip(f"{exc}; run `make baseline` first.")
    return load_baseline(path)


@pytest.fixture(scope="module")
def ablation(baseline, pg_engine):
    rules_hash_before = sha256_file(RULES_PATH)
    report, details = run_ablation(engine=pg_engine, baseline=baseline)
    assert sha256_file(RULES_PATH) == rules_hash_before
    return report, details


def test_all_rules_are_isolated(ablation):
    report, _ = ablation
    testable = report[report["testable"]]
    assert set(testable["rule_id"]) == {"R1", "R2", "R3", "R4"}
    assert testable["other_wallets_changed"].eq(False).all()
    assert testable["isolated"].all()


def test_each_primary_target_loses_exact_weight(ablation):
    report, _ = ablation
    for row in report[report["testable"]].itertuples(index=False):
        assert row.raw_delta == row.expected_weight, row.rule_id
        assert row.score_delta == row.expected_weight, row.rule_id


def test_exchange_hot_r2_arithmetic(ablation):
    _, details = ablation
    exchange = details["exchange_hot_r2"]
    assert exchange["baseline_raw"] == 35
    assert exchange["ablated_raw"] == 15
    assert exchange["baseline_adjusted"] == 10.5
    assert exchange["ablated_adjusted"] == 4.5
    assert exchange["baseline_score"] == 10
    assert exchange["ablated_score"] == 4
    assert exchange["score_delta"] == 6


def test_r5_is_documented_untestable(ablation):
    report, _ = ablation
    r5 = report[report["rule_id"] == "R5"].iloc[0]
    assert not bool(r5["testable"])
    assert not bool(r5["other_wallets_changed"])
    assert "Apriori" in r5["note"]


@pytest.mark.skip(
    reason="no Apriori rules on fixture; covered by tests/test_apriori.py"
)
def test_r5_ablation_target():
    """The Phase 6 target table explicitly marks R5 untestable on fixture."""


def test_rules_yaml_and_scores_restored(baseline, pg_engine):
    # `ablation` has already run in this module; this test states the invariant
    # independently so a failure points at restore, not at an individual rule.
    assert_baseline_restored(engine=pg_engine, baseline=baseline)
