from __future__ import annotations

import json

import pandas as pd

from chainlens.apriori import run_apriori
from chainlens.cluster import run_cluster
from chainlens.detection import ALLOWED_SIGNALS
from chainlens.graph import run_graph
from chainlens.seed_scores import seed_scores

REQUIRED_EVIDENCE_FIELDS = {
    "dbscan_cluster": {"eps", "min_samples", "scaler_version"},
    "dbscan_noise": {"reason"},
    "pagerank": {"pagerank"},
    "pagerank_top5": {"pagerank", "threshold"},
    "cycle_member": {"cycle_path", "edge_values", "time_span_hours", "scc_size"},
    "fanout": {"unique_counterparties", "tx_count_out", "fanout_ratio"},
    "sanctioned_1hop": {"seed", "direction", "seed_source"},
    "louvain_community": {"community_size"},
    "apriori_match": {"antecedent", "support", "confidence", "lift"},
}


def _snapshot(pg_engine) -> list[tuple]:
    frame = pd.read_sql(
        "SELECT address, signal_name, signal_value::text AS signal_value, "
        "evidence::text AS evidence FROM address_flags",
        pg_engine,
    )
    return list(
        frame.sort_values(["address", "signal_name"]).itertuples(
            index=False, name=None
        )
    )


def test_only_allowed_signal_names_are_written(pg_engine):
    names = set(
        pd.read_sql("SELECT DISTINCT signal_name FROM address_flags", pg_engine)[
            "signal_name"
        ]
    )
    assert names <= ALLOWED_SIGNALS


def test_every_evidence_document_is_json_with_required_fields(pg_engine):
    frame = pd.read_sql(
        "SELECT signal_name, evidence::text AS evidence FROM address_flags",
        pg_engine,
    )
    assert not frame.empty
    for row in frame.itertuples(index=False):
        evidence = json.loads(row.evidence)
        assert isinstance(evidence, dict)
        required = REQUIRED_EVIDENCE_FIELDS[row.signal_name]
        assert required <= set(evidence), (row.signal_name, evidence)


def test_detect_stages_are_idempotent(pg_engine):
    before = _snapshot(pg_engine)
    run_cluster(pg_engine)
    run_graph(pg_engine)
    run_apriori(pg_engine)
    seed_scores(pg_engine)
    after = _snapshot(pg_engine)
    assert after == before
