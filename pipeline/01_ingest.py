#!/usr/bin/env python3
"""Phase 2 label, known-address and leakage-safe split stages.

Usage::

    python pipeline/01_ingest.py labels
    python pipeline/01_ingest.py known
    python pipeline/01_ingest.py split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chainlens.config import is_sample_mode, load_data_config, project_path  # noqa: E402
from chainlens.db import apply_schema, get_pg_engine  # noqa: E402
from chainlens.fixtures import FIXTURE_KNOWN, address_for  # noqa: E402
from chainlens.labels import (  # noqa: E402
    ADDRESS_RE,
    load_label_frames,
    normalise_address,
    source_metadata,
)
from chainlens.manifest import record_file  # noqa: E402

SEED = 42
TEST_SIZE = 0.3


def run_labels() -> pd.DataFrame:
    labels, report = load_label_frames()
    engine = get_pg_engine()
    apply_schema(engine)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM labels"))
        conn.execute(
            text(
                """
                INSERT INTO labels (address, label, source, confidence, as_of)
                VALUES (:address, :label, :source, :confidence, :as_of)
                """
            ),
            labels.to_dict("records"),
        )

    labels_path = project_path(load_data_config()["ingest"]["labels_parquet"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path, index=False)
    record_file(
        engine,
        labels_path,
        file_path="data/processed/labels_snapshot.parquet",
        row_count=len(labels),
        metadata={"stage": "labels", "sources": sorted(labels["source"].unique().tolist())},
    )

    meta = source_metadata()["sources"]
    for source, info in report.items():
        engine_meta = {
            "stage": "labels_source",
            "source": source,
            "as_of": meta[source].get("snapshot_date")
            or meta[source].get("publication_date"),
            "url": meta[source].get("url"),
            "note": meta[source].get("note"),
        }
        record_file(
            engine,
            ROOT / info["file"],
            file_path=info["file"],
            row_count=info["rows"],
            metadata=engine_meta,
        )

    counts = (
        labels.groupby(["source", "label"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["source", "label"])
    )
    print("Label source x label counts")
    print(counts.to_string(index=False))
    print(f"Total label rows: {len(labels):,}")
    return labels


def transaction_source() -> Path:
    config = load_data_config()
    raw = project_path(config["ingest"]["raw_transactions"])
    if raw.exists():
        return raw
    if is_sample_mode():
        sample = project_path(config["paths"]["sample_parquet"])
        if not sample.exists():
            raise FileNotFoundError(f"Missing sample Parquet: {sample}")
        print(f"Sample mode: reading {sample} instead of {raw}")
        return sample
    raise FileNotFoundError(
        f"Missing {raw}. Run the BigQuery export first, or set "
        "CHAINLENS_SAMPLE_MODE=1 for the synthetic fixture path."
    )


def load_contract_addresses(tx_path: Path) -> tuple[set[str], set[str]]:
    config = load_data_config()
    con = duckdb.connect()
    try:
        eoa_rows = con.execute(
            "SELECT DISTINCT lower(from_address) FROM read_parquet(?) WHERE from_address IS NOT NULL",
            [str(tx_path)],
        ).fetchall()
        eoas = {row[0] for row in eoa_rows}

        contracts_path = project_path(config["ingest"]["raw_contracts"])
        if contracts_path.exists():
            contract_rows = con.execute(
                "SELECT DISTINCT lower(address) FROM read_parquet(?) WHERE address IS NOT NULL",
                [str(contracts_path)],
            ).fetchall()
        else:
            contract_rows = con.execute(
                """
                SELECT DISTINCT lower(receipt_contract_address)
                FROM read_parquet(?)
                WHERE receipt_contract_address IS NOT NULL
                """,
                [str(tx_path)],
            ).fetchall()
        contracts = {row[0] for row in contract_rows if row[0]}
    finally:
        con.close()
    return eoas, contracts


def run_known() -> pd.DataFrame:
    tx_path = transaction_source()
    eoas, contract_candidates = load_contract_addresses(tx_path)
    contracts = {
        address for address in contract_candidates if address not in eoas and ADDRESS_RE.fullmatch(address)
    }

    seed_path = project_path(load_data_config()["ingest"]["known_seed"])
    seed = json.loads(seed_path.read_text(encoding="utf-8"))["addresses"]

    records: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for item in seed:
        address = normalise_address(item["address"])
        if not address:
            raise RuntimeError(f"Invalid seed address: {item['address']!r}")
        if address not in seen:
            records.append(
                {
                    "address": address,
                    "category": item["category"],
                    "note": item.get("note", "seed, not exhaustive"),
                }
            )
            seen.add(address)
    if is_sample_mode():
        for name, (category, note) in FIXTURE_KNOWN.items():
            address = address_for(name)
            if address not in seen:
                records.append(
                    {
                        "address": address,
                        "category": category,
                        "note": note,
                    }
                )
                seen.add(address)
    for address in sorted(contracts):
        if address not in seen:
            records.append(
                {
                    "address": address,
                    "category": "contract",
                    "note": "detected from contracts.parquet / receipt_contract_address",
                }
            )
            seen.add(address)

    engine = get_pg_engine()
    apply_schema(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM known_addresses"))
        conn.execute(
            text(
                "INSERT INTO known_addresses (address, category, note) "
                "VALUES (:address, :category, :note)"
            ),
            records,
        )

    df = pd.DataFrame.from_records(records)
    summary = (
        df.groupby("category", as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("category")
    )
    print("known_addresses categories")
    print(summary.to_string(index=False))
    if not ((df["category"] == "exchange").any() and (df["category"] == "mixer").any()):
        raise RuntimeError("known_addresses gate failed: need >=1 exchange and >=1 mixer")

    summary_path = ROOT / "data" / "processed" / "known_addresses_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "rows": len(df),
                "categories": dict(zip(summary["category"], summary["rows"], strict=True)),
                "seed_path": str(seed_path.relative_to(ROOT)),
                "seed_note": "seed, not exhaustive",
                "sample_mode": is_sample_mode(),
                "fixture_known_rows": len(FIXTURE_KNOWN) if is_sample_mode() else 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    record_file(
        engine,
        summary_path,
        file_path="data/processed/known_addresses_summary.json",
        row_count=len(df),
        metadata={"stage": "known"},
    )
    return df


def build_strata(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    priority = {"ofac": 0, "kaggle": 1, "darklist": 2}
    grouped: dict[str, set[tuple[str, str]]] = {}
    for row in labels.itertuples(index=False):
        grouped.setdefault(row.address, set()).add((row.source, row.label))
    rows = []
    for address, pairs in grouped.items():
        selected = sorted(pairs, key=lambda pair: (priority.get(pair[0], 99), pair))
        source, label = selected[0]
        rows.append(
            {
                "address": address,
                "stratum": f"{source}:{label}",
                "sources": sorted({s for s, _ in pairs}),
            }
        )
    frame = pd.DataFrame.from_records(rows).sort_values("address").reset_index(drop=True)
    return frame, dict(frame["stratum"].value_counts())


def ensure_source_coverage(
    train: list[str],
    test: list[str],
    by_address: dict[str, set[str]],
) -> list[str]:
    """Move train addresses into test until every >=2-positive source is covered."""
    totals: dict[str, int] = {}
    for sources in by_address.values():
        for source in sources:
            totals[source] = totals.get(source, 0) + 1
    test_sources = {source for address in test for source in by_address[address]}

    for source in sorted(totals):
        if totals[source] < 2 or source in test_sources:
            continue
        candidates = [
            address
            for address in train
            if source in by_address[address]
        ]
        if not candidates:
            raise RuntimeError(f"Cannot cover source {source!r} in test split")
        candidates.sort(key=lambda address: (len(by_address[address]), address))
        moved = candidates[0]
        train.remove(moved)
        test.append(moved)
        test_sources.update(by_address[moved])
        print(f"Coverage adjustment: moved {moved} to test for source {source}")
    return train


def run_split() -> dict:
    labels = pd.read_parquet(project_path(load_data_config()["ingest"]["labels_parquet"]))
    frame, strata_counts = build_strata(labels)
    singleton_strata = {name for name, count in strata_counts.items() if count < 2}

    common = frame[~frame["stratum"].isin(singleton_strata)].reset_index(drop=True)
    singletons = frame[frame["stratum"].isin(singleton_strata)].reset_index(drop=True)

    if len(common) >= 2:
        train, test = train_test_split(
            common["address"].tolist(),
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=common["stratum"].tolist(),
        )
    else:
        train, test = common["address"].tolist(), []

    # Singletons cannot be stratified; alternate deterministically so they are
    # represented on both sides when possible.
    for index, row in singletons.sort_values("address").reset_index(drop=True).iterrows():
        (test if index % 2 == 0 else train).append(row["address"])

    by_address: dict[str, set[str]] = {
        address: set(sources)
        for address, sources in zip(frame["address"], frame["sources"], strict=True)
    }
    train = ensure_source_coverage(train, test, by_address)

    train_set, test_set = set(train), set(test)
    if train_set & test_set:
        raise RuntimeError("Leakage gate failed: train and test overlap")
    if len(train_set) + len(test_set) != len(frame):
        raise RuntimeError("Split gate failed: some labeled addresses have no split")

    assigned = pd.DataFrame(
        [{"address": a, "split": "train", "seed": SEED} for a in sorted(train_set)]
        + [{"address": a, "split": "test", "seed": SEED} for a in sorted(test_set)]
    )

    engine = get_pg_engine()
    apply_schema(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM label_splits"))
        conn.execute(
            text("INSERT INTO label_splits (address, split, seed) VALUES (:address, :split, :seed)"),
            assigned.to_dict("records"),
        )

    merged = labels.merge(assigned, on="address", how="inner")
    summary: dict = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "total_labeled_addresses": int(frame.shape[0]),
        "train_addresses": len(train_set),
        "test_addresses": len(test_set),
        "train_test_disjoint": True,
        "by_source": {},
        "insufficient_for_evaluation": [],
    }
    for (source, label), group in merged.groupby(["source", "label"]):
        total = int(group.shape[0])
        train_count = int((group["split"] == "train").sum())
        test_count = int((group["split"] == "test").sum())
        sufficient = total >= 2 and test_count > 0
        summary["by_source"][source] = {
            "label": label,
            "total": total,
            "train": train_count,
            "test": test_count,
            "sufficient_for_evaluation": sufficient,
        }
        if total < 2:
            summary["insufficient_for_evaluation"].append(source)
        if total >= 2 and test_count == 0:
            raise RuntimeError(f"Split gate failed: no test positives for {source}")

    print("Leakage-safe split summary (addresses, not transactions)")
    print(
        f"{'source':10s} {'label':12s} {'total':>7s} {'train':>7s} {'test':>7s}"
    )
    for source, info in sorted(summary["by_source"].items()):
        print(
            f"{source:10s} {info['label']:12s} {info['total']:7d} "
            f"{info['train']:7d} {info['test']:7d}"
        )
    if summary["insufficient_for_evaluation"]:
        print(
            "Insufficient for evaluation: "
            + ", ".join(sorted(summary["insufficient_for_evaluation"]))
        )

    summary_path = project_path(load_data_config()["ingest"]["split_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    record_file(
        engine,
        summary_path,
        file_path="data/processed/split_summary.json",
        row_count=len(assigned),
        metadata={
            "stage": "split",
            "seed": SEED,
            "test_size": TEST_SIZE,
            "disjoint": True,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["labels", "known", "split"])
    args = parser.parse_args()
    if args.stage == "labels":
        run_labels()
    elif args.stage == "known":
        run_known()
    else:
        run_split()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
