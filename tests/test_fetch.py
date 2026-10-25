"""Ingestion: normalisation, cache/resume, labels, split, context, HTTP client."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import fetch_data as fetch


def addr(n: int) -> str:
    return f"0x{n:040x}"


def raw_row(tx_hash: str = "0x1", frm: str | None = None, to: str | None = None) -> dict:
    return {
        "hash": tx_hash,
        "blockNumber": "1",
        "timeStamp": "1704067200",
        "from": frm or addr(1).upper().replace("0X", "0x"),
        "to": to if to is not None else addr(2),
        "value": "1000000000000000000",
        "gas": "21000",
        "gasPrice": "1000000000",
        "txreceipt_status": "1",
        "contractAddress": "",
        "isError": "0",
    }


class FakeClient:
    chain_id = 1

    def __init__(self, rows: list[dict] | None = None, error: Exception | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.error = error
        self.calls = 0

    def txlist(self, address: str, **kwargs) -> list[dict]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.rows


class FakeBlockClient:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def block_transactions(self, block: int) -> list[dict]:
        self.calls.append(block)
        return [
            {"from": addr(10), "to": addr(11)},
            {"from": addr(12), "to": None},
        ]


def test_normalise_tx_and_address() -> None:
    row = fetch.normalise_tx(raw_row())
    assert row["tx_hash"] == "0x1"
    assert row["from_address"] == addr(1)
    assert row["to_address"] == addr(2)
    assert row["value_wei"] == "1000000000000000000"
    assert row["block_timestamp"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert row["source"] == "etherscan"
    assert fetch.normalise_address(addr(1).upper().replace("0X", "0x")) == addr(1)
    assert fetch.normalise_address("0x123") is None
    assert fetch.normalise_address(None) is None
    assert fetch.normalise_tx(raw_row(to=""))["to_address"] is None


def test_cache_resume_refresh_and_error_recovery(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    client = FakeClient([raw_row()])
    frame, meta, hit = fetch.fetch_address(client, addr(1), cache)
    assert hit is False
    assert client.calls == 1
    assert len(frame) == 1
    assert meta["complete"] is True and meta["rows"] == 1
    assert (cache / f"{addr(1)}.parquet").exists()
    assert (cache / f"{addr(1)}.meta.json").exists()

    frame2, meta2, hit2 = fetch.fetch_address(client, addr(1), cache)
    assert hit2 is True
    assert client.calls == 1
    pd.testing.assert_frame_equal(frame, frame2)

    _, _, hit3 = fetch.fetch_address(client, addr(1), cache, refresh=True)
    assert hit3 is False
    assert client.calls == 2

    empty_client = FakeClient([])
    empty_frame, empty_meta, _ = fetch.fetch_address(empty_client, addr(2), cache)
    assert empty_frame.empty
    assert list(empty_frame.columns) == fetch.RAW_COLUMNS
    assert empty_meta["rows"] == 0 and empty_meta["complete"] is True

    bad = FakeClient(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        fetch.fetch_address(bad, addr(3), cache)
    bad_meta = json.loads((cache / f"{addr(3)}.meta.json").read_text())
    assert bad_meta["complete"] is False
    good = FakeClient([raw_row("0x3", frm=addr(3), to=addr(4))])
    frame3, _, hit4 = fetch.fetch_address(good, addr(3), cache)
    assert hit4 is False
    assert len(frame3) == 1


def test_fetch_dataset_redacts_api_key(tmp_path: Path, capsys) -> None:
    class ErrorClient(FakeClient):
        api_key = "SECRET_TEST_KEY"

    client = ErrorClient(error=RuntimeError("upstream failed for SECRET_TEST_KEY"))
    frame, stats = fetch.fetch_dataset([addr(1)], client, tmp_path / "cache")
    captured = capsys.readouterr().out
    assert frame.empty
    assert stats.failed == 1
    assert "SECRET_TEST_KEY" not in captured
    assert "***" in captured


def test_combine_cache_deduplicates_filters_and_stats(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fetch.fetch_address(FakeClient([raw_row("0x1")]), addr(1), cache)
    fetch.fetch_address(
        FakeClient([raw_row("0x2", frm=addr(2), to=addr(3))]), addr(2), cache
    )
    combined = fetch.combine_cache(cache)
    assert len(combined) == 2
    only_first = fetch.combine_cache(cache, addresses=[addr(1)])
    assert len(only_first) == 1
    assert only_first.iloc[0]["tx_hash"] == "0x1"
    assert fetch.cache_stats(cache) == {"cached_addresses": 2, "cache_files": 2}


def test_load_labels_positives_negatives_and_conflict(tmp_path: Path) -> None:
    A, B, C, D, E, F, N1 = (addr(i) for i in range(1, 8))
    (tmp_path / "ofac_eth.txt").write_text(f"{A}\n# comment\n{N1}\n", encoding="utf-8")
    (tmp_path / "darklist.json").write_text(json.dumps([{"address": B}]), encoding="utf-8")
    pd.DataFrame({
        "address": [C, D, E, F, N1],
        "flag": [1, 0, 0, 0, 0],
    }).to_csv(tmp_path / "kaggle_fraud.csv", index=False)

    labels = fetch.load_labels(tmp_path, negative_count=2, seed=42)
    positives = labels[labels["label"] != "negative"]
    negatives = labels[labels["label"] == "negative"]
    assert set(positives["address"]) == {A, B, C, N1}
    assert len(negatives) == 2
    assert not set(negatives["address"]) & set(positives["address"])
    assert labels["stratum"].notna().all()
    assert len(labels[labels["address"] == N1]) == 1
    assert labels.loc[labels["address"] == N1, "label"].iloc[0] == "sanctioned"

    all_negatives = fetch.load_labels(tmp_path, negative_count=0, seed=42)
    assert len(all_negatives[all_negatives["label"] == "negative"]) == 3


def test_split_is_deterministic_and_leakage_safe() -> None:
    positives = [{"address": addr(i), "label": "fraud", "source": "test"} for i in range(1, 21)]
    negatives = [{"address": addr(100 + i), "label": "negative", "source": "test"} for i in range(20)]
    labels = pd.DataFrame(positives + negatives)

    first = fetch.split_addresses(labels, seed=42, test_size=0.3)
    second = fetch.split_addresses(labels, seed=42, test_size=0.3)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 40
    assert set(first["split"]) == {"train", "test"}
    assert len(first[first["split"] == "test"]) == 12
    assert first["address"].is_unique

    test_labels = set(first.loc[first["split"] == "test", "address"])
    assert test_labels & set(labels.loc[labels["label"] == "fraud", "address"])
    assert test_labels & set(labels.loc[labels["label"] == "negative", "address"])

    tiny = fetch.split_addresses(labels.head(2), seed=42)
    assert set(tiny["split"]) == {"train"}


def test_build_plan_caps_deterministically() -> None:
    labels = pd.DataFrame(
        [{"address": addr(i), "label": "fraud"} for i in range(1, 11)]
        + [{"address": addr(100 + i), "label": "negative"} for i in range(10)]
    )
    plan = fetch.build_address_plan(labels, max_addresses=6, seed=42)
    again = fetch.build_address_plan(labels, max_addresses=6, seed=42)
    assert plan == again
    assert len(plan["positives"]) == 3
    assert len(plan["negatives"]) == 3
    assert set(plan["positives"]) <= set(labels.loc[labels["label"] == "fraud", "address"])
    assert set(plan["negatives"]) <= set(labels.loc[labels["label"] == "negative", "address"])

    full = fetch.build_address_plan(labels, max_addresses=None)
    assert len(full["positives"]) == 10 and len(full["negatives"]) == 10


def test_context_sampling_excludes_and_is_deterministic() -> None:
    client = FakeBlockClient()
    chosen, blocks = fetch.sample_context_addresses(
        client, exclude={addr(10)}, block_start=100, block_end=110,
        n_blocks=3, max_addresses=2, seed=42,
    )
    assert len(blocks) == 3
    assert blocks == sorted(blocks)
    assert set(chosen) <= {addr(11), addr(12)}
    assert addr(10) not in chosen
    assert len(chosen) <= 2
    chosen2, blocks2 = fetch.sample_context_addresses(
        client, exclude={addr(10)}, block_start=100, block_end=110,
        n_blocks=3, max_addresses=2, seed=42,
    )
    assert (chosen, blocks) == (chosen2, blocks2)
    assert fetch.sample_context_addresses(
        client, exclude=set(), block_start=110, block_end=100,
    ) == ([], [])


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.status_code = status
        self.calls = 0

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> dict:
        return self.payload


def test_http_client_success_empty_invalid_key_and_rate_limit(monkeypatch) -> None:
    client = fetch.EtherscanClient("test-key", calls_per_second=1000)
    payload = {"status": "1", "message": "OK", "result": [raw_row()]}
    client.session.get = lambda *args, **kwargs: FakeResponse(payload)
    rows = client.txlist(addr(1))
    assert len(rows) == 1

    client.session.get = lambda *args, **kwargs: FakeResponse(
        {"status": "0", "message": "No transactions found", "result": []}
    )
    assert client.txlist(addr(1)) == []

    client.session.get = lambda *args, **kwargs: FakeResponse(
        {"status": "0", "message": "NOTOK", "result": "Invalid API Key"}
    )
    with pytest.raises(RuntimeError, match="ETHERSCAN_API_KEY"):
        client.txlist(addr(1))

    responses = [
        FakeResponse({"status": "0", "message": "NOTOK", "result": "Max rate limit reached"}),
        FakeResponse(payload),
    ]
    calls = {"n": 0}

    def get(*args, **kwargs):
        calls["n"] += 1
        return responses[min(calls["n"] - 1, 1)]

    monkeypatch.setattr("time.sleep", lambda *_: None)
    client.session.get = get
    assert len(client.txlist(addr(1))) == 1
    assert calls["n"] == 2


def test_api_key_loader(monkeypatch) -> None:
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    assert fetch.load_api_key(required=False) is None
    with pytest.raises(SystemExit):
        fetch.load_api_key(required=True)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "secret")
    assert fetch.load_api_key() == "secret"
