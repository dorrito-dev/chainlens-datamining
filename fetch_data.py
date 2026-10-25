"""Stage 1: ingestion.

Etherscan V2 client, per-address Parquet cache with resume, label loading
(OFAC / darklist / Kaggle positives and Kaggle flag=0 negatives), deterministic
address split, and context sampling from random blocks.

Secrets are read from ETHERSCAN_API_KEY in the environment (loaded from .env).
The key is never written to a cache file, a log line, or run_info.json.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")
RAW_COLUMNS = [
    "tx_hash", "block_number", "block_timestamp", "from_address", "to_address",
    "value_wei", "gas", "gas_price", "receipt_status",
    "receipt_contract_address", "is_error", "source", "fetched_at",
]

load_dotenv(PROJECT_ROOT / ".env")


# ── address helpers ────────────────────────────────────────────────────────────

def normalise_address(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    addr = str(value).strip().lower()
    return addr if ADDRESS_RE.fullmatch(addr) else None


def load_api_key(required: bool = True) -> str | None:
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not key:
        if required:
            raise SystemExit(
                "ETHERSCAN_API_KEY is missing. Copy .env.example to .env and add "
                "your free key from https://etherscan.io/myapikey"
            )
        return None
    return key


# ── Etherscan client ───────────────────────────────────────────────────────────

@dataclass
class FetchStats:
    calls: int = 0
    cache_hits: int = 0
    empty: int = 0
    failed: int = 0
    addresses: int = 0
    rows: int = 0


class EtherscanClient:
    """Small, rate-limited Etherscan V2 client.

    The free tier is 3 calls/second and 100,000 calls/day. Every request needs a
    key. `txlist` returns normal transactions only.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.etherscan.io/v2/api",
        chain_id: int = 1,
        calls_per_second: float = 3.0,
        timeout: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chain_id = int(chain_id)
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = 1.0 / max(calls_per_second, 0.1)
        self._next_allowed = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ChainLens/2.0 (university project)"})

    # -- transport ------------------------------------------------------------
    def _throttle(self) -> None:
        now = time.monotonic()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
        self._next_allowed = max(time.monotonic(), self._next_allowed) + self.min_interval

    def _redacted(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: ("***" if k == "apikey" else v) for k, v in params.items()}

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        params["chainid"] = str(self.chain_id)
        params["apikey"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    raise RuntimeError("HTTP 429 rate limit")
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 - network layer is deliberately broad
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 16))
                continue

            # Proxy responses have no status field.
            if "status" not in data:
                return data

            status = str(data.get("status"))
            message = str(data.get("message", ""))
            result = data.get("result")
            result_text = str(result)
            if status == "1":
                return data

            lowered = f"{message} {result_text}".lower()
            if "no transactions found" in lowered or "no records found" in lowered:
                return data
            if "rate limit" in lowered or "max calls per sec" in lowered:
                last_error = RuntimeError(result_text or message)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 16))
                continue
            if "invalid api key" in lowered or "missing or invalid" in lowered:
                raise RuntimeError(
                    "Etherscan rejected ETHERSCAN_API_KEY. Regenerate it at "
                    "https://etherscan.io/myapikey and update .env."
                )
            raise RuntimeError(f"Etherscan error ({self._redacted(params)}): {message} {result_text}")

        # Never let a requests/HTTPError string (which can contain the full URL
        # with `apikey=...`) reach a log or the UI.
        safe_error = str(last_error).replace(self.api_key, "***") if last_error else "unknown"
        raise RuntimeError(f"Etherscan request failed after retries: {safe_error}")

    # -- endpoints ------------------------------------------------------------
    def account(self, action: str, **params: Any) -> list[dict[str, Any]]:
        data = self._call({"module": "account", "action": action, **params})
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def txlist(
        self,
        address: str,
        *,
        offset: int = 500,
        sort: str = "desc",
        page: int = 1,
        startblock: int = 0,
        endblock: int = 99_999_999,
    ) -> list[dict[str, Any]]:
        return self.account(
            "txlist",
            address=address,
            startblock=str(startblock),
            endblock=str(endblock),
            page=str(page),
            offset=str(offset),
            sort=sort,
        )

    def block_number(self) -> int:
        data = self._call({"module": "proxy", "action": "eth_blockNumber"})
        result = data.get("result")
        if result is None:
            raise RuntimeError(f"Could not read latest block: {data}")
        return int(str(result), 16) if str(result).startswith("0x") else int(result)

    def block_transactions(self, block_number: int) -> list[dict[str, Any]]:
        data = self._call({
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(int(block_number)),
            "boolean": "true",
        })
        result = data.get("result")
        if not isinstance(result, dict):
            return []
        return list(result.get("transactions") or [])

    def check_key(self) -> bool:
        data = self._call({"module": "stats", "action": "ethprice"})
        return str(data.get("status")) == "1"


# ── normalisation ──────────────────────────────────────────────────────────────

def _utc(timestamp: object) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(int(str(timestamp)), unit="s", tz="UTC")
    except (TypeError, ValueError):
        return None


def normalise_tx(raw: dict[str, Any], fetched_at: str | None = None) -> dict[str, Any]:
    """Etherscan `txlist` row -> canonical raw schema. Wei stays a string."""
    return {
        "tx_hash": str(raw.get("hash", "")).lower(),
        "block_number": int(raw.get("blockNumber", 0) or 0),
        "block_timestamp": _utc(raw.get("timeStamp")),
        "from_address": normalise_address(raw.get("from")),
        "to_address": normalise_address(raw.get("to")),
        "value_wei": str(raw.get("value", "0") or "0"),
        "gas": int(raw.get("gas", 0) or 0),
        "gas_price": int(raw.get("gasPrice", 0) or 0),
        "receipt_status": int(raw.get("txreceipt_status", 1) or 1),
        "receipt_contract_address": normalise_address(raw.get("contractAddress")),
        "is_error": int(raw.get("isError", 0) or 0),
        "source": "etherscan",
        "fetched_at": fetched_at or datetime.now(UTC).isoformat(timespec="seconds"),
    }


def empty_raw() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in RAW_COLUMNS})


# ── cache ──────────────────────────────────────────────────────────────────────

def _cache_paths(cache_dir: Path, address: str) -> tuple[Path, Path]:
    return cache_dir / f"{address}.parquet", cache_dir / f"{address}.meta.json"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def fetch_address(
    client: EtherscanClient,
    address: str,
    cache_dir: str | Path,
    *,
    offset: int = 500,
    sort: str = "desc",
    startblock: int = 0,
    endblock: int = 99_999_999,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Fetch one address, or read its cache. Returns (frame, meta, cache_hit)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, meta_path = _cache_paths(cache_dir, address)

    if parquet_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cache_matches = (
            meta.get("complete")
            and int(meta.get("offset", -1)) == int(offset)
            and int(meta.get("chain_id", -1)) == int(client.chain_id)
            and int(meta.get("startblock", -1)) == int(startblock)
            and int(meta.get("endblock", -1)) == int(endblock)
        )
        if cache_matches:
            return pd.read_parquet(parquet_path), meta, True

    try:
        rows = client.txlist(
            address, offset=offset, sort=sort,
            startblock=startblock, endblock=endblock,
        )
    except Exception as exc:  # noqa: BLE001 - cache incomplete, retry next run
        meta = {
            "address": address, "complete": False, "rows": 0,
            "error": str(exc)[:500], "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "chain_id": client.chain_id, "offset": offset, "sort": sort,
        }
        _atomic_json(meta, meta_path)
        raise

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    frame = pd.DataFrame([normalise_tx(row, fetched_at) for row in rows], columns=RAW_COLUMNS)
    if frame.empty:
        frame = empty_raw()
    else:
        frame = frame.drop_duplicates(subset=["tx_hash"]).reset_index(drop=True)

    blocks = pd.to_numeric(frame["block_number"], errors="coerce").dropna()
    meta = {
        "address": address,
        "complete": True,
        "rows": int(len(frame)),
        "truncated": bool(len(rows) >= offset),
        "fetched_at": fetched_at,
        "chain_id": client.chain_id,
        "offset": offset,
        "sort": sort,
        "startblock": int(startblock),
        "endblock": int(endblock),
        "first_block": int(blocks.min()) if not blocks.empty else None,
        "last_block": int(blocks.max()) if not blocks.empty else None,
        "api_version": "v2",
    }
    _atomic_parquet(frame, parquet_path)
    _atomic_json(meta, meta_path)
    return frame, meta, False


def combine_cache(
    cache_dir: str | Path,
    addresses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Combine cached Parquet files, optionally only for the requested addresses."""
    cache_dir = Path(cache_dir)
    if addresses is None:
        paths = sorted(cache_dir.glob("*.parquet"))
    else:
        paths = [
            cache_dir / f"{address}.parquet"
            for address in dict.fromkeys(addresses)
            if (cache_dir / f"{address}.parquet").exists()
        ]
    if not paths:
        return empty_raw()
    frames = [pd.read_parquet(p) for p in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_raw()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame.drop_duplicates(subset=["tx_hash"]).reset_index(drop=True)
    return frame.sort_values(["block_number", "tx_hash"]).reset_index(drop=True)


def cache_stats(cache_dir: str | Path) -> dict[str, int]:
    cache_dir = Path(cache_dir)
    metas = list(cache_dir.glob("*.meta.json"))
    complete = 0
    for meta_path in metas:
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("complete"):
                complete += 1
        except Exception:  # noqa: BLE001
            pass
    return {"cached_addresses": complete, "cache_files": len(metas)}


# ── labels and splits ─────────────────────────────────────────────────────────

PRIORITY = {"sanctioned": 0, "fraud": 1, "darklist": 2, "negative": 9}
LABEL_PRIORITY = {"sanctioned": 0, "fraud": 1, "darklist": 2, "negative": 9}


def _label_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["address", "label", "source", "confidence", "as_of"])
    frame = pd.DataFrame(rows)
    frame["address"] = frame["address"].map(normalise_address)
    frame = frame.dropna(subset=["address"])
    return frame.drop_duplicates(subset=["address", "source", "label"]).reset_index(drop=True)


def load_labels(
    labels_dir: str | Path = "data/labels",
    *,
    negative_count: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load real labels: OFAC + darklist + Kaggle positives, plus sampled Kaggle negatives."""
    d = Path(labels_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    sources_path = d / "SOURCES.json"
    meta = json.loads(sources_path.read_text(encoding="utf-8"))["sources"] if sources_path.exists() else {}

    rows: list[dict[str, Any]] = []

    ofac_path = d / "ofac_eth.txt"
    if ofac_path.exists():
        for line in ofac_path.read_text(encoding="utf-8").splitlines():
            addr = normalise_address(line)
            if addr:
                rows.append({
                    "address": addr, "label": "sanctioned", "source": "ofac",
                    "confidence": float(meta.get("ofac", {}).get("confidence", 1.0)),
                    "as_of": meta.get("ofac", {}).get("snapshot_date", "2026-09-30"),
                })

    dark_path = d / "darklist.json"
    if dark_path.exists():
        raw = json.loads(dark_path.read_text(encoding="utf-8"))
        for item in raw:
            addr = normalise_address(item.get("address") if isinstance(item, dict) else item)
            if addr:
                rows.append({
                    "address": addr, "label": "darklist", "source": "darklist",
                    "confidence": float(meta.get("darklist", {}).get("confidence", 0.8)),
                    "as_of": meta.get("darklist", {}).get("snapshot_date", "2020-11-18"),
                })

    kaggle_path = d / "kaggle_fraud.csv"
    negatives: list[dict[str, Any]] = []
    if kaggle_path.exists():
        raw = pd.read_csv(kaggle_path, dtype={"address": "string", "flag": "Int64"})
        for _, record in raw.iterrows():
            addr = normalise_address(record.get("address"))
            if not addr:
                continue
            flag_value = record.get("flag", 0)
            if pd.isna(flag_value):
                continue
            flag = int(flag_value)
            item = {
                "address": addr,
                "label": "fraud" if flag == 1 else "negative",
                "source": "kaggle",
                "confidence": float(meta.get("kaggle", {}).get("confidence", 0.9)) if flag == 1 else 1.0,
                "as_of": meta.get("kaggle", {}).get("publication_date", "2021-01-03"),
            }
            if flag == 1:
                rows.append(item)
            else:
                negatives.append(item)

    positives = _label_frame(rows)
    positive_set = set(positives["address"])
    negatives = [n for n in negatives if n["address"] not in positive_set]
    negatives = _label_frame(negatives)
    if not negatives.empty and negative_count and len(negatives) > negative_count:
        negatives = negatives.sample(n=negative_count, random_state=seed).reset_index(drop=True)

    frame = pd.concat([positives, negatives], ignore_index=True)
    if frame.empty:
        return pd.DataFrame(columns=["address", "label", "source", "confidence", "as_of"])
    frame["stratum"] = frame["source"].astype(str) + "_" + frame["label"].astype(str)
    return frame.sort_values(["address", "label"]).reset_index(drop=True)


def load_fixture_labels(path: str | Path = "data/sample/labels_fixture.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    frame = pd.read_csv(p, dtype={"address": "string"})
    frame["address"] = frame["address"].map(normalise_address)
    frame = frame.dropna(subset=["address"]).copy()
    if "split" not in frame.columns:
        frame["split"] = "train"
    if "seed" not in frame.columns:
        frame["seed"] = 42
    return frame.reset_index(drop=True)


def _priority_row(group: pd.DataFrame) -> pd.Series:
    return group.sort_values("label", key=lambda s: s.map(LABEL_PRIORITY).fillna(5)).iloc[0]


def split_addresses(
    labels: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.3,
) -> pd.DataFrame:
    """Leakage-safe address split. Stratifies by label group where possible."""
    if labels.empty:
        return pd.DataFrame(columns=["address", "split", "seed"])

    # One row per address, highest-priority label wins.
    frame = labels.copy()
    frame["_priority"] = frame["label"].map(LABEL_PRIORITY).fillna(5)
    unique = (
        frame.sort_values(["address", "_priority"])
        .drop_duplicates("address", keep="first")
        .reset_index(drop=True)
    )
    unique = unique[["address", "label"]].copy()
    unique["stratum"] = unique["label"].map({"negative": "negative"}).fillna("positive")

    if len(unique) < 4:
        unique["split"] = "train"
        unique["seed"] = seed
        return unique[["address", "split", "seed"]]

    # Keep the test split meaningful for both classes when possible.
    counts = unique["stratum"].value_counts()
    if (counts >= 2).all() and len(unique) >= 10:
        strat = unique["stratum"]
    else:
        strat = None

    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        unique.index, test_size=test_size, random_state=seed, stratify=strat,
    )
    unique["split"] = "train"
    unique.loc[test_idx, "split"] = "test"
    unique["seed"] = seed
    return unique[["address", "split", "seed"]].reset_index(drop=True)


def build_address_plan(
    labels: pd.DataFrame,
    *,
    max_addresses: int | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split labels into positives and negative pool, with an optional dev cap."""
    positives = labels.loc[labels["label"] != "negative", "address"].drop_duplicates().tolist()
    negatives = labels.loc[labels["label"] == "negative", "address"].drop_duplicates().tolist()
    if max_addresses:
        half = max(1, max_addresses // 2)
        rng = random.Random(seed)
        positives = rng.sample(positives, min(half, len(positives)))
        negatives = rng.sample(negatives, min(half, len(negatives)))
    return {"positives": positives, "negatives": negatives}


# ── context sampling ──────────────────────────────────────────────────────────

def sample_context_addresses(
    client: EtherscanClient,
    *,
    exclude: set[str],
    block_start: int,
    block_end: int,
    n_blocks: int = 20,
    max_addresses: int = 500,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Sample active addresses from random blocks, excluding labelled addresses."""
    rng = random.Random(seed)
    if block_end <= block_start:
        return [], []
    span = block_end - block_start
    blocks = sorted({block_start + rng.randrange(max(1, span)) for _ in range(n_blocks)})
    addresses: set[str] = set()
    for block in blocks:
        for tx in client.block_transactions(block):
            for side in ("from", "to"):
                addr = normalise_address(tx.get(side))
                if addr and addr not in exclude:
                    addresses.add(addr)
    chosen = sorted(addresses)[:max_addresses]
    return chosen, blocks


# ── data config ───────────────────────────────────────────────────────────────

def load_data_config(path: str | Path = "config/data.yaml") -> dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_known_addresses(path: str | Path = "config/known_addresses.json") -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.suffix == ".csv":
        frame = pd.read_csv(p, dtype={"address": "string"})
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload.get("addresses", payload))
    if frame.empty:
        return pd.DataFrame(columns=["address", "category", "note"])
    frame["address"] = frame["address"].map(normalise_address)
    frame["category"] = frame["category"].astype(str).str.strip().str.lower()
    if "note" not in frame.columns:
        frame["note"] = ""
    return frame.dropna(subset=["address"]).drop_duplicates("address").reset_index(drop=True)


def load_sample_transactions(path: str | Path = "data/sample/transactions.parquet") -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return pd.read_parquet(p)


def fetch_dataset(
    addresses: Iterable[str],
    client: EtherscanClient,
    cache_dir: str | Path,
    *,
    offset: int = 500,
    sort: str = "desc",
    startblock: int = 0,
    endblock: int = 99_999_999,
    refresh: bool = False,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, FetchStats]:
    addresses = list(dict.fromkeys(addresses))
    stats = FetchStats(addresses=len(addresses))
    started = time.monotonic()
    try:
        for i, address in enumerate(addresses, start=1):
            try:
                _, meta, cache_hit = fetch_address(
                    client, address, cache_dir, offset=offset, sort=sort,
                    startblock=startblock, endblock=endblock, refresh=refresh,
                )
            except Exception as exc:  # noqa: BLE001 - one bad address must not kill the run
                if "rejected ETHERSCAN_API_KEY" in str(exc):
                    raise
                stats.failed += 1
                message = str(exc).replace(getattr(client, "api_key", ""), "***")[:160]
                print(f"[fetch] {i}/{len(addresses)} failed {address}: {message}", flush=True)
                continue
            stats.calls += 0 if cache_hit else 1
            stats.cache_hits += 1 if cache_hit else 0
            stats.rows += int(meta.get("rows", 0))
            stats.empty += 1 if int(meta.get("rows", 0)) == 0 else 0
            if i % progress_every == 0 or i == len(addresses):
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(addresses) - i) / rate if rate > 0 else 0.0
                print(
                    f"[fetch] {i}/{len(addresses)} addresses • {stats.rows:,} rows • "
                    f"{stats.cache_hits} cache hits • {rate:.1f}/s • ETA {eta/60:.1f} min",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[fetch] interrupted; combining the per-address cache so far", flush=True)

    frame = combine_cache(cache_dir, addresses)
    stats.rows = len(frame)
    return frame, stats


def write_raw(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "transactions_raw.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path
