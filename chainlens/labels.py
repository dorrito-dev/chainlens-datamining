"""Label snapshot loading and validation (OFAC, darklist, Kaggle)."""

from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd

from chainlens.config import PROJECT_ROOT

ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")
LABELS_DIR = PROJECT_ROOT / "data" / "labels"


def normalise_address(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    address = str(value).strip().lower()
    return address if ADDRESS_RE.fullmatch(address) else None


def source_metadata() -> dict:
    return json.loads((LABELS_DIR / "SOURCES.json").read_text(encoding="utf-8"))


def load_label_frames() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = source_metadata()["sources"]
    frames: list[pd.DataFrame] = []
    report: dict[str, dict] = {}

    ofac_path = LABELS_DIR / "ofac_eth.txt"
    ofac_rows = []
    for line in ofac_path.read_text(encoding="utf-8").splitlines():
        address = normalise_address(line)
        if address:
            ofac_rows.append(address)
    ofac = pd.DataFrame(
        {
            "address": ofac_rows,
            "label": "sanctioned",
            "source": "ofac",
            "confidence": float(meta["ofac"]["confidence"]),
            "as_of": date.fromisoformat(meta["ofac"]["snapshot_date"]),
        }
    )
    frames.append(ofac)
    report["ofac"] = {"rows": len(ofac), "file": "data/labels/ofac_eth.txt"}

    dark_path = LABELS_DIR / "darklist.json"
    dark_raw = json.loads(dark_path.read_text(encoding="utf-8"))
    dark_rows = [a for item in dark_raw if (a := normalise_address(item.get("address")))]
    dark = pd.DataFrame(
        {
            "address": dark_rows,
            "label": "darklist",
            "source": "darklist",
            "confidence": float(meta["darklist"]["confidence"]),
            "as_of": date.fromisoformat(meta["darklist"]["snapshot_date"]),
        }
    )
    frames.append(dark)
    report["darklist"] = {"rows": len(dark), "file": "data/labels/darklist.json"}

    kaggle_path = LABELS_DIR / "kaggle_fraud.csv"
    kaggle_raw = pd.read_csv(kaggle_path, dtype={"address": "string", "flag": "Int64"})
    positives = kaggle_raw.loc[kaggle_raw["flag"] == 1, "address"]
    kaggle_rows = [a for value in positives if (a := normalise_address(value))]
    kaggle = pd.DataFrame(
        {
            "address": kaggle_rows,
            "label": "fraud",
            "source": "kaggle",
            "confidence": float(meta["kaggle"]["confidence"]),
            "as_of": date.fromisoformat(meta["kaggle"]["publication_date"]),
        }
    )
    frames.append(kaggle)
    report["kaggle"] = {"rows": len(kaggle), "file": "data/labels/kaggle_fraud.csv"}

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.drop_duplicates(subset=["address", "source", "label"]).reset_index(drop=True)
    labels["as_of"] = pd.to_datetime(labels["as_of"]).dt.date

    if (~labels["address"].str.fullmatch(ADDRESS_RE)).any():
        raise RuntimeError("Invalid address(es) after normalisation")
    if labels["as_of"].isna().any():
        raise RuntimeError("Every label must carry an as_of snapshot date")

    return labels, report
