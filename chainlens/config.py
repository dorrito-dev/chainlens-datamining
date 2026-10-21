"""Configuration and path helpers shared by Phase 2 pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"


def load_data_config(path: Path | str = DATA_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_sample_mode() -> bool:
    """True when the fixture Parquet substitute is explicitly enabled.

    Real runs must never silently fall back to synthetic data, so this is
    opt-in via CHAINLENS_SAMPLE_MODE=1 (CI and local verification) or the
    config flag.
    """
    env_value = os.getenv("CHAINLENS_SAMPLE_MODE", "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    config = load_data_config()
    return bool(config.get("ingest", {}).get("sample_mode", False))


def get_window(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_data_config()
    window = config["window"]
    if window.get("block_start") is None or window.get("block_end") is None:
        raise RuntimeError(
            "window.block_start / window.block_end are not set in config/data.yaml"
        )
    return window
