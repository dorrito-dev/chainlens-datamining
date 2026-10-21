from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import text

from chainlens.db import get_pg_engine

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.getenv("CHAINLENS_CLEAN_CLONE") != "1",
    reason="manual clean-clone gate; set CHAINLENS_CLEAN_CLONE=1 to run",
)


def _load_local_env() -> None:
    """Make the host-side assertion see the same values `make` reads."""
    if os.getenv("POSTGRES_PASSWORD"):
        return
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def test_clean_clone_sample_chain():
    """Run the documented sample-mode chain against a fresh environment."""
    _load_local_env()
    env = {**os.environ, "CHAINLENS_SAMPLE_MODE": "1"}
    for command in (
        ["make", "up"],
        ["make", "sample"],
        ["make", "overlap"],
        ["make", "ingest-local"],
        ["make", "features-local"],
        ["make", "detect-local"],
        ["make", "score-local"],
        ["make", "baseline-local"],
        ["make", "ablate-local"],
    ):
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    engine = get_pg_engine()
    with engine.connect() as conn:
        feature_count = int(
            conn.execute(text("SELECT count(*) FROM address_features")).scalar_one()
        )
        score_count = int(
            conn.execute(text("SELECT count(*) FROM risk_scores")).scalar_one()
        )
        unscored = int(
            conn.execute(
                text("SELECT count(*) FROM risk_scores WHERE tier = 'unscored'")
            ).scalar_one()
        )
    assert feature_count > 0
    assert score_count == feature_count
    assert unscored == 0
