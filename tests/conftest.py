"""Shared pytest fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the sample pipeline once and share its output directory."""
    output = tmp_path_factory.mktemp("chainlens_sample_output")
    env = os.environ.copy()
    env["CHAINLENS_OUTPUT_DIR"] = str(output)
    result = subprocess.run(
        [sys.executable, "run.py", "--mode", "sample", "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"sample pipeline failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return output
