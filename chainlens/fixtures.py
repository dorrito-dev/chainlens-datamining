"""Stable fixture identities shared by sample generation and Phase 3 backfill.

The synthetic Parquet addresses are deterministic hashes of these names. Keeping
the constants here (rather than duplicating addresses in Phase 3) prevents the
fixture labels from drifting away from the transactions they describe.
"""

from __future__ import annotations

import hashlib

SEED = 42

# Only the three labels Phase 3 must restore for end-to-end R3/R5 exercises.
FIXTURE_LABELS: dict[str, tuple[str, str, float]] = {
    "SANCTIONED_SEED": ("sanctioned", "fixture", 1.0),
    "KAGGLE_FRAUD": ("fraud", "fixture", 0.9),
    "DARKLIST_EXTRA": ("darklist", "fixture", 0.8),
}

# Fixed placements required by the Phase 3 regression gate. They mirror Phase 1.
FIXTURE_SPLITS: dict[str, str] = {
    "SANCTIONED_SEED": "train",
    "DARKLIST_EXTRA": "train",
    "KAGGLE_FRAUD": "test",
}

# Phase 5 fixture scoring depends on EXCHANGE_HOT being a known exchange.
# Phase 2's real seed loader replaces the whole table, so sample mode must
# re-add these synthetic identities after loading the real curated seed.
FIXTURE_KNOWN: dict[str, tuple[str, str]] = {
    "EXCHANGE_HOT": ("exchange", "Synthetic exchange hot wallet for down-weighting."),
    "CONTRACT_ADDR": ("contract", "Synthetic contract address."),
    "MIXER_SOURCE": ("mixer", "Synthetic mixer-tagged source."),
}


def address_for(name: str) -> str:
    """Return the deterministic lowercase address used by the fixture builder."""
    digest = hashlib.sha256(f"chainlens:{SEED}:{name}".encode()).hexdigest()
    return "0x" + digest[:40]
