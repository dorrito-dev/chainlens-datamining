from __future__ import annotations

import pandas as pd

from chainlens.config import load_data_config, project_path
from chainlens.features import build_raw_features


def test_build_raw_features_is_deterministic():
    clean_path = project_path(load_data_config()["ingest"]["clean_parquet"])
    first = build_raw_features(str(clean_path))
    second = build_raw_features(str(clean_path))
    pd.testing.assert_frame_equal(
        first.sort_values("address").reset_index(drop=True),
        second.sort_values("address").reset_index(drop=True),
    )
