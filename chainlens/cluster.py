"""DBSCAN clustering over dynamically discovered scaled address features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sqlalchemy import text
from sqlalchemy.engine import Engine

from chainlens.detection import CLUSTER_SIGNALS, load_detection_config, write_flags
from chainlens.features import get_scaled_feature_columns


def run_cluster(engine: Engine) -> dict:
    config = load_detection_config()["dbscan"]
    eps = float(config["eps"])
    min_samples = int(config["min_samples"])
    eligibility_min_tx = int(config.get("eligibility_min_tx", 3))
    if eps <= 0:
        raise RuntimeError(f"dbscan.eps must be positive, got {eps}")
    if min_samples < 2:
        raise RuntimeError(f"dbscan.min_samples must be >= 2, got {min_samples}")

    columns, scaler_version = get_scaled_feature_columns(engine)
    query = (
        "SELECT address, "
        + ", ".join(columns)
        + " FROM address_features WHERE tx_count_total >= :min_tx "
        "ORDER BY address"
    )
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(query), conn, params={"min_tx": eligibility_min_tx}
        )
    if frame.empty:
        raise RuntimeError("No addresses eligible for DBSCAN")

    matrix = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "DBSCAN feature matrix contains NULL/non-finite values for "
            f"scaler_version={scaler_version}; re-run `make features`"
        )

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
    cluster_count = len(set(labels.tolist()) - {-1})
    noise_count = int((labels == -1).sum())

    rows: list[dict] = []
    for address, label in zip(frame["address"], labels, strict=True):
        rows.append(
            {
                "address": address,
                "signal_name": "dbscan_cluster",
                "signal_value": int(label),
                "evidence": {
                    "eps": eps,
                    "min_samples": min_samples,
                    "scaler_version": scaler_version,
                },
            }
        )
        if int(label) == -1:
            rows.append(
                {
                    "address": address,
                    "signal_name": "dbscan_noise",
                    "signal_value": 1,
                    "evidence": {"reason": "noise"},
                }
            )

    written = write_flags(engine, rows, reset_signals=CLUSTER_SIGNALS)
    print(
        f"DBSCAN: {len(frame)} eligible, {cluster_count} clusters, "
        f"{noise_count} noise, eps={eps}, min_samples={min_samples}"
    )
    print(f"DBSCAN features ({scaler_version}): {', '.join(columns)}")
    return {
        "eligible": len(frame),
        "clusters": cluster_count,
        "noise": noise_count,
        "eps": eps,
        "min_samples": min_samples,
        "scaler_version": scaler_version,
        "features": columns,
        "flags_written": written,
    }
