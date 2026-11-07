"""ChainLens v2 orchestrator.

Usage:
    python run.py --mode sample
    python run.py --mode real --max-addresses 400
    python run.py --mode real --skip-fetch

Stages: fetch -> clean/features -> warehouse -> patterns -> score -> evaluate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import evaluate as evaluate_mod
import features as features_mod
import fetch_data as fetch_mod
import patterns as patterns_mod
import scoring as scoring_mod

PROJECT_ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _packages() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for name in ("pandas", "numpy", "duckdb", "networkx", "scikit-learn", "mlxtend", "streamlit"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _flatten_traces(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in traces.itertuples(index=False):
        trace = row.trace if isinstance(row.trace, dict) else json.loads(row.trace_json)
        flat = {
            "address": row.address,
            "score": trace["score"],
            "tier": trace["tier"],
            "raw_score": trace["raw_score"],
            "top_driver": trace["top_driver"],
        }
        for rule_id in scoring_mod.RULE_IDS:
            rule = trace["rules"][rule_id]
            flat[f"{rule_id}_hit"] = rule["hit"]
            flat[f"{rule_id}_contribution"] = rule["contribution"]
        flat["clamped"] = trace["clamped"]
        flat["downweighted"] = trace["downweight"]["applied"]
        rows.append(flat)
    return pd.DataFrame(rows)


def _sample_inputs(data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_dir = PROJECT_ROOT / data_cfg["paths"]["sample_dir"]
    raw = fetch_mod.load_sample_transactions(sample_dir / "transactions.parquet")
    labels = fetch_mod.load_fixture_labels(sample_dir / "labels_fixture.csv")
    known = fetch_mod.load_known_addresses(sample_dir / "known_fixture.csv")
    splits = labels[["address", "split", "seed"]].drop_duplicates("address").reset_index(drop=True)
    return raw, labels, known, splits


def _real_inputs(
    data_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir = PROJECT_ROOT / (args.output or data_cfg["paths"]["output_dir"])
    cache_dir = PROJECT_ROOT / data_cfg["paths"]["cache_dir"]
    sampling = data_cfg.get("sampling", {})
    window = data_cfg.get("window", {})
    etherscan = data_cfg.get("etherscan", {})
    seed = int(args.seed or sampling.get("seed", 42))
    negative_count = int(sampling.get("negative_count", 2000))
    max_addresses = args.max_addresses if args.max_addresses is not None else sampling.get("max_addresses")

    if args.skip_fetch:
        required = ["transactions_raw.parquet", "labels.parquet", "splits.parquet"]
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            raise SystemExit(f"--skip-fetch needs {missing} in {output_dir}")
        raw = pd.read_parquet(output_dir / "transactions_raw.parquet")
        labels = pd.read_parquet(output_dir / "labels.parquet")
        splits = pd.read_parquet(output_dir / "splits.parquet")
        known = fetch_mod.load_known_addresses(PROJECT_ROOT / "config" / "known_addresses.json")

        context_blocks: list[int] = []
        context_file = output_dir / "context_blocks.json"
        if context_file.exists():
            try:
                context_blocks = list(json.loads(context_file.read_text()).get("blocks", []))
            except (json.JSONDecodeError, AttributeError):
                context_blocks = []

        previous_window: dict[str, Any] = {}
        previous_calls: dict[str, Any] = {}
        previous_info = output_dir / "run_info.json"
        if previous_info.exists():
            try:
                previous_payload = json.loads(previous_info.read_text())
                previous_window = previous_payload.get("window", {})
                previous_calls = previous_payload.get("calls", {})
            except (json.JSONDecodeError, AttributeError):
                previous_window = {}
                previous_calls = {}
        offset = int(
            args.max_txs
            or previous_window.get("max_txs_per_address")
            or window.get("max_txs_per_address", 500)
        )
        return raw, labels, known, splits, {
            "calls": int(previous_calls.get("etherscan", 0)),
            "cache_hits": int(previous_calls.get("cache_hits", 0)),
            "skipped_fetch": True,
            "offset": offset,
            "startblock": int(previous_window.get("startblock", 0)),
            "endblock": int(previous_window.get("endblock", 99_999_999)),
            "context_blocks": context_blocks,
        }

    labels = fetch_mod.load_labels(
        PROJECT_ROOT / data_cfg["paths"]["labels_dir"],
        negative_count=negative_count, seed=seed,
    )
    known = fetch_mod.load_known_addresses(PROJECT_ROOT / "config" / "known_addresses.json")
    plan = fetch_mod.build_address_plan(labels, max_addresses=max_addresses, seed=seed)

    client = fetch_mod.EtherscanClient(
        fetch_mod.load_api_key(required=True),
        base_url=etherscan.get("base_url", "https://api.etherscan.io/v2/api"),
        chain_id=int(etherscan.get("chain_id", 1)),
        calls_per_second=float(etherscan.get("calls_per_second", 3)),
        timeout=int(etherscan.get("timeout_seconds", 30)),
        max_retries=int(etherscan.get("max_retries", 5)),
    )
    print(f"[fetch] checking Etherscan key: {client.check_key()}", flush=True)

    context_labels = pd.DataFrame(columns=["address", "label", "source", "confidence", "as_of", "stratum"])
    context_blocks: list[int] = []
    if not args.no_context and int(sampling.get("context_random_blocks", 20)) > 0:
        latest = client.block_number()
        if window.get("mode") == "fixed_blocks":
            block_start = int(window.get("fixed_block_start", max(0, latest - 2_000_000)))
            block_end = int(window.get("fixed_block_end", latest))
        else:
            block_start = max(0, latest - 2_000_000)
            block_end = latest
        if block_end <= block_start:
            block_end = latest
        exclude = set(plan["positives"]) | set(plan["negatives"])
        context, context_blocks = fetch_mod.sample_context_addresses(
            client, exclude=exclude, block_start=block_start, block_end=block_end,
            n_blocks=int(sampling.get("context_random_blocks", 20)),
            max_addresses=int(args.context_max or sampling.get("context_max_addresses", 500)),
            seed=seed,
        )
        if context:
            context_labels = pd.DataFrame({
                "address": context, "label": "context", "source": "context",
                "confidence": 1.0, "as_of": _now()[:10], "stratum": "context",
            })
        print(f"[fetch] context: {len(context)} addresses from {len(context_blocks)} blocks", flush=True)

    addresses = plan["positives"] + plan["negatives"] + context_labels["address"].tolist()
    offset = int(args.max_txs or window.get("max_txs_per_address", 500))
    startblock = 0
    endblock = 99_999_999
    if window.get("mode") == "fixed_blocks":
        startblock = int(window.get("fixed_block_start", 0))
        endblock = int(window.get("fixed_block_end", 99_999_999))

    raw, stats = fetch_mod.fetch_dataset(
        addresses, client, cache_dir, offset=offset, sort="desc",
        startblock=startblock, endblock=endblock, refresh=args.refresh,
    )
    fetch_mod.write_raw(raw, output_dir)

    base_labels = pd.concat([labels, context_labels], ignore_index=True)
    selected = set(plan["positives"]) | set(plan["negatives"])
    base_labels = base_labels[
        base_labels["address"].isin(selected) | (base_labels["label"] == "context")
    ].reset_index(drop=True)
    split_base = base_labels[base_labels["label"] != "context"]
    splits = fetch_mod.split_addresses(
        split_base, seed=seed, test_size=float(sampling.get("test_size", 0.3)),
    )
    if not context_labels.empty:
        context_splits = context_labels[["address"]].copy()
        context_splits["split"] = "context"
        context_splits["seed"] = seed
        splits = pd.concat([splits, context_splits], ignore_index=True)

    _write_json({"blocks": context_blocks, "count": len(context_blocks)}, output_dir / "context_blocks.json")
    return raw, base_labels, known, splits, {
        "calls": stats.calls, "cache_hits": stats.cache_hits, "failed": stats.failed,
        "addresses": stats.addresses, "context_blocks": context_blocks,
        "offset": offset, "startblock": startblock, "endblock": endblock,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _now()
    data_cfg = fetch_mod.load_data_config(PROJECT_ROOT / "config" / "data.yaml")
    output_dir = PROJECT_ROOT / (args.output or data_cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = data_cfg.get("sampling", {})
    seed = int(args.seed or sampling.get("seed", 42))

    if args.mode == "sample":
        raw, labels, known, splits = _sample_inputs(data_cfg)
        fetch_info = {"calls": 0, "cache_hits": 0, "data_source": "sample"}
    else:
        raw, labels, known, splits, fetch_info = _real_inputs(data_cfg, args)
        fetch_info["data_source"] = "etherscan_v2"

    print(f"[data] raw rows={len(raw):,} labels={len(labels):,} splits={len(splits):,}", flush=True)
    if raw.empty:
        raise SystemExit("No transactions collected. Check the API key/window or use --mode sample.")

    warehouse = output_dir / "chainlens.duckdb"
    con = features_mod.connect(warehouse)
    try:
        clean = features_mod.clean_transactions(con, raw)
        print(f"[features] clean rows={len(clean):,}", flush=True)
        if clean.empty:
            raise SystemExit("All transactions were filtered out by cleaning.")
        raw_features = features_mod.compute_features(con)
        print(f"[features] addresses={len(raw_features):,}", flush=True)

        train_addresses = set(splits.loc[splits["split"] == "train", "address"])
        features_scaled, scaler_version, scaler_rows = features_mod.scale_features(raw_features, train_addresses)
        print(f"[features] scaler={scaler_version} rows={len(scaler_rows)}", flush=True)
        features_mod.write_feature_tables(
            con, features_scaled, scaler_rows, labels, splits, known,
        )

        rules_cfg = scoring_mod.load_rules_config(PROJECT_ROOT / "config" / "rules.yaml", scaler_version)
        detection_cfg = patterns_mod.load_detection_config(PROJECT_ROOT / "config" / "detection.yaml")
        detection = patterns_mod.run_detection(
            con, features_scaled, labels, splits, rules_cfg, detection_cfg,
            known_addresses=known,
        )
        flags = detection["flags"]
        print(
            f"[patterns] graph={detection['graph'].number_of_nodes():,} nodes/"
            f"{detection['graph'].number_of_edges():,} edges; "
            f"cycles={detection['cycle_summary']['qualifying_cycles']} "
            f"dbscan={detection['dbscan_summary']['clusters']} clusters/"
            f"{detection['dbscan_summary']['noise']} noise; "
            f"apriori={detection['apriori_status']}; "
            f"louvain={detection['louvain_communities']} communities",
            flush=True,
        )

        evaluated_at = _now()
        scores, traces = scoring_mod.evaluate_scores(
            features_scaled, flags, known, rules_cfg, evaluated_at=evaluated_at,
        )
        for trace in traces["trace"]:
            scoring_mod.validate_trace(trace)
        print(
            f"[score] {len(scores):,} wallets; tiers="
            f"{scores['tier'].value_counts().to_dict()}",
            flush=True,
        )

        evaluation = evaluate_mod.evaluate(
            scores, labels, splits, features_scaled,
            threshold=rules_cfg.tier_high_min, include_fixture=(args.mode == "sample"),
        )
        ablation_frame = evaluate_mod.ablation(
            scores, features_scaled, flags, known, labels, splits, rules_cfg,
            threshold=rules_cfg.tier_high_min, include_fixture=(args.mode == "sample"),
        )

        # Output contract
        features_mod.write_parquet(raw, output_dir / "transactions_raw.parquet")
        features_mod.write_parquet(clean, output_dir / "transactions_clean.parquet")
        features_mod.write_parquet(features_scaled, output_dir / "wallet_features.parquet")
        features_mod.write_parquet(patterns_mod.flags_to_frame(flags), output_dir / "flags.parquet")
        features_mod.write_parquet(scores, output_dir / "scores.parquet")
        traces_out = scoring_mod.traces_to_frame(traces)
        features_mod.write_parquet(traces_out, output_dir / "traces.parquet")
        features_mod.write_parquet(patterns_mod.graph_edges_frame(detection["graph"]), output_dir / "graph_edges.parquet")
        features_mod.write_parquet(labels, output_dir / "labels.parquet")
        features_mod.write_parquet(splits, output_dir / "splits.parquet")
        features_mod.write_parquet(known, output_dir / "known_addresses.parquet")

        scores.to_csv(output_dir / "scores.csv", index=False)
        _flatten_traces(traces).to_csv(output_dir / "traces.csv", index=False)
        evaluate_mod.write_evaluation(evaluation, ablation_frame, output_dir)

        # Warehouse mirrors
        features_mod.write_table(con, "wallet_flags", patterns_mod.flags_to_frame(flags))
        features_mod.write_table(con, "risk_scores", scores)
        features_mod.write_table(con, "score_trace", traces_out)
        features_mod.write_table(con, "graph_edges", patterns_mod.graph_edges_frame(detection["graph"]))
        features_mod.write_table(con, "run_info", pd.DataFrame([{"run_info": json.dumps({}, default=str)}]))

        rules_text = (PROJECT_ROOT / "config" / "rules.yaml").read_text(encoding="utf-8")
        block_numbers = pd.to_numeric(clean["block_number"], errors="coerce").dropna()
        run_info = {
            "run_id": f"{started_at}-{rules_cfg.rules_version}",
            "mode": args.mode,
            "seed": seed,
            "data_source": fetch_info.get("data_source", "sample"),
            "chain_id": int(data_cfg.get("etherscan", {}).get("chain_id", 1)),
            "window": {
                **data_cfg.get("window", {}),
                **(
                    {
                        "max_txs_per_address": int(fetch_info["offset"]),
                        "startblock": int(fetch_info["startblock"]),
                        "endblock": int(fetch_info["endblock"]),
                    }
                    if fetch_info.get("offset") is not None
                    else {}
                ),
            },
            "context_blocks": fetch_info.get("context_blocks", []),
            "skipped_fetch": bool(fetch_info.get("skipped_fetch", False)),
            "block_range": {
                "min": int(block_numbers.min()) if not block_numbers.empty else None,
                "max": int(block_numbers.max()) if not block_numbers.empty else None,
            },
            "rows": {
                "raw": int(len(raw)),
                "clean": int(len(clean)),
                "features": int(len(features_scaled)),
                "labels": int(labels["address"].nunique()) if not labels.empty else 0,
                "label_rows": int(len(labels)),
                "positives": int(labels.loc[labels["label"].isin(["sanctioned", "darklist", "fraud"]), "address"].nunique()) if not labels.empty else 0,
                "negatives": int(labels.loc[labels["label"] == "negative", "address"].nunique()) if not labels.empty else 0,
                "context": int(labels.loc[labels["label"] == "context", "address"].nunique()) if not labels.empty else 0,
                "test_positives": int(evaluation.get("test_positives", 0)),
                "test_negatives": int(evaluation.get("test_negatives", 0)),
            },
            "calls": {
                "etherscan": int(fetch_info.get("calls", 0)),
                "cache_hits": int(fetch_info.get("cache_hits", 0)),
            },
            "api_key_present": bool(fetch_mod.load_api_key(required=False)) if args.mode == "real" else False,
            "rules_sha256": hashlib.sha256(rules_text.encode()).hexdigest(),
            "rules_version": rules_cfg.rules_version,
            "rules_version_label": rules_cfg.rules_version_label,
            "scaler_version": scaler_version,
            "engine_version": "v2",
            "packages": _packages(),
            "git_commit": _git_commit(),
            "started_at": started_at,
            "finished_at": _now(),
            "evaluation": {
                key: evaluation.get(key)
                for key in ("insufficient", "precision", "recall", "f1", "test_positives", "test_negatives")
            }
            | {
                "medium_f1": ((evaluation.get("operating_points") or {}).get("medium_25") or {}).get("f1"),
            },
            "patterns": {
                "graph_nodes": detection["graph"].number_of_nodes(),
                "graph_edges": detection["graph"].number_of_edges(),
                "cycles": detection["cycle_summary"]["qualifying_cycles"],
                "dbscan_clusters": detection["dbscan_summary"]["clusters"],
                "dbscan_noise": detection["dbscan_summary"]["noise"],
                "apriori_rules": len(detection["apriori_rules"]),
                "apriori_status": detection["apriori_status"],
                "dbscan_eps": detection["dbscan_summary"].get("eps"),
                "dbscan_min_samples": detection["dbscan_summary"].get("min_samples"),
                "louvain_communities": detection["louvain_communities"],
                **detection.get("pattern_stats", {}),
            },
        }
        _write_json(run_info, output_dir / "run_info.json")
        features_mod.write_table(con, "run_info", pd.DataFrame([{"run_info": json.dumps(run_info, default=str)}]))

        print(f"[eval] {evaluation}", flush=True)
        print(f"[done] output -> {output_dir}", flush=True)
        return run_info
    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChainLens v2 orchestrator")
    parser.add_argument("--mode", choices=("sample", "real"), default="sample")
    parser.add_argument("--max-addresses", type=int, default=None,
                        help="Development cap for real mode (half positives, half negatives)")
    parser.add_argument("--max-txs", type=int, default=None,
                        help="Override the per-address most-recent transaction cap")
    parser.add_argument("--context-max", type=int, default=None,
                        help="Override the maximum number of random-block context addresses")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached addresses")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse transactions_raw/labels/splits")
    parser.add_argument("--no-context", action="store_true", help="Skip random-block context sampling")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> int:
    """Console-script entry point (`uv run chainlens ...`)."""
    return 0 if run(build_parser().parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
