"""ChainLens dashboard.

Read-only Streamlit workspace for the artifacts written by run.py. The app never
imports scoring/patterns/evaluate and never recomputes features, flags, scores, or
evaluation metrics. It only renders and summarises data/output/.

Run:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("CHAINLENS_OUTPUT_DIR", APP_DIR / "data" / "output"))
TIER_ORDER = ["high", "medium", "low"]
TIER_COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#64748b"}
TIER_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
SIGNAL_LABELS = {
    "cycle_member": "R1 · cycle",
    "fanout": "R2 · fan-out",
    "sanctioned_1hop": "R3 · sanctioned 1-hop",
    "pagerank_top5": "R4 · PageRank top 5%",
    "apriori_match": "R5 · Apriori match",
    "dbscan_noise": "DBSCAN · noise",
    "dbscan_cluster": "DBSCAN · cluster",
    "louvain_community": "Louvain · community",
}
WALLET_SIGNALS = [
    "cycle_member", "fanout", "sanctioned_1hop",
    "pagerank_top5", "apriori_match", "dbscan_noise",
]
FLAG_COLORS = {
    "cycle_member": "#f97316",
    "fanout": "#eab308",
    "sanctioned_1hop": "#ef4444",
    "pagerank_top5": "#a78bfa",
    "apriori_match": "#38bdf8",
    "dbscan_noise": "#94a3b8",
    "": "#475569",
}
COMMUNITY_COLORS = [
    "#38bdf8", "#a78bfa", "#f472b6", "#f97316", "#facc15",
    "#34d399", "#fb7185", "#c084fc", "#2dd4bf", "#f59e0b",
]
FEATURE_FIELDS = [
    "tx_count_out", "tx_count_in", "tx_count_total", "unique_counterparties",
    "unique_senders", "fanout_ratio", "out_in_ratio", "total_value_eth",
    "avg_value_eth", "median_value_eth", "net_flow_eth", "active_days",
    "round_number_ratio", "is_contract",
]
VIEWS = [
    "Overview",
    "Triage",
    "Wallet dossier",
    "Network",
    "Patterns",
    "Evaluation",
    "Data & export",
]
FILTER_KEYS = [
    "flt_tiers", "flt_score", "flt_rules", "flt_known", "flt_label",
    "flt_split", "flt_flagged", "flt_watchlist", "flt_search",
]
EXPORT_FILES = [
    "scores.parquet", "scores.csv", "flags.parquet", "traces.parquet",
    "traces.csv", "graph_edges.parquet", "evaluation.json", "evaluation.md",
    "ablation.csv", "run_info.json",
]
ARTIFACT_SPECS = {
    "scores.parquet": "Wallet scores and tiers",
    "traces.parquet": "Per-wallet R1–R5 trace",
    "flags.parquet": "Rule, DBSCAN, PageRank and Louvain signals",
    "graph_edges.parquet": "Aggregated transaction graph",
    "wallet_features.parquet": "Raw and train-scaled wallet features",
    "labels.parquet": "Labels and provenance",
    "splits.parquet": "Leakage-safe train/test split",
    "known_addresses.parquet": "Down-weight seed addresses",
    "transactions_clean.parquet": "Cleaned transaction evidence",
    "evaluation.json": "Held-out evaluation metrics",
    "ablation.csv": "Leave-one-rule-out ablation",
    "run_info.json": "Run provenance and versions",
}

st.set_page_config(
    page_title="ChainLens · Wallet risk workspace",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1600px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; letter-spacing: -0.02em; }
    .cl-kicker { color:#7dd3fc; font-size:.72rem; font-weight:700; letter-spacing:.14em;
                 text-transform:uppercase; margin-bottom:.2rem; }
    .cl-header { border-bottom:1px solid #1e293b; padding:.2rem 0 .9rem; margin-bottom:1rem; }
    .cl-header h1 { margin:0; font-size:1.65rem; letter-spacing:-.03em; }
    .cl-header p { margin:.35rem 0 0; color:#94a3b8; font-size:.9rem; }
    .cl-badge { display:inline-block; padding:.12rem .52rem; border-radius:999px;
                font-size:.72rem; font-weight:700; letter-spacing:.02em; }
    .cl-high { background:#7f1d1d; color:#fecaca; }
    .cl-medium { background:#78350f; color:#fde68a; }
    .cl-low { background:#1e293b; color:#cbd5e1; }
    .cl-chip { display:inline-block; padding:.18rem .55rem; border-radius:6px;
               background:#111827; border:1px solid #1e293b; color:#cbd5e1;
               font-size:.76rem; margin:0 .3rem .3rem 0; }
    .cl-muted { color:#94a3b8; font-size:.85rem; }
    div[data-testid="stSidebar"] { border-right:1px solid #1e293b; }
    div[data-testid="stSidebar"] .block-container { padding-top:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _stamp(directory: Path) -> float:
    names = [
        "scores.parquet", "traces.parquet", "flags.parquet", "graph_edges.parquet",
        "wallet_features.parquet", "transactions_clean.parquet", "evaluation.json",
        "ablation.csv", "run_info.json", "labels.parquet", "splits.parquet",
        "known_addresses.parquet",
    ]
    values = [
        (directory / name).stat().st_mtime
        for name in names
        if (directory / name).exists()
    ]
    return max(values) if values else 0.0


def _norm_address(value: Any) -> str:
    return str(value).lower() if value is not None else ""


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def short(address: str, left: int = 6, right: int = 4) -> str:
    text = _norm_address(address)
    return f"{text[:left]}…{text[-right:]}" if len(text) > left + right + 2 else text


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number:,.{digits}f}"


def pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number * 100:.1f}%"


def tier_badge(tier: str) -> str:
    key = str(tier).lower()
    klass = f"cl-{key}" if key in TIER_COLORS else "cl-low"
    return f'<span class="cl-badge {klass}">{TIER_LABELS.get(key, key.upper())}</span>'


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def chart_layout(fig: go.Figure, height: int = 320, **kwargs: Any) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=12),
        **kwargs,
    )
    return fig


def evidence_summary(rule_id: str, inputs: Any) -> str:
    if not isinstance(inputs, dict):
        return "—"
    if rule_id == "R1":
        return f"{inputs.get('cycle_count', 0)} cycle(s), SCC {inputs.get('scc_size', '—')}"
    if rule_id == "R2":
        return (f"{inputs.get('unique_counterparties', '—')} counterparties, "
                f"out/in {fmt(inputs.get('out_in_ratio'), 2)}")
    if rule_id == "R3":
        return f"seed {short(inputs.get('seed', '—'))} ({inputs.get('direction', '—')})"
    if rule_id == "R4":
        return f"PageRank {fmt(inputs.get('pagerank'), 6)} ≥ {fmt(inputs.get('threshold'), 6)}"
    if rule_id == "R5":
        antecedent = inputs.get("antecedent") or []
        return f"{' & '.join(antecedent)} → flagged; lift {fmt(inputs.get('lift'), 2)}"
    return json.dumps(inputs, default=str)[:160]


# ── artifact loading ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_artifacts(output_dir: str, stamp: float) -> dict[str, Any]:
    del stamp  # cache key only
    root = Path(output_dir)
    data: dict[str, Any] = {
        "root": root,
        "missing": [],
        "errors": [],
        "warnings": [],
    }

    def parquet(name: str, required: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
        path = root / name
        if not path.exists():
            if required:
                data["missing"].append(name)
                data["errors"].append(f"Missing required artifact: {name}")
            return pd.DataFrame()
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception as exc:  # pragma: no cover - defensive around corrupt files
            data["warnings"].append(f"Could not read {name}: {exc}")
            return pd.DataFrame()

    def csv(name: str) -> pd.DataFrame:
        path = root / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover
            data["warnings"].append(f"Could not read {name}: {exc}")
            return pd.DataFrame()

    def js(name: str, required: bool = False) -> dict[str, Any]:
        path = root / name
        if not path.exists():
            if required:
                data["missing"].append(name)
                data["errors"].append(f"Missing required artifact: {name}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            data["warnings"].append(f"Could not parse {name}: {exc}")
            return {}

    scores = parquet("scores.parquet", required=True)
    traces_index = parquet("traces.parquet", columns=["address", "rules_version"])
    features = parquet("wallet_features.parquet")
    flags = parquet("flags.parquet")
    edges = parquet("graph_edges.parquet")
    labels = parquet("labels.parquet")
    splits = parquet("splits.parquet")
    known = parquet("known_addresses.parquet")
    evaluation = js("evaluation.json")
    run_info = js("run_info.json")
    ablation = csv("ablation.csv")

    for frame in (scores, traces_index, features, flags, labels, splits, known):
        if not frame.empty and "address" in frame.columns:
            frame["address"] = frame["address"].map(_norm_address)
    if not edges.empty:
        for column in ("from_address", "to_address"):
            if column in edges.columns:
                edges[column] = edges[column].map(_norm_address)

    if scores.empty:
        data.update({
            "scores": scores, "traces": traces_index, "features": features,
            "flags": flags, "edges": edges, "labels": labels, "splits": splits,
            "known": known, "evaluation": evaluation, "run_info": run_info,
            "ablation": ablation,
        })
        return data

    required = {"address", "raw_score", "score", "tier", "top_driver"}
    missing_columns = sorted(required - set(scores.columns))
    if missing_columns:
        data["errors"].append(f"scores.parquet is missing columns: {', '.join(missing_columns)}")

    scores = scores.copy()
    scores["address"] = scores["address"].map(_norm_address)
    for column in ("raw_score", "score"):
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(0).astype(int)
    scores["tier"] = scores.get("tier", "low").astype(str).str.lower()
    scores["top_driver"] = scores.get("top_driver", "").fillna("")

    label_map: dict[str, str] = {}
    source_map: dict[str, str] = {}
    if not labels.empty:
        if "label" in labels.columns:
            label_map = (
                labels.assign(label=labels["label"].astype(str))
                .groupby("address")["label"]
                .apply(lambda values: ", ".join(sorted(set(values))))
                .to_dict()
            )
        if "source" in labels.columns:
            source_map = (
                labels.assign(source=labels["source"].astype(str))
                .groupby("address")["source"]
                .apply(lambda values: ", ".join(sorted(set(values))))
                .to_dict()
            )
    split_map = (
        dict(zip(splits["address"], splits["split"], strict=False))
        if not splits.empty and "split" in splits.columns else {}
    )
    known_map = (
        dict(zip(known["address"], known["category"], strict=False))
        if not known.empty and "category" in known.columns else {}
    )

    scores["label"] = scores["address"].map(label_map).fillna("")
    scores["source"] = scores["address"].map(source_map).fillna("")
    scores["split"] = scores["address"].map(split_map).fillna("")
    scores["known_category"] = scores["address"].map(known_map).fillna("")

    flag_sets: dict[str, set[str]] = {}
    signal_hits: dict[str, set[str]] = {signal: set() for signal in SIGNAL_LABELS}
    community_map: dict[str, int] = {}
    dbscan_map: dict[str, int] = {}
    if not flags.empty and "signal_name" in flags.columns:
        for row in flags.itertuples(index=False):
            address = _norm_address(getattr(row, "address", ""))
            signal = str(getattr(row, "signal_name", ""))
            if not address:
                continue
            flag_sets.setdefault(address, set()).add(signal)
            if signal in signal_hits:
                signal_hits[signal].add(address)
            if signal in {"louvain_community", "dbscan_cluster"}:
                try:
                    value = int(float(getattr(row, "signal_value", -1)))
                except (TypeError, ValueError):
                    value = -1
                (community_map if signal == "louvain_community" else dbscan_map)[address] = value

    for signal in SIGNAL_LABELS:
        scores[f"hit_{signal}"] = scores["address"].isin(signal_hits.get(signal, set()))
    scores["flag_count"] = scores[[f"hit_{signal}" for signal in WALLET_SIGNALS]].sum(axis=1).astype(int)

    def primary_flag(address: str) -> str:
        signals = flag_sets.get(address, set())
        for signal in WALLET_SIGNALS:
            if signal in signals:
                return signal
        return ""

    scores["primary_flag"] = scores["address"].map(primary_flag)
    scores["community_id"] = scores["address"].map(community_map)
    scores["dbscan_cluster"] = scores["address"].map(dbscan_map)

    if feature_columns := (set(features.columns) if not features.empty else set()):
        if "scaler_version" not in feature_columns:
            data["warnings"].append("wallet_features.parquet has no scaler_version column.")

    if run_info:
        if run_info.get("rows", {}).get("features") not in (None, len(scores)):
            data["warnings"].append(
                "run_info.rows.features does not match the number of scored wallets "
                f"({run_info['rows'].get('features')} vs {len(scores)})."
            )
        score_version = str(scores["rules_version"].iloc[0]) if "rules_version" in scores else ""
        if score_version and run_info.get("rules_version") not in (None, score_version):
            data["warnings"].append(
                f"rules_version mismatch: scores={score_version}, "
                f"run_info={run_info.get('rules_version')}."
            )
        if not features.empty and "scaler_version" in features:
            feature_versions = {str(v) for v in features["scaler_version"].dropna().unique()}
            run_scaler = run_info.get("scaler_version")
            if run_scaler and feature_versions and run_scaler not in feature_versions:
                data["warnings"].append(
                    f"scaler_version mismatch: features={sorted(feature_versions)}, "
                    f"run_info={run_scaler}."
                )
        if not traces_index.empty and "rules_version" in traces_index:
            trace_versions = {str(v) for v in traces_index["rules_version"].dropna().unique()}
            if trace_versions and run_info.get("rules_version") not in (None, *trace_versions):
                data["warnings"].append("traces.parquet rules_version does not match run_info.")
        if run_info.get("skipped_fetch"):
            data["warnings"].append(
                "run_info reports --skip-fetch: fetch counters describe a previous run, "
                "not this analytics run."
            )
        score_mtime = (root / "scores.parquet").stat().st_mtime if (root / "scores.parquet").exists() else 0
        info_mtime = (root / "run_info.json").stat().st_mtime if (root / "run_info.json").exists() else 0
        if info_mtime and score_mtime > info_mtime + 2:
            data["warnings"].append("scores.parquet is newer than run_info.json; artifacts may be mixed.")

    if evaluation and not evaluation.get("operating_points"):
        data["warnings"].append("evaluation.json has no operating_points. High and Medium blocks may be incomplete.")

    data.update({
        "scores": scores,
        "traces": traces_index,
        "features": features,
        "flags": flags,
        "edges": edges,
        "labels": labels,
        "splits": splits,
        "known": known,
        "evaluation": evaluation,
        "run_info": run_info,
        "ablation": ablation,
        "flag_sets": flag_sets,
        "community_map": community_map,
        "dbscan_map": dbscan_map,
    })
    return data


@st.cache_data(show_spinner=False)
def load_trace(output_dir: str, address: str, stamp: float) -> dict[str, Any] | None:
    del stamp
    path = Path(output_dir) / "traces.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, filters=[("address", "==", address)])
    except Exception:
        frame = pd.read_parquet(path)
        frame = frame[frame["address"].astype(str).str.lower() == address]
    if frame.empty:
        return None
    value = frame.iloc[0].get("trace_json")
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else None


@st.cache_data(show_spinner=False)
def load_wallet_transactions(output_dir: str, address: str, stamp: float) -> pd.DataFrame:
    del stamp
    path = Path(output_dir) / "transactions_clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    filters = [
        [("from_address", "==", address)],
        [("to_address", "==", address)],
    ]
    try:
        frame = pd.read_parquet(path, filters=filters)
    except Exception:
        frame = pd.read_parquet(path)
        frame = frame[
            (frame["from_address"].astype(str).str.lower() == address)
            | (frame["to_address"].astype(str).str.lower() == address)
        ]
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["from_address"] = frame["from_address"].map(_norm_address)
    frame["to_address"] = frame["to_address"].map(_norm_address)
    frame["direction"] = frame.apply(
        lambda row: "out" if row["from_address"] == address else "in", axis=1
    )
    frame["counterparty"] = frame.apply(
        lambda row: row["to_address"] if row["direction"] == "out" else row["from_address"],
        axis=1,
    )
    if "block_timestamp" in frame:
        frame["block_timestamp"] = pd.to_datetime(frame["block_timestamp"], errors="coerce")
    return frame.sort_values("block_timestamp", ascending=False, na_position="last")


@st.cache_data(show_spinner=False)
def compute_layout(
    nodes: tuple[str, ...],
    edge_pairs: tuple[tuple[str, str], ...],
) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_pairs)
    return nx.spring_layout(graph, seed=42, k=0.9, iterations=60)


@st.cache_data(show_spinner=False)
def build_export_zip(output_dir: str, stamp: float) -> bytes:
    del stamp
    root = Path(output_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in EXPORT_FILES:
            path = root / name
            if path.exists():
                archive.write(path, arcname=name)
    return buffer.getvalue()




# ── session state & navigation ────────────────────────────────────────────────
if "active_view" not in st.session_state:
    st.session_state.active_view = "Overview"
if "selected_wallet" not in st.session_state:
    st.session_state.selected_wallet = ""
if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()
if "triage_page" not in st.session_state:
    st.session_state.triage_page = 0
if "triage_page_size" not in st.session_state:
    st.session_state.triage_page_size = 50


def navigate(view: str, wallet: str | None = None) -> None:
    """Request a view change from a button, applied before the nav radio renders.

    Writing st.session_state.nav_radio directly here would fail: the sidebar
    radio is already instantiated by the time row buttons run. Stash the request
    and let main() apply it at the start of the next run.
    """
    if wallet:
        st.session_state.selected_wallet = wallet
    st.session_state["_pending_view"] = view
    st.rerun()


def copy_popover(address: str, key: str) -> None:
    """Compact per-row control exposing the full address with a copy button."""
    with st.popover("⧉", help="Show and copy wallet address", key=key):
        st.code(address, language=None)


def copyable_address(address: str) -> None:
    """Full address in a code block; Streamlit adds a one-click copy icon."""
    st.code(address, language=None)



# ── view: Overview ────────────────────────────────────────────────────────────
def render_overview(bundle: dict[str, Any]) -> None:
    scores: pd.DataFrame = bundle["scores"]
    run_info: dict[str, Any] = bundle["run_info"]
    evaluation: dict[str, Any] = bundle["evaluation"]
    st.markdown('<div class="cl-kicker">Run Overview & Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>ChainLens Intelligence Overview</h1>'
        '<p>Read-only monitoring workspace for pipeline execution artifacts, risk distributions, and triage priorities.</p></div>',
        unsafe_allow_html=True,
    )

    mode = run_info.get("mode", "real" if len(scores) > 1000 else "sample").upper()
    data_source = run_info.get("data_source", "Etherscan API" if mode == "REAL" else "Sample fixture")
    st.info(
        f"**Pipeline Run**: {mode} Mode · Data source: **{data_source}** · "
        f"**{len(scores):,}** wallets analyzed · Rules SHA: `{run_info.get('rules_sha256', '—')[:12]}…` · "
        f"Scaler: `{run_info.get('scaler_version', '—')}`"
    )

    # Top metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Analyzed Wallets", f"{len(scores):,}", "Total wallets scored in dataset")
    with c2:
        high_cnt = int((scores["tier"] == "high").sum())
        metric_card("High Risk (≥60)", f"{high_cnt:,}", "Tier: High risk score 60-100")
    with c3:
        med_cnt = int((scores["tier"] == "medium").sum())
        metric_card("Medium Risk (25-59)", f"{med_cnt:,}", "Tier: Medium risk score 25-59")
    with c4:
        low_cnt = int((scores["tier"] == "low").sum())
        metric_card("Low Risk (0-24)", f"{low_cnt:,}", "Tier: Low risk score 0-24")
    with c5:
        flagged_cnt = int((scores["flag_count"] > 0).sum())
        metric_card("Flagged Wallets", f"{flagged_cnt:,}", "Wallets triggering at least one heuristic or ML signal")
    with c6:
        ap = evaluation.get("average_precision")
        metric_card("Test Avg Precision", pct(ap), "Area under PR curve on held-out test split")

    if high_cnt == 0:
        st.warning(
            "ℹ️ **Operational Note: High Tier (≥60) is currently empty.** On real transaction subgraphs, "
            "triggering multiple compound rules (R1+R2+R3) is rare. The **Medium Tier (25–59)** represents "
            f"the active investigative queue ({med_cnt:,} wallets) with an operating Average Precision of {pct(ap)}."
        )

    st.markdown("### Risk & Signal Distributions")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Risk Score Distribution (0–100)**")
        score_counts = scores["score"].value_counts().sort_index()
        fig_score = go.Figure(go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            marker_color=[
                TIER_COLORS["high"] if s >= 60 else TIER_COLORS["medium"] if s >= 25 else TIER_COLORS["low"]
                for s in score_counts.index
            ],
            hovertemplate="Score: %{x}<br>Wallets: %{y:,}<extra></extra>",
        ))
        chart_layout(fig_score, height=280, xaxis_title="Composite Score", yaxis_title="Wallet Count")
        fig_score.update_layout(bargap=0.1)
        st.plotly_chart(fig_score, width="stretch")

    with col_right:
        st.markdown("**Signal & Rule Hit Frequencies**")
        rule_hits = {}
        for sig, lbl in SIGNAL_LABELS.items():
            col_name = f"hit_{sig}"
            if col_name in scores.columns:
                rule_hits[lbl] = int(scores[col_name].sum())
        if rule_hits:
            fig_rules = go.Figure(go.Bar(
                x=list(rule_hits.values()),
                y=list(rule_hits.keys()),
                orientation="h",
                marker_color="#38bdf8",
                text=[f"{v:,}" for v in rule_hits.values()],
                textposition="auto",
                hovertemplate="%{y}: %{x:,} hits<extra></extra>",
            ))
            chart_layout(fig_rules, height=280, xaxis_title="Hits", yaxis_title="")
            fig_rules.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rules, width="stretch")

    st.divider()

    st.markdown("### Priority Triage Queue (Top Risk Candidates)")
    st.caption("Highest-scoring candidates requiring immediate compliance and forensic review.")
    top_candidates = scores[scores["score"] >= 25].sort_values(
        by=["score", "raw_score"], ascending=False
    ).head(10)

    if top_candidates.empty:
        st.info("No candidates scored at or above Medium threshold (25).")
    else:
        for row in top_candidates.itertuples():
            addr = row.address
            score_val = row.score
            raw_val = row.raw_score
            tier_val = row.tier
            driver_val = row.top_driver
            flags_val = row.flag_count
            lbl_val = getattr(row, "label", "") or "—"

            c_badge, c_addr, c_score, c_driver, c_flags, c_label, c_copy, c_action = st.columns([1, 2.5, 1.2, 1.5, 1, 1.5, 0.5, 1.2])
            with c_badge:
                st.markdown(tier_badge(tier_val), unsafe_allow_html=True)
            with c_addr:
                st.markdown(f"`{short(addr, 10, 8)}`")
            with c_score:
                st.markdown(f"**{score_val}** / 100 <span class='cl-muted'>(raw {raw_val})</span>", unsafe_allow_html=True)
            with c_driver:
                st.markdown(f"<span class='cl-chip'>{driver_val or 'None'}</span>", unsafe_allow_html=True)
            with c_flags:
                st.markdown(f"{flags_val} signal(s)")
            with c_label:
                st.markdown(f"<span class='cl-muted'>{lbl_val}</span>", unsafe_allow_html=True)
            with c_copy:
                copy_popover(addr, f"top_copy_{addr}")
            with c_action:
                if st.button("Dossier →", key=f"top_{addr}"):
                    navigate("Wallet dossier", addr)


# ── view: Triage ──────────────────────────────────────────────────────────────
def render_triage(bundle: dict[str, Any]) -> None:
    scores: pd.DataFrame = bundle["scores"]
    st.markdown('<div class="cl-kicker">Investigative Queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Wallet Risk Triage</h1>'
        '<p>Search, filter, and triage flagged wallets across heuristic rules and behavioural patterns.</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Filter Controls", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            selected_tiers = st.multiselect(
                "Risk Tier",
                options=TIER_ORDER,
                default=["high", "medium"],
                format_func=lambda t: TIER_LABELS.get(t, t.capitalize()),
                key="flt_tiers",
            )
        with f2:
            min_score, max_score = st.slider(
                "Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5,
                key="flt_score",
            )
        with f3:
            rule_options = list(WALLET_SIGNALS)
            selected_rules = st.multiselect(
                "Triggered Rule",
                options=rule_options,
                default=[],
                format_func=lambda r: SIGNAL_LABELS.get(r, r),
                key="flt_rules",
            )
        with f4:
            split_options = ["All", "train", "test"]
            selected_split = st.selectbox("Split Partition", options=split_options, key="flt_split")

        f5, f6, f7, f8 = st.columns(4)
        with f5:
            known_cats = sorted(set(scores["known_category"].dropna().unique()) - {""})
            selected_known = st.multiselect(
                "Known Entity Category",
                options=known_cats,
                default=[],
                key="flt_known",
            )
        with f6:
            search_query = st.text_input("Search Address or Label", placeholder="0x... or label", key="flt_search").strip().lower()
        with f7:
            flagged_only = st.checkbox("Flagged wallets only (≥1 signal)", value=False, key="flt_flagged")
        with f8:
            watchlist_only = st.checkbox("Watchlist only", value=False, key="flt_watchlist")

        c_reset, _ = st.columns([1, 5])
        with c_reset:
            if st.button("↺ Reset Filters"):
                for k in FILTER_KEYS:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    # Apply filters
    filtered = scores.copy()
    if selected_tiers:
        filtered = filtered[filtered["tier"].isin(selected_tiers)]
    if min_score > 0 or max_score < 100:
        filtered = filtered[(filtered["score"] >= min_score) & (filtered["score"] <= max_score)]
    if selected_rules:
        for r in selected_rules:
            col = f"hit_{r}"
            if col in filtered.columns:
                filtered = filtered[filtered[col]]
    if selected_split and selected_split != "All":
        filtered = filtered[filtered["split"] == selected_split]
    if selected_known:
        filtered = filtered[filtered["known_category"].isin(selected_known)]
    if search_query:
        mask = (
            filtered["address"].str.contains(search_query, na=False)
            | filtered["label"].str.lower().str.contains(search_query, na=False)
            | filtered["source"].str.lower().str.contains(search_query, na=False)
        )
        filtered = filtered[mask]
    if flagged_only:
        filtered = filtered[filtered["flag_count"] > 0]
    if watchlist_only:
        filtered = filtered[filtered["address"].isin(st.session_state.watchlist)]

    st.markdown(
        f"**Displaying {len(filtered):,} matching wallets** (of {len(scores):,} total in cohort)"
    )

    # Export filtered button
    csv_bytes = filtered[[
        "address", "score", "raw_score", "tier", "top_driver",
        "flag_count", "known_category", "split", "label", "source"
    ]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Filtered Wallets (CSV)",
        data=csv_bytes,
        file_name="chainlens_triage_filtered.csv",
        mime="text/csv",
    )

    # Pagination controls
    page_size = st.selectbox("Rows per page", options=[25, 50, 100], index=1, key="ps_select")
    total_pages = max(1, math.ceil(len(filtered) / page_size))
    current_page = max(0, min(st.session_state.triage_page, total_pages - 1))

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ Previous Page", disabled=(current_page == 0)):
            st.session_state.triage_page = max(0, current_page - 1)
            st.rerun()
    with col_info:
        st.markdown(f"<div style='text-align:center; padding-top:0.4rem;'>Page <b>{current_page + 1}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next Page ▶", disabled=(current_page >= total_pages - 1)):
            st.session_state.triage_page = min(total_pages - 1, current_page + 1)
            st.rerun()

    start_idx = current_page * page_size
    page_data = filtered.iloc[start_idx : start_idx + page_size]

    if page_data.empty:
        st.info("No wallets match the specified filters.")
        return

    # Render table header
    st.markdown(
        """
        <div style="display:grid; grid-template-columns: 0.8fr 2.5fr 1fr 1.2fr 1fr 1.2fr 1.5fr 1fr;
                    font-weight:700; color:#94a3b8; font-size:0.8rem; border-bottom:1px solid #334155; padding-bottom:0.4rem; margin-top:0.8rem;">
            <div>TIER</div>
            <div>ADDRESS</div>
            <div>SCORE</div>
            <div>DRIVER</div>
            <div>SIGNALS</div>
            <div>KNOWN / SPLIT</div>
            <div>LABEL / SOURCE</div>
            <div>ACTION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row in page_data.itertuples():
        addr = row.address
        tier_val = row.tier
        score_val = row.score
        raw_val = row.raw_score
        driver_val = row.top_driver or "—"
        flags_val = row.flag_count
        known_val = row.known_category or "—"
        split_val = row.split or "—"
        label_val = getattr(row, "label", "") or "—"
        source_val = getattr(row, "source", "") or ""

        meta_sub = f"{known_val} · {split_val}" if known_val != "—" else split_val
        label_disp = f"{label_val} ({source_val})" if source_val and label_val != "—" else label_val

        c_t, c_a, c_s, c_d, c_sig, c_m, c_l, c_copy, c_act = st.columns([0.8, 2.5, 1, 1.2, 1, 1.2, 1.5, 0.5, 1])
        with c_t:
            st.markdown(tier_badge(tier_val), unsafe_allow_html=True)
        with c_a:
            is_starred = "⭐ " if addr in st.session_state.watchlist else ""
            st.markdown(f"`{is_starred}{short(addr, 8, 6)}`")
        with c_s:
            st.markdown(f"**{score_val}** <span class='cl-muted'>({raw_val})</span>", unsafe_allow_html=True)
        with c_d:
            st.markdown(f"<span class='cl-chip'>{driver_val}</span>", unsafe_allow_html=True)
        with c_sig:
            st.markdown(f"{flags_val}")
        with c_m:
            st.markdown(f"<span class='cl-muted'>{meta_sub}</span>", unsafe_allow_html=True)
        with c_l:
            st.markdown(f"<span class='cl-muted'>{short(label_disp, 16, 4)}</span>", unsafe_allow_html=True)
        with c_copy:
            copy_popover(addr, f"triage_copy_{addr}")
        with c_act:
            if st.button("Inspect", key=f"triage_btn_{addr}"):
                navigate("Wallet dossier", addr)


# ── view: Wallet dossier ──────────────────────────────────────────────────────
def render_dossier(bundle: dict[str, Any]) -> None:
    scores: pd.DataFrame = bundle["scores"]
    features: pd.DataFrame = bundle["features"]
    edges: pd.DataFrame = bundle["edges"]

    st.markdown('<div class="cl-kicker">Subject Forensics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Wallet Investigative Dossier</h1>'
        '<p>Detailed multi-rule evidence trace, transaction timeline, graph neighborhood, and feature percentiles.</p></div>',
        unsafe_allow_html=True,
    )

    # Select wallet control
    candidate_list = scores.sort_values("score", ascending=False)["address"].head(100).tolist()
    curr_addr = st.session_state.selected_wallet
    if curr_addr not in candidate_list and curr_addr in set(scores["address"]):
        candidate_list = [curr_addr] + candidate_list

    c_sel, c_in = st.columns([2, 2])
    with c_sel:
        idx = candidate_list.index(curr_addr) if curr_addr in candidate_list else 0
        chosen = st.selectbox(
            "Select from priority candidate list (Top 100)",
            options=candidate_list,
            index=idx,
            format_func=lambda a: f"{short(a)} · Score {scores.loc[scores['address']==a, 'score'].values[0]} ({scores.loc[scores['address']==a, 'tier'].values[0].upper()})" if a in set(scores['address']) else a,
            key="dossier_select_box"
        )
        if chosen != curr_addr:
            st.session_state.selected_wallet = chosen
            curr_addr = chosen
    with c_in:
        custom = st.text_input("Or enter any 0x wallet address", placeholder="0x...").strip().lower()
        if custom and custom in set(scores["address"]) and custom != curr_addr:
            st.session_state.selected_wallet = custom
            st.rerun()

    addr = st.session_state.selected_wallet
    if not addr or addr not in set(scores["address"]):
        st.warning("Please select or enter a valid scored wallet address.")
        return

    row = scores[scores["address"] == addr].iloc[0]
    tier_val = row["tier"]
    score_val = int(row["score"])
    raw_val = int(row["raw_score"])
    driver_val = row["top_driver"]
    known_val = row["known_category"]
    split_val = row["split"]
    label_val = row["label"]
    source_val = row["source"]

    # Dossier Banner
    st.markdown("---")
    top_c1, top_c2, top_c3 = st.columns([3, 1, 1])
    with top_c1:
        st.markdown("### Address")
        copyable_address(addr)
        badge_html = tier_badge(tier_val)
        known_chip = f"<span class='cl-chip'>Known: {known_val} (0.3x multiplier)</span>" if known_val else ""
        split_chip = f"<span class='cl-chip'>Split: {split_val}</span>" if split_val else ""
        st.markdown(f"{badge_html} {known_chip} {split_chip}", unsafe_allow_html=True)
        if label_val:
            st.markdown(f"**Known Label**: `{label_val}` (Source: {source_val})")
        etherscan_url = f"https://etherscan.io/address/{addr}"
        st.markdown(f"[↗ View on Etherscan]({etherscan_url})", unsafe_allow_html=True)

    with top_c2:
        metric_card("Composite Score", f"{score_val} / 100", f"Clamped risk score (raw: {raw_val})")
    with top_c3:
        if addr in st.session_state.watchlist:
            if st.button("★ Remove Watchlist", key="btn_wl_rem"):
                st.session_state.watchlist.remove(addr)
                st.rerun()
        else:
            if st.button("☆ Add to Watchlist", key="btn_wl_add"):
                st.session_state.watchlist.add(addr)
                st.rerun()

    stamp = _stamp(OUTPUT_DIR)
    trace = load_trace(str(OUTPUT_DIR), addr, stamp)

    # Rule Evidence Cards
    st.markdown("### Heuristic & Machine Learning Evidence (R1–R5)")
    if not trace or "rules" not in trace:
        st.info("Detailed trace JSON is not available for this address.")
    else:
        rules_data = trace["rules"]
        rule_cards = [
            ("R1", "Circular Flow Detection", 25, "Cycle length 2–5, values within ±2%, completed within 24h"),
            ("R2", "Scatter–Gather Smurf Motif", 20, "Scatter precedes gather through intermediaries within 72h"),
            ("R3", "Sanctioned Entity 1-Hop", 30, "≥0.01 ETH direct exposure to a frozen OFAC seed, scaled by exposure"),
            ("R4", "PageRank Network Prominence", 15, "Continuous value-weighted centrality tail (≥90th pct)"),
            ("R5", "Apriori Behavioural Match", 20, "Frequent association itemset matching known illicit profiles"),
        ]

        cols = st.columns(5)
        for i, (rid, rname, rwt, rdesc) in enumerate(rule_cards):
            rinfo = rules_data.get(rid, {})
            triggered = bool(rinfo.get("hit", False))
            pts = float(rinfo.get("contribution", 0))
            pts_label = f"{pts:.1f}".rstrip("0").rstrip(".") if pts else "0"
            inputs = rinfo.get("inputs", {})

            with cols[i]:
                border_color = "#ef4444" if triggered else "#334155"
                bg_color = "#1e1e2d" if triggered else "#0f172a"
                st.markdown(
                    f"""
                    <div style="border: 1px solid {border_color}; background:{bg_color}; border-radius:8px; padding:10px; min-height:160px;">
                        <div style="font-weight:700; color:{'#f87171' if triggered else '#94a3b8'}; font-size:0.85rem;">{rid} · {rwt} pts</div>
                        <div style="font-size:0.8rem; font-weight:600; margin:4px 0;">{rname}</div>
                        <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">{rdesc}</div>
                        <div style="font-size:0.85rem; font-weight:700;">{'🚨 TRIGGERED (+'+pts_label+' pts)' if triggered else '⚪ NOT TRIGGERED'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Details", expanded=False):
                    st.caption(evidence_summary(rid, inputs))
                    st.json(inputs)

    st.markdown("### Score Contribution Breakdown")
    rule_pts = {}
    if trace and "rules" in trace:
        for r_id, r_info in trace["rules"].items():
            rule_pts[f"{r_id} ({r_info.get('name', r_id)})"] = r_info.get("contribution", 0)

    if rule_pts:
        fig_bar = go.Figure(go.Bar(
            x=list(rule_pts.values()),
            y=list(rule_pts.keys()),
            orientation="h",
            marker_color=["#ef4444" if v > 0 else "#475569" for v in rule_pts.values()],
            text=[f"+{float(v):.1f} pts" if v > 0 else "0" for v in rule_pts.values()],
            textposition="auto",
        ))
        chart_layout(fig_bar, height=220, xaxis_title="Score Points", yaxis_title="")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, width="stretch")

    tab_tx, tab_feat, tab_nb, tab_raw = st.tabs([
        "📜 Transaction Evidence",
        "📊 Behavioral Features",
        "🕸️ Graph Neighbors",
        "💾 Raw Case File (JSON)",
    ])

    with tab_tx:
        st.markdown("**Cleaned Transactions**")
        tx_df = load_wallet_transactions(str(OUTPUT_DIR), addr, stamp)
        if tx_df.empty:
            st.info("No transaction events found for this wallet in cleaned dataset.")
        else:
            st.caption(f"Found {len(tx_df):,} transaction records touching this wallet.")
            if "value_eth" in tx_df.columns:
                fig_tx = go.Figure()
                fig_tx.add_trace(go.Scatter(
                    x=tx_df["block_timestamp"],
                    y=tx_df["value_eth"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=["#38bdf8" if d == "in" else "#f97316" for d in tx_df["direction"]],
                        opacity=0.8,
                    ),
                    text=[f"{d.upper()}: {v:.4f} ETH<br>Counterparty: {short(c)}" for d, v, c in zip(tx_df["direction"], tx_df["value_eth"], tx_df["counterparty"], strict=False)],
                    hovertemplate="%{text}<extra></extra>",
                ))
                chart_layout(fig_tx, height=240, xaxis_title="Timestamp", yaxis_title="ETH Value")
                st.plotly_chart(fig_tx, width="stretch")

            display_cols = [c for c in ["block_timestamp", "direction", "counterparty", "value_eth", "tx_hash"] if c in tx_df.columns]
            st.dataframe(
                tx_df[display_cols].head(50),
                width="stretch",
                hide_index=True,
            )

    with tab_feat:
        st.markdown("**Engineered Features & Cohort Percentiles**")
        if features.empty:
            st.info("Wallet features artifact not found.")
        else:
            w_feat = features[features["address"] == addr]
            if w_feat.empty:
                st.info("No feature row found for this address.")
            else:
                w_row = w_feat.iloc[0]
                feat_rows = []
                for col in FEATURE_FIELDS:
                    if col in features.columns:
                        val = w_row[col]
                        pct_val = (features[col] <= val).mean() * 100.0 if len(features) > 0 else 50.0
                        feat_rows.append({
                            "Feature": col,
                            "Value": fmt(val, 4),
                            "Percentile": f"{pct_val:.1f}%",
                        })
                st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)

    with tab_nb:
        st.markdown("**Direct Graph Neighbors (1-Hop Counterparties)**")
        if edges.empty:
            st.info("Graph edges artifact not available.")
        else:
            nb_edges = edges[(edges["from_address"] == addr) | (edges["to_address"] == addr)]
            if nb_edges.empty:
                st.info("No neighbor edges found in graph.")
            else:
                st.caption(f"Total {len(nb_edges)} active connection(s) in transaction graph.")
                st.dataframe(nb_edges.head(50), width="stretch", hide_index=True)

    with tab_raw:
        st.markdown("**Full Case File Export**")
        case_file = {
            "address": addr,
            "score": score_val,
            "raw_score": raw_val,
            "tier": tier_val,
            "top_driver": driver_val,
            "trace": trace,
        }
        st.json(case_file)
        st.download_button(
            "⬇️ Download Case File (JSON)",
            data=json.dumps(case_file, indent=2, default=str).encode("utf-8"),
            file_name=f"case_file_{addr[:10]}.json",
            mime="application/json",
        )


# ── view: Network ─────────────────────────────────────────────────────────────
def render_network(bundle: dict[str, Any]) -> None:
    scores: pd.DataFrame = bundle["scores"]
    edges: pd.DataFrame = bundle["edges"]
    community_map: dict[str, int] = bundle["community_map"]
    dbscan_map: dict[str, int] = bundle["dbscan_map"]

    st.markdown('<div class="cl-kicker">Graph Topology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Transaction Subgraph & Community Structure</h1>'
        '<p>Interactive visualization of transaction flow, ego networks, and modular Louvain communities.</p></div>',
        unsafe_allow_html=True,
    )

    if edges.empty:
        st.warning("Graph edges artifact (`graph_edges.parquet`) is empty or missing.")
        return

    # Connectivity is computed up front so the page can default to a wallet
    # that actually has a neighbourhood worth showing.
    endpoints = pd.concat([edges["from_address"], edges["to_address"]], ignore_index=True)
    degree = endpoints.value_counts()
    score_lookup = dict(zip(scores["address"], scores["score"], strict=False))
    tier_lookup = dict(zip(scores["address"], scores["tier"], strict=False))
    flag_lookup = dict(zip(scores["address"], scores["primary_flag"], strict=False))

    def neighbours(address: str) -> set[str]:
        out = set(edges.loc[edges["from_address"] == address, "to_address"])
        inbound = set(edges.loc[edges["to_address"] == address, "from_address"])
        return out | inbound

    def default_focus_for(selected: str) -> str:
        """Keep an explicitly chosen wallet only if it has a usable ego network."""
        if selected and len(neighbours(selected)) >= 3:
            return selected
        pool = scores[scores["tier"].isin(["high", "medium"])]
        pool = pool[pool["address"].isin(degree.index)]
        if pool.empty:
            pool = scores[scores["address"].isin(degree.index)]
        if pool.empty:
            return selected or scores.iloc[0]["address"]
        ranked = pool.assign(connections=pool["address"].map(degree))
        return str(ranked.sort_values(["connections", "score"], ascending=False).iloc[0]["address"])

    selected = st.session_state.selected_wallet
    if st.session_state.get("net_focus_for") != selected:
        st.session_state["net_focus_for"] = selected
        st.session_state["net_focus"] = default_focus_for(selected)

    hubs = degree.head(10).index.tolist()
    hub_labels: dict[str, str] = {}
    for hub in hubs:
        label = f"{short(hub, 6, 4)} · {int(degree.get(hub, 0))} links · {tier_lookup.get(hub, 'low')}"
        while label in hub_labels:
            label += " "
        hub_labels[label] = hub

    def apply_hub() -> None:
        pick = st.session_state.get("net_hub_pick")
        if pick in hub_labels:
            st.session_state["net_focus"] = hub_labels[pick]

    c1, c2, c3, c4, c5 = st.columns([2.2, 2.1, 0.9, 1.2, 1.6])
    with c1:
        focus_addr = st.text_input("Focal Address (0x...)", key="net_focus").strip().lower()
    with c2:
        st.selectbox(
            "Or pick a hub",
            options=list(hub_labels),
            index=None,
            placeholder="Most connected…",
            key="net_hub_pick",
            on_change=apply_hub,
        )
    with c3:
        max_hops = st.selectbox("Max Hops", options=[1, 2], index=1)
    with c4:
        max_nodes = st.slider("Max Subgraph Nodes", min_value=10, max_value=80, value=35)
    with c5:
        color_by = st.selectbox("Color Nodes By", options=["Risk Tier", "Primary Flag", "Louvain Community", "DBSCAN Cluster"])

    # Extract ego subgraph. Direct neighbours are always kept; second-hop nodes
    # fill the remaining budget by degree so the result is deterministic.
    hop1_out = edges[edges["from_address"] == focus_addr]["to_address"].tolist()
    hop1_in = edges[edges["to_address"] == focus_addr]["from_address"].tolist()
    sub_nodes = {focus_addr, *hop1_out, *hop1_in}

    if max_hops == 2 and len(sub_nodes) < max_nodes:
        hop2 = edges[edges["from_address"].isin(sub_nodes) | edges["to_address"].isin(sub_nodes)]
        extra = (set(hop2["from_address"]) | set(hop2["to_address"])) - sub_nodes
        ranked_extra = sorted(extra, key=lambda node: (-int(degree.get(node, 0)), node))
        sub_nodes.update(ranked_extra[: max_nodes - len(sub_nodes)])

    # Safety limit, never drop the focal node.
    if len(sub_nodes) > max_nodes:
        ranked_nodes = sorted(sub_nodes - {focus_addr}, key=lambda node: (-int(degree.get(node, 0)), node))
        sub_nodes = {focus_addr, *ranked_nodes[: max_nodes - 1]}

    focus_links = len(set(hop1_out) | set(hop1_in))
    if focus_links <= 1:
        st.caption(
            f"⚠️ Focal wallet has {focus_links} connection in the graph. "
            "Set Max Hops to 2, or pick a hub from the dropdown above."
        )

    sub_edges = edges[edges["from_address"].isin(sub_nodes) & edges["to_address"].isin(sub_nodes)]
    edge_pairs = tuple((str(r.from_address), str(r.to_address)) for r in sub_edges.itertuples())
    pos = compute_layout(tuple(sub_nodes), edge_pairs)

    # Plotly figure
    edge_x, edge_y = [], []
    for u, v in edge_pairs:
        if u in pos and v in pos:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#475569"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []

    for node in sub_nodes:
        if node in pos:
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            is_focal = (node == focus_addr)
            node_size.append(18 if is_focal else 10)

            t = tier_lookup.get(node, "low")
            s = score_lookup.get(node, 0)
            fl = flag_lookup.get(node, "")
            comm = community_map.get(node, -1)
            db = dbscan_map.get(node, -1)

            if color_by == "Risk Tier":
                node_color.append(TIER_COLORS.get(t, "#64748b"))
            elif color_by == "Primary Flag":
                node_color.append(FLAG_COLORS.get(fl, "#64748b"))
            elif color_by == "Louvain Community":
                node_color.append(COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)] if comm >= 0 else "#64748b")
            else:
                node_color.append("#ef4444" if db == -1 else "#38bdf8")

            node_text.append(
                f"Address: {node}<br>Tier: {t.upper()} (Score: {s})<br>"
                f"Flag: {fl or 'None'}<br>Community: {comm}<br>DBSCAN: {db}"
            )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color="#ffffff")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    chart_layout(
        fig, height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(f"Rendered subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} directed transaction edges.")

    # Subgraph Table & Export
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            "⬇️ Export Subgraph Edges (CSV)",
            data=sub_edges.to_csv(index=False).encode("utf-8"),
            file_name="subgraph_edges.csv",
            mime="text/csv",
        )
    with col_exp2:
        if st.button("Inspect Focal Wallet in Dossier"):
            if focus_addr in set(scores["address"]):
                navigate("Wallet dossier", focus_addr)
            else:
                st.warning("Focal address is not a scored wallet.")


# ── view: Patterns ────────────────────────────────────────────────────────────
def render_patterns(bundle: dict[str, Any]) -> None:
    flags: pd.DataFrame = bundle["flags"]
    scores: pd.DataFrame = bundle["scores"]

    st.markdown('<div class="cl-kicker">Algorithm Signals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Anomaly & Pattern Detection Intelligence</h1>'
        '<p>Detailed inspection of graph heuristic rules, unsupervised DBSCAN clusters, and Louvain modularity.</p></div>',
        unsafe_allow_html=True,
    )

    t_r1, t_r2, t_r3, t_r4, t_r5, t_db, t_louv = st.tabs([
        "R1 · Circular Flow",
        "R2 · Outbound Fan-Out",
        "R3 · Sanctioned 1-Hop",
        "R4 · PageRank Top 5%",
        "R5 · Apriori Rules",
        "DBSCAN Density",
        "Louvain Communities",
    ])

    def render_pattern_table(sig_name: str, title: str, desc: str) -> None:
        st.markdown(f"### {title}")
        st.caption(desc)
        sub_flags = flags[flags["signal_name"] == sig_name] if not flags.empty and "signal_name" in flags.columns else pd.DataFrame()
        if sub_flags.empty:
            st.info(f"No wallets triggered signal `{sig_name}` in current run.")
        else:
            st.markdown(f"**{len(sub_flags):,} wallets flagged**")
            merged = sub_flags.merge(scores[["address", "score", "tier"]], on="address", how="left")
            st.dataframe(merged.head(100), width="stretch", hide_index=True)

    with t_r1:
        render_pattern_table("cycle_member", "R1: Circular Fund Flow Detection", "Directed cycles of length 2–5 with transfer amounts within ±2% inside a 24-hour window.")
    with t_r2:
        render_pattern_table("fanout", "R2: Scatter–Gather Smurf Motif", "Wallets scattering to ≥20 counterparties whose intermediaries forward funds onwards (scatter precedes gather, 72-hour window).")
    with t_r3:
        render_pattern_table("sanctioned_1hop", "R3: Sanctioned Entity Proximity", "Wallets with ≥0.01 ETH direct exposure to a frozen OFAC/sanctioned seed; severity scales with exposure fraction.")
    with t_r4:
        render_pattern_table("pagerank_top5", "R4: Continuous PageRank Centrality", "Wallets in the continuous value-weighted PageRank tail (≥90th percentile); known hubs excluded.")
    with t_r5:
        render_pattern_table("apriori_match", "R5: Apriori Association Rule Matches", "Wallets satisfying frequent behavioural itemset rules learned from flagged training examples.")
    with t_db:
        st.markdown("### DBSCAN Density-Based Outlier Detection")
        st.caption("Unsupervised clustering in standardized feature space. Points labelled noise (-1) are behavioral outliers.")
        noise_flags = flags[flags["signal_name"] == "dbscan_noise"] if not flags.empty and "signal_name" in flags.columns else pd.DataFrame()
        cluster_flags = flags[flags["signal_name"] == "dbscan_cluster"] if not flags.empty and "signal_name" in flags.columns else pd.DataFrame()

        m1, m2 = st.columns(2)
        with m1:
            metric_card("DBSCAN Noise Outliers", f"{len(noise_flags):,}", "Feature outliers (min_samples=5, eps=1.5)")
        with m2:
            n_clusters = cluster_flags["signal_value"].nunique() if not cluster_flags.empty else 0
            metric_card("Dense Clusters Formed", f"{n_clusters:,}", "Dense behavioral clusters")
        if not noise_flags.empty:
            st.markdown("**Noise Outlier Wallets**")
            st.dataframe(noise_flags.head(50), width="stretch", hide_index=True)
    with t_louv:
        st.markdown("### Louvain Graph Modularity Communities")
        st.caption("Community detection partitioning the transaction graph into modular, densely-connected clusters.")
        comm_flags = flags[flags["signal_name"] == "louvain_community"] if not flags.empty and "signal_name" in flags.columns else pd.DataFrame()
        if comm_flags.empty:
            st.info("No Louvain community signals recorded.")
        else:
            comm_sizes = comm_flags["signal_value"].value_counts()
            metric_card("Communities Discovered", f"{len(comm_sizes):,}")
            st.markdown("**Top Community Sizes**")
            st.bar_chart(comm_sizes.head(15))


# ── view: Evaluation ──────────────────────────────────────────────────────────
def render_evaluation(bundle: dict[str, Any]) -> None:
    evaluation: dict[str, Any] = bundle["evaluation"]
    ablation: pd.DataFrame = bundle["ablation"]
    st.markdown('<div class="cl-kicker">Model Validation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Performance Evaluation & Rule Ablation</h1>'
        '<p>Empirical metrics on held-out test split, operating thresholds, and leave-one-rule-out impact.</p></div>',
        unsafe_allow_html=True,
    )

    if not evaluation or evaluation.get("insufficient", False):
        st.warning("Held-out evaluation was marked as insufficient or is unavailable.")
        if evaluation:
            st.json(evaluation)
        return

    st.info(
        "📌 **Methodology Note**: " + str(evaluation.get("note", "P/R/F1 are sample metrics on a case-control sample."))
    )

    op_points = evaluation.get("operating_points", {})
    high_op = op_points.get("high_60", {})
    med_op = op_points.get("medium_25", {})

    st.markdown("### Operating Points Comparison")
    c_high, c_med = st.columns(2)

    with c_high:
        st.markdown("#### High Risk Threshold (Score ≥ 60)")
        p_h = high_op.get("precision", 0.0)
        r_h = high_op.get("recall", 0.0)
        f1_h = high_op.get("f1", 0.0)
        tp_h, fp_h = high_op.get("tp", 0), high_op.get("fp", 0)
        tn_h, fn_h = high_op.get("tn", 0), high_op.get("fn", 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", pct(p_h))
        m2.metric("Recall", pct(r_h))
        m3.metric("F1 Score", pct(f1_h))

        matrix_h = [[tn_h, fp_h], [fn_h, tp_h]]
        fig_h = go.Figure(go.Heatmap(
            z=matrix_h,
            x=["Pred Neg", "Pred Pos"],
            y=["Actual Neg", "Actual Pos"],
            text=[[str(v) for v in r] for r in matrix_h],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        chart_layout(fig_h, height=220)
        fig_h.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_h, width="stretch")

    with c_med:
        st.markdown("#### Medium Risk Threshold (Score ≥ 25)")
        p_m = med_op.get("precision", 0.0)
        r_m = med_op.get("recall", 0.0)
        f1_m = med_op.get("f1", 0.0)
        tp_m, fp_m = med_op.get("tp", 0), med_op.get("fp", 0)
        tn_m, fn_m = med_op.get("tn", 0), med_op.get("fn", 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", pct(p_m))
        m2.metric("Recall", pct(r_m))
        m3.metric("F1 Score", pct(f1_m))

        matrix_m = [[tn_m, fp_m], [fn_m, tp_m]]
        fig_m = go.Figure(go.Heatmap(
            z=matrix_m,
            x=["Pred Neg", "Pred Pos"],
            y=["Actual Neg", "Actual Pos"],
            text=[[str(v) for v in r] for r in matrix_m],
            texttemplate="%{text}",
            colorscale="Greens",
            showscale=False,
        ))
        chart_layout(fig_m, height=220)
        fig_m.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_m, width="stretch")

    st.markdown("### Leave-One-Rule-Out Ablation Study")
    st.caption("Quantifies the marginal predictive value of each individual rule by measuring test F1 degradation when ablated.")
    if ablation.empty:
        st.info("Ablation study artifact (`ablation.csv`) not found.")
    else:
        st.dataframe(ablation, width="stretch", hide_index=True)
        if "delta_f1" in ablation.columns and "rule" in ablation.columns:
            fig_abl = go.Figure(go.Bar(
                x=ablation["rule"],
                y=ablation["delta_f1"],
                marker_color=["#ef4444" if v < 0 else "#22c55e" for v in ablation["delta_f1"]],
                text=[f"{v:+.3f}" for v in ablation["delta_f1"]],
                textposition="outside",
            ))
            chart_layout(fig_abl, height=260, xaxis_title="Ablated Rule", yaxis_title="Δ F1 Score")
            st.plotly_chart(fig_abl, width="stretch")


# ── view: Data & export ───────────────────────────────────────────────────────
def render_data_export(bundle: dict[str, Any]) -> None:
    run_info: dict[str, Any] = bundle["run_info"]

    st.markdown('<div class="cl-kicker">Artifact Inventory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cl-header"><h1>Pipeline Artifacts & System Provenance</h1>'
        '<p>Audit trail, data hashes, individual artifact downloads, and full archival export package.</p></div>',
        unsafe_allow_html=True,
    )

    # Artifact inventory table
    st.markdown("### Output Artifact Inventory")
    inv_rows = []
    for fname, desc in ARTIFACT_SPECS.items():
        fpath = OUTPUT_DIR / fname
        exists = fpath.exists()
        size_str = f"{fpath.stat().st_size / 1024:.1f} KB" if exists else "—"
        if exists and fpath.stat().st_size > 1024 * 1024:
            size_str = f"{fpath.stat().st_size / (1024 * 1024):.2f} MB"
        mtime_str = str(pd.to_datetime(fpath.stat().st_mtime, unit="s"))[:19] if exists else "—"
        inv_rows.append({
            "Artifact": fname,
            "Description": desc,
            "Size": size_str,
            "Last Modified": mtime_str,
            "Status": "✅ Present" if exists else "❌ Missing",
        })
    st.dataframe(pd.DataFrame(inv_rows), width="stretch", hide_index=True)

    st.markdown("### Provenance & Execution Hashes")
    st.json(run_info)

    st.markdown("### Export Bundle")
    stamp = _stamp(OUTPUT_DIR)
    zip_bytes = build_export_zip(str(OUTPUT_DIR), stamp)
    st.download_button(
        "📦 Download Complete Artifact Archive (.zip)",
        data=zip_bytes,
        file_name="chainlens_artifacts_bundle.zip",
        mime="application/zip",
    )

    st.divider()
    st.markdown("### About ChainLens Architecture")
    st.markdown(
        """
        **Pipeline Design (7 Core Modules):**
        - `fetch_data.py`: Ingestion from Etherscan V2 API with local Parquet disk caching.
        - `features.py`: Feature engineering using DuckDB and train-split scaling.
        - `patterns.py`: NetworkX graph algorithms, Louvain modularity, DBSCAN clustering, and Apriori rules.
        - `scoring.py`: Heuristic rules R1–R5 composite scoring (clamped 0–100) and structured trace generation.
        - `evaluate.py`: Test-split P/R/F1, PR-AUC, and leave-one-rule-out ablation.
        - `run.py`: Orchestrator writing all output artifacts to `data/output/`.
        - `dashboard.py`: Strictly read-only Streamlit operational analyst workspace.
        """
    )


# ── main entry point ──────────────────────────────────────────────────────────
def main() -> None:
    stamp = _stamp(OUTPUT_DIR)
    bundle = load_artifacts(str(OUTPUT_DIR), stamp)

    errors = bundle.get("errors", [])
    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        st.stop()

    scores = bundle.get("scores", pd.DataFrame())
    if scores.empty:
        st.warning("No scores found. Please run the pipeline first: `uv run chainlens --mode sample`")
        st.stop()

    # Apply a view change requested by a button on the previous run. This must
    # happen before the sidebar radio is created so its widget state can be set.
    pending_view = st.session_state.pop("_pending_view", None)
    if pending_view in VIEWS:
        st.session_state.nav_radio = pending_view
        st.session_state.active_view = pending_view

    # Pre-populate session state
    if not st.session_state.selected_wallet or st.session_state.selected_wallet not in set(scores["address"]):
        high_or_med = scores[scores["tier"].isin(["high", "medium"])]
        if not high_or_med.empty:
            st.session_state.selected_wallet = high_or_med.iloc[0]["address"]
        else:
            st.session_state.selected_wallet = scores.iloc[0]["address"]

    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="cl-kicker">ChainLens</div>', unsafe_allow_html=True)
        st.markdown("### Risk Workspace")

        current_view = st.session_state.active_view
        view_idx = VIEWS.index(current_view) if current_view in VIEWS else 0
        if "nav_radio" in st.session_state:
            nav_choice = st.radio("Navigate Views", VIEWS, key="nav_radio")
        else:
            nav_choice = st.radio("Navigate Views", VIEWS, index=view_idx, key="nav_radio")
        if nav_choice != current_view:
            st.session_state.active_view = nav_choice

        st.divider()
        st.markdown("**Quick Wallet Jump**")
        with st.form("jump_form", clear_on_submit=True):
            jump = st.text_input("Enter 0x address", placeholder="0x...")
            submitted = st.form_submit_button("Open dossier")
        if submitted:
            jump = jump.strip().lower()
            if jump in set(scores["address"]):
                navigate("Wallet dossier", jump)
            else:
                st.warning("Address not found in cohort")

        st.divider()
        run_info = bundle.get("run_info", {})
        mode_val = run_info.get("mode", "sample").upper()
        st.caption(f"**Cohort Size**: {len(scores):,} wallets ({mode_val})")
        if st.session_state.watchlist:
            st.caption(f"⭐ **Watchlist**: {len(st.session_state.watchlist)} wallets")

        warnings = bundle.get("warnings", [])
        if warnings:
            with st.expander(f"⚠️ Notices ({len(warnings)})", expanded=False):
                for w in warnings:
                    st.caption(f"• {w}")

    # Render active view
    active = st.session_state.active_view
    if active == "Overview":
        render_overview(bundle)
    elif active == "Triage":
        render_triage(bundle)
    elif active == "Wallet dossier":
        render_dossier(bundle)
    elif active == "Network":
        render_network(bundle)
    elif active == "Patterns":
        render_patterns(bundle)
    elif active == "Evaluation":
        render_evaluation(bundle)
    elif active == "Data & export":
        render_data_export(bundle)


if __name__ == "__main__":
    main()
