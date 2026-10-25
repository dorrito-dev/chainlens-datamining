"""ChainLens dashboard.

Read-only Streamlit app. It loads the artifacts written by run.py and never
recomputes features, flags, scores, or evaluation metrics. The network tab uses
NetworkX only for a spring layout of an already-computed edge list.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import math
import os
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
TIER_ICONS = {"high": "🔴", "medium": "🟠", "low": "⚪"}

st.set_page_config(
    page_title="ChainLens · Ethereum wallet risk",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    .cl-hero { padding: 1.1rem 1.3rem; border: 1px solid #1e293b;
               border-radius: 14px; background: linear-gradient(120deg, #0f172a, #111827); }
    .cl-hero h1 { margin: 0; font-size: 1.8rem; letter-spacing: -0.02em; }
    .cl-hero p { margin: .35rem 0 0; color: #94a3b8; }
    .cl-note { color: #94a3b8; font-size: .9rem; }
    .cl-badge { display:inline-block; padding:.12rem .5rem; border-radius:999px;
                font-size:.75rem; font-weight:600; }
    .cl-high { background:#7f1d1d; color:#fecaca; }
    .cl-medium { background:#78350f; color:#fde68a; }
    .cl-low { background:#1e293b; color:#cbd5e1; }
    div[data-testid="stSidebar"] { border-right: 1px solid #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── data loading ──────────────────────────────────────────────────────────────
def _stamp(directory: Path) -> float:
    files = [
        "scores.parquet", "traces.parquet", "flags.parquet", "graph_edges.parquet",
        "wallet_features.parquet", "evaluation.json", "ablation.csv", "run_info.json",
        "labels.parquet", "splits.parquet", "known_addresses.parquet",
    ]
    values = []
    for name in files:
        path = directory / name
        if path.exists():
            values.append(path.stat().st_mtime)
    return max(values) if values else 0.0


@st.cache_data(show_spinner=False)
def load_artifacts(output_dir: str, stamp: float) -> dict[str, Any]:
    del stamp  # cache key only
    root = Path(output_dir)
    data: dict[str, Any] = {"root": root, "missing": []}

    def parquet(name: str, required: bool = True) -> pd.DataFrame:
        path = root / name
        if not path.exists():
            if required:
                data["missing"].append(name)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def csv(name: str) -> pd.DataFrame:
        path = root / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def js(name: str) -> dict[str, Any]:
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    scores = parquet("scores.parquet")
    traces = parquet("traces.parquet")
    if not traces.empty and "trace_json" in traces.columns:
        traces["trace"] = traces["trace_json"].map(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
        traces = traces.drop(columns=["trace_json"])
    features = parquet("wallet_features.parquet", required=False)
    flags = parquet("flags.parquet", required=False)
    edges = parquet("graph_edges.parquet", required=False)
    labels = parquet("labels.parquet", required=False)
    splits = parquet("splits.parquet", required=False)
    known = parquet("known_addresses.parquet", required=False)

    data.update({
        "scores": scores,
        "traces": traces,
        "features": features,
        "flags": flags,
        "edges": edges,
        "labels": labels,
        "splits": splits,
        "known": known,
        "evaluation": js("evaluation.json"),
        "run_info": js("run_info.json"),
        "ablation": csv("ablation.csv"),
    })
    return data


def short(address: str) -> str:
    return f"{address[:6]}…{address[-4:]}" if isinstance(address, str) and len(address) > 12 else str(address)


def tier_badge(tier: str) -> str:
    klass = f"cl-{tier}" if tier in TIER_COLORS else "cl-low"
    return f'<span class="cl-badge {klass}">{TIER_ICONS.get(tier, "⚪")} {tier.upper()}</span>'


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "—"
        return f"{value:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def evidence_summary(rule_id: str, inputs: Any) -> str:
    if not isinstance(inputs, dict):
        return "—"
    if rule_id == "R1":
        return f"{inputs.get('cycle_count', 0)} cycle(s), SCC {inputs.get('scc_size', '—')}"
    if rule_id == "R2":
        return f"{inputs.get('unique_counterparties', '—')} counterparties, out/in {fmt(inputs.get('out_in_ratio'), 2)}"
    if rule_id == "R3":
        return f"seed {short(str(inputs.get('seed', '—')))} ({inputs.get('direction', '—')})"
    if rule_id == "R4":
        return f"PageRank {fmt(inputs.get('pagerank'), 6)} ≥ p95 {fmt(inputs.get('threshold'), 6)}"
    if rule_id == "R5":
        antecedent = inputs.get("antecedent") or []
        return f"{' & '.join(antecedent)} → flagged; conf {fmt(inputs.get('confidence'), 3)}"
    return json.dumps(inputs, default=str)[:120]


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


# ── load ──────────────────────────────────────────────────────────────────────
artifacts = load_artifacts(str(OUTPUT_DIR), _stamp(OUTPUT_DIR))
scores: pd.DataFrame = artifacts["scores"]
traces: pd.DataFrame = artifacts["traces"]
features: pd.DataFrame = artifacts["features"]
flags: pd.DataFrame = artifacts["flags"]
edges: pd.DataFrame = artifacts["edges"]
labels: pd.DataFrame = artifacts["labels"]
splits: pd.DataFrame = artifacts["splits"]
known: pd.DataFrame = artifacts["known"]
evaluation: dict[str, Any] = artifacts["evaluation"]
run_info: dict[str, Any] = artifacts["run_info"]
ablation: pd.DataFrame = artifacts["ablation"]

if scores.empty:
    st.markdown(
        '<div class="cl-hero"><h1>🔗 ChainLens</h1>'
        '<p>Ethereum wallet risk scoring with auditable R1–R5 rules.</p></div>',
        unsafe_allow_html=True,
    )
    st.error(f"No scores found in `{OUTPUT_DIR}`.")
    st.code("python run.py --mode sample\nstreamlit run dashboard.py", language="bash")
    st.stop()

# Convenience joins (labels are display-only; scores are already computed).
label_map: dict[str, str] = {}
if not labels.empty and "label" in labels.columns:
    label_map = (
        labels.assign(label=labels["label"].astype(str))
        .groupby("address")["label"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .to_dict()
    )
source_map: dict[str, str] = {}
if not labels.empty and "source" in labels.columns:
    source_map = (
        labels.assign(source=labels["source"].astype(str))
        .groupby("address")["source"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .to_dict()
    )
split_map: dict[str, str] = {}
if not splits.empty and "split" in splits.columns:
    split_map = dict(zip(splits["address"], splits["split"], strict=False))
known_map: dict[str, str] = {}
if not known.empty and "category" in known.columns:
    known_map = dict(zip(known["address"], known["category"], strict=False))

trace_by_address: dict[str, dict[str, Any]] = {}
if not traces.empty and "trace" in traces.columns:
    trace_by_address = dict(zip(traces["address"], traces["trace"], strict=False))

scores = scores.copy()
scores["label"] = scores["address"].map(label_map).fillna("")
scores["source"] = scores["address"].map(source_map).fillna("")
scores["split"] = scores["address"].map(split_map).fillna("")
scores["known_category"] = scores["address"].map(known_map).fillna("")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 ChainLens")
    st.caption("Auditable Ethereum wallet risk")
    source = run_info.get("data_source", "unknown")
    if source == "sample":
        st.warning("Sample fixture data — not real Etherscan results.", icon="🧪")
    else:
        st.success(f"Data source: {source}", icon="🌐")
    st.divider()
    st.markdown("**Run**")
    st.caption(
        f"Rules `{run_info.get('rules_version_label', '—')}` · "
        f"hash `{run_info.get('rules_version', '—')}`"
    )
    st.caption(f"Scaler `{run_info.get('scaler_version', '—')}`")
    st.caption(f"Commit `{run_info.get('git_commit') or '—'}`")
    st.caption(f"Generated `{run_info.get('finished_at', '—')}`")
    st.divider()
    tier_filter = st.multiselect(
        "Tier", TIER_ORDER, default=TIER_ORDER,
        format_func=lambda value: f"{TIER_ICONS[value]} {value}",
    )
    score_min, score_max = st.slider("Score range", 0, 100, (0, 100), step=1)
    search = st.text_input("Address contains", placeholder="0x…")
    if st.button("Reset filters", width="stretch"):
        st.rerun()
    st.divider()
    st.caption("Read-only: this app never recomputes scores.")
    st.caption(f"Artifacts: `{OUTPUT_DIR}`")

filtered = scores[
    scores["tier"].isin(tier_filter)
    & scores["score"].between(score_min, score_max)
]
if search.strip():
    filtered = filtered[filtered["address"].str.contains(search.strip().lower(), na=False)]
filtered = filtered.sort_values(["score", "address"], ascending=[False, True]).reset_index(drop=True)

st.markdown(
    """
    <div class="cl-hero">
      <h1>🔗 ChainLens</h1>
      <p>Ethereum wallet risk scoring · R1 circular flow · R2 fan-out · R3 sanction 1-hop ·
         R4 PageRank · R5 Apriori · DBSCAN · Louvain</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_overview, tab_wallets, tab_network, tab_eval, tab_about = st.tabs(
    ["📊 Overview", "🔎 Wallets", "🕸️ Network", "📐 Evaluation", "📘 About"]
)


# ── overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    total = len(scores)
    counts = scores["tier"].value_counts().to_dict()
    high = int(counts.get("high", 0))
    medium = int(counts.get("medium", 0))
    low = int(counts.get("low", 0))
    eval_f1 = evaluation.get("f1")
    eval_precision = evaluation.get("precision")
    eval_recall = evaluation.get("recall")
    flagged = int((scores["raw_score"] > 0).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Wallets scored", f"{total:,}")
    with c2:
        metric_card("🔴 High", f"{high:,}", "Score ≥ 60 (locked tier boundary)")
    with c3:
        metric_card("🟠 Medium", f"{medium:,}", "Score 25–59")
    with c4:
        metric_card("Flagged by ≥1 rule", f"{flagged:,}", "Raw score > 0")
    with c5:
        metric_card("Test F1", pct(eval_f1) if eval_f1 is not None else "insufficient",
                    "High-tier operating point on the leakage-safe test split")

    if evaluation.get("insufficient"):
        st.info(
            "The test split has fewer than 3 labelled positives or negatives. "
            "P/R/F1 is intentionally reported as **insufficient** rather than as a fake number.",
            icon="ℹ️",
        )

    left, right = st.columns([1.15, 1.0])
    with left:
        st.subheader("Tier distribution")
        tier_counts = scores["tier"].value_counts().reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(go.Bar(
            x=tier_counts.index, y=tier_counts.values,
            marker_color=[TIER_COLORS[t] for t in tier_counts.index],
            text=[f"{int(v):,}" for v in tier_counts.values], textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=330, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="wallets", xaxis_title="", showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Score distribution")
        fig = go.Figure(go.Histogram(
            x=scores["score"], nbinsx=40, marker_color="#38bdf8",
            hovertemplate="score %{x}: %{y}<extra></extra>",
        ))
        fig.add_vline(x=60, line_dash="dash", line_color="#ef4444",
                      annotation_text="High ≥ 60", annotation_position="top")
        fig.add_vline(x=25, line_dash="dot", line_color="#f59e0b",
                      annotation_text="Medium ≥ 25", annotation_position="top")
        fig.update_layout(
            template="plotly_dark", height=330, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="score", yaxis_title="wallets", showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.0, 1.3])
    with left:
        st.subheader("Top drivers")
        driver_counts = scores["top_driver"].fillna("none").value_counts()
        fig = go.Figure(go.Bar(
            x=driver_counts.values, y=driver_counts.index, orientation="h",
            marker_color="#a78bfa",
            text=driver_counts.values, textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=320, margin=dict(l=10, r=25, t=10, b=10),
            xaxis_title="wallets", yaxis_title="", showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Highest-scoring wallets")
        top = filtered.head(15)[["address", "score", "tier", "top_driver", "label", "known_category"]]
        st.dataframe(
            top, hide_index=True, width="stretch", height=320,
            column_config={
                "address": st.column_config.TextColumn("address", width="medium"),
                "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%d"),
                "tier": st.column_config.TextColumn("tier", width="small"),
                "top_driver": st.column_config.TextColumn("driver", width="small"),
                "label": st.column_config.TextColumn("label", width="small"),
                "known_category": st.column_config.TextColumn("known", width="small"),
            },
        )

    st.download_button(
        "⬇️ Download filtered scores (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="chainlens_scores_filtered.csv",
        mime="text/csv",
        width="content",
    )


# ── wallets ──────────────────────────────────────────────────────────────────
with tab_wallets:
    st.subheader("Wallet explorer")
    st.caption("Select a row to open its auditable rule trace. The trace is an artifact; nothing is recomputed.")

    table = filtered[["address", "score", "tier", "raw_score", "top_driver", "label", "known_category", "split"]]
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=330,
        on_select="rerun", selection_mode="single-row", key="wallet_table",
        column_config={
            "address": st.column_config.TextColumn("address", width="medium"),
            "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%d"),
            "tier": st.column_config.TextColumn("tier", width="small"),
            "raw_score": st.column_config.NumberColumn("raw", width="small"),
            "top_driver": st.column_config.TextColumn("driver", width="small"),
            "label": st.column_config.TextColumn("label", width="small"),
            "known_category": st.column_config.TextColumn("known", width="small"),
            "split": st.column_config.TextColumn("split", width="small"),
        },
    )

    selected_rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if selected_rows:
        st.session_state["selected_address"] = table.iloc[selected_rows[0]]["address"]
    if not filtered.empty and "selected_address" not in st.session_state:
        st.session_state["selected_address"] = filtered.iloc[0]["address"]

    options = filtered["address"].tolist()
    current = st.session_state.get("selected_address")
    if current not in options:
        current = options[0] if options else None
    if options:
        chosen = st.selectbox(
            "Selected wallet", options, index=options.index(current),
            format_func=lambda a: f"{short(a)} · score {int(scores.loc[scores['address'] == a, 'score'].iloc[0])}",
        )
        st.session_state["selected_address"] = chosen
    else:
        chosen = None
        st.warning("No wallets match the current filters.")

    if chosen:
        row = scores.loc[scores["address"] == chosen].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Score**  \n## {int(row['score'])} / 100")
        with c2:
            st.markdown("**Tier**  \n" + tier_badge(str(row["tier"])), unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Raw score**  \n## {int(row['raw_score'])}")
        with c4:
            st.markdown(f"**Top driver**  \n## {row['top_driver'] or '—'}")

        meta_cols = st.columns(4)
        meta_cols[0].caption(f"Address `{chosen}`")
        meta_cols[1].caption(f"Label: {row['label'] or '—'}")
        meta_cols[2].caption(f"Source: {row['source'] or '—'}")
        meta_cols[3].caption(f"Split: {row['split'] or '—'} | Known: {row['known_category'] or '—'}")

        trace = trace_by_address.get(chosen)
        if not trace:
            st.warning("No trace artifact for this wallet.")
        else:
            st.markdown(
                f"Down-weight applied: **{trace['downweight']['applied']}** · "
                f"factor **{trace['downweight']['factor']}** · "
                f"clamped: **{trace['clamped']}** · "
                f"rules `{trace['rules_version']}`"
            )
            rules = trace["rules"]
            trace_rows = []
            for rule_id in ("R1", "R2", "R3", "R4", "R5"):
                rule = rules[rule_id]
                trace_rows.append({
                    "rule": rule_id,
                    "name": rule["name"],
                    "hit": "✅" if rule["hit"] else "—",
                    "weight": rule["weight"],
                    "contribution": rule["contribution"],
                    "evidence": evidence_summary(rule_id, rule.get("inputs")),
                })
            st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")

            contributions = pd.DataFrame([
                {"rule": f"{rule_id} · {rules[rule_id]['name']}", "contribution": rules[rule_id]["contribution"]}
                for rule_id in ("R1", "R2", "R3", "R4", "R5")
            ])
            fig = go.Figure(go.Bar(
                x=contributions["contribution"],
                y=contributions["rule"],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#334155" for v in contributions["contribution"]],
                text=contributions["contribution"], textposition="outside",
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=10, r=30, t=10, b=10),
                xaxis_title="contribution", yaxis_title="", showlegend=False,
                xaxis_range=[0, max(35, int(contributions["contribution"].max()) + 8)],
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("Rule evidence (JSON)"):
                st.json({rule_id: rules[rule_id].get("inputs") for rule_id in rules})
            st.download_button(
                "⬇️ Download this trace (JSON)",
                json.dumps(trace, indent=2, default=str).encode("utf-8"),
                file_name=f"trace_{chosen}.json",
                mime="application/json",
            )

        if not features.empty:
            feature_row = features.loc[features["address"] == chosen]
            if not feature_row.empty:
                with st.expander("Feature snapshot"):
                    keep = [
                        "tx_count_out", "tx_count_in", "tx_count_total", "unique_counterparties",
                        "fanout_ratio", "out_in_ratio", "total_value_eth", "median_value_eth",
                        "net_flow_eth", "round_number_ratio", "active_days", "is_contract",
                    ]
                    snapshot = feature_row[[c for c in keep if c in feature_row.columns]].T
                    snapshot.columns = ["value"]
                    st.dataframe(snapshot, width="stretch")


# ── network ───────────────────────────────────────────────────────────────────
with tab_network:
    st.subheader("Transaction network")
    st.caption("Spring layout is visualisation only. Nodes are coloured by the already-computed tier.")

    if edges.empty:
        st.info("No `graph_edges.parquet` artifact found.")
    else:
        c1, c2, c3 = st.columns([1.4, 0.8, 0.8])
        with c1:
            focus_options = scores["address"].tolist()
            default_focus = st.session_state.get("selected_address", focus_options[0])
            focus = st.selectbox(
                "Focus address", focus_options,
                index=focus_options.index(default_focus) if default_focus in focus_options else 0,
                format_func=lambda a: f"{short(a)} · {int(scores.loc[scores['address'] == a, 'score'].iloc[0])} "
                                      f"{scores.loc[scores['address'] == a, 'tier'].iloc[0]}",
                key="network_focus",
            )
        with c2:
            depth = st.radio("Hops", [1, 2], horizontal=True, index=1)
        with c3:
            max_nodes = st.slider("Max nodes", 30, 400, 180, 10)

        adjacency: dict[str, set[str]] = {}
        for row in edges.itertuples(index=False):
            adjacency.setdefault(row.from_address, set()).add(row.to_address)
            adjacency.setdefault(row.to_address, set()).add(row.from_address)

        visited = {focus}
        frontier = {focus}
        for _ in range(depth):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= adjacency.get(node, set())
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if len(visited) >= max_nodes:
                break

        score_lookup = dict(zip(scores["address"], scores["score"], strict=False))
        tier_lookup = dict(zip(scores["address"], scores["tier"], strict=False))
        driver_lookup = dict(zip(scores["address"], scores["top_driver"], strict=False))
        selected = sorted(visited, key=lambda a: (-score_lookup.get(a, 0), a))[:max_nodes]
        if focus not in selected:
            selected.append(focus)
        selected_set = set(selected)

        sub = nx.DiGraph()
        sub.add_nodes_from(selected)
        for row in edges.itertuples(index=False):
            if row.from_address in selected_set and row.to_address in selected_set:
                sub.add_edge(
                    row.from_address, row.to_address,
                    tx_count=int(row.tx_count),
                    value=float(row.total_value_eth),
                )
        if sub.number_of_nodes() == 0:
            st.warning("No edges in the selected neighbourhood.")
        else:
            pos = nx.spring_layout(sub, seed=42, k=0.9, iterations=60)
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for u, v in sub.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.7, color="#334155"),
                hoverinfo="none", showlegend=False,
            )

            node_x, node_y, node_color, node_size, node_text, node_hover = [], [], [], [], [], []
            for node in sub.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                tier = tier_lookup.get(node, "low")
                node_color.append(TIER_COLORS.get(tier, "#64748b"))
                degree = sub.degree(node)
                node_size.append(10 + min(18, degree * 1.5) + (8 if node == focus else 0))
                score = score_lookup.get(node, 0)
                node_text.append(short(node) if score >= 20 or node == focus else "")
                node_hover.append(
                    f"<b>{node}</b><br>score {int(score)} · {tier}<br>"
                    f"driver {driver_lookup.get(node) or '—'}<br>degree {degree}"
                )
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(
                    size=node_size, color=node_color,
                    line=dict(width=[2 if n == focus else 0.5 for n in sub.nodes()],
                              color=["#f8fafc" if n == focus else "#0f172a" for n in sub.nodes()]),
                ),
                text=node_text, textposition="top center",
                textfont=dict(size=9, color="#cbd5e1"),
                hovertext=node_hover, hoverinfo="text",
                showlegend=False,
            )
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                template="plotly_dark", height=620, showlegend=False,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                hovermode="closest",
            )
            st.plotly_chart(fig, width="stretch")

            legend = " · ".join(
                f'<span class="cl-badge cl-{tier}">{TIER_ICONS[tier]} {tier}</span>'
                for tier in TIER_ORDER
            )
            st.markdown(legend, unsafe_allow_html=True)
            st.caption(
                f"Showing {sub.number_of_nodes()} nodes and {sub.number_of_edges()} edges "
                f"within {depth} hop(s) of `{short(focus)}`."
            )

            neighbours = pd.DataFrame([
                {
                    "address": n,
                    "score": int(score_lookup.get(n, 0)),
                    "tier": tier_lookup.get(n, "low"),
                    "driver": driver_lookup.get(n),
                    "degree": sub.degree(n),
                }
                for n in sub.nodes()
            ]).sort_values(["score", "degree"], ascending=False)
            st.dataframe(neighbours.head(100), hide_index=True, width="stretch")


# ── evaluation ────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Held-out evaluation")
    st.caption(
        "Rules and the scaler are fitted on the train split. Test labels are never used by R5. "
        "The sample is case-control, so P/R/F1 are sample metrics, not population estimates."
    )

    if evaluation.get("insufficient"):
        st.warning(
            f"**Insufficient test labels.** Positives: {evaluation.get('test_positives', 0)}, "
            f"negatives: {evaluation.get('test_negatives', 0)}. At least 3 of each are required.",
            icon="⚠️",
        )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Precision", pct(evaluation.get("precision")))
    with c2:
        metric_card("Recall", pct(evaluation.get("recall")))
    with c3:
        metric_card("F1", pct(evaluation.get("f1")))
    with c4:
        metric_card("Average precision", pct(evaluation.get("average_precision")))

    st.caption(
        f"Test addresses: {evaluation.get('test_addresses', 0)} · "
        f"positives: {evaluation.get('test_positives', 0)} · "
        f"negatives: {evaluation.get('test_negatives', 0)} · "
        f"dropped (no activity): {evaluation.get('dropped_positives', 0)} positives, "
        f"{evaluation.get('dropped_negatives', 0)} negatives."
    )

    left, right = st.columns([0.9, 1.2])
    with left:
        st.markdown("**Confusion matrix** at High ≥ 60")
        tp, fp = int(evaluation.get("tp", 0) or 0), int(evaluation.get("fp", 0) or 0)
        tn, fn = int(evaluation.get("tn", 0) or 0), int(evaluation.get("fn", 0) or 0)
        matrix = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=["pred low/med", "pred high"],
            y=["actual negative", "actual positive"],
            text=[[str(v) for v in row] for row in matrix],
            texttemplate="%{text}", colorscale="Blues", showscale=False,
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("**Leave-one-rule-out ablation**")
        if ablation.empty:
            st.info("No ablation artifact.")
        else:
            show = ablation.copy()
            for col in ("baseline_f1", "ablated_f1", "delta_f1"):
                if col in show.columns:
                    show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
            st.dataframe(
                show, hide_index=True, width="stretch",
                column_config={
                    "rule": st.column_config.TextColumn("rule", width="small"),
                    "affected_wallets": st.column_config.NumberColumn("wallets", width="small"),
                    "baseline_f1": st.column_config.NumberColumn("base F1", format="%.3f"),
                    "ablated_f1": st.column_config.NumberColumn("without", format="%.3f"),
                    "delta_f1": st.column_config.NumberColumn("Δ F1", format="%.3f"),
                    "tier_changes": st.column_config.NumberColumn("tier Δ", width="small"),
                },
            )
            delta = ablation.dropna(subset=["delta_f1"]) if "delta_f1" in ablation.columns else pd.DataFrame()
            if delta.empty:
                st.info("Δ F1 is undefined because the test split is too small. Ablation still shows tier changes.")
            else:
                fig = go.Figure(go.Bar(
                    x=delta["rule"], y=delta["delta_f1"],
                    marker_color=["#ef4444" if v < 0 else "#22c55e" for v in delta["delta_f1"]],
                    text=[f"{v:+.3f}" for v in delta["delta_f1"]], textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_title="Δ F1 when removed", xaxis_title="",
                )
                st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("**Reproducibility metadata**")
    st.json(run_info)
    st.download_button(
        "⬇️ Download evaluation.json",
        json.dumps(evaluation, indent=2, default=str).encode("utf-8"),
        file_name="evaluation.json",
        mime="application/json",
    )


# ── about ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("How ChainLens works")
    st.markdown(
        """
        **Pipeline (7 files).** `fetch_data.py` → `features.py` → `patterns.py` →
        `scoring.py` → `evaluate.py`, orchestrated by `run.py`; `dashboard.py` reads
        only `data/output/`.

        | Rule | Signal | Weight | Plain-English meaning |
        |---|---|---:|---|
        | R1 | circular flow | 25 | A short 3–5 hop cycle where edge values are within 5% and the cycle fits in 24h |
        | R2 | fan-out | 20 | Many distinct outbound counterparties with a high outbound share |
        | R3 | sanctioned 1-hop | 30 | One hop from a train-split sanctioned seed (both directions) |
        | R4 | PageRank top 5% | 15 | High value-weighted PageRank in the transaction graph |
        | R5 | Apriori match | 20 | Behavioural itemset match to train-split flagged wallets (confidence ≥ 0.8) |

        **Tiers.** Low 0–24 · Medium 25–59 · High 60–100. The raw maximum is 110 and is
        hard-clamped to 100. Known exchange/bridge/mixer/contract addresses are multiplied
        by 0.3 **before** clamping and tiering.

        **What this is not.** A score is not proof of wrongdoing. R4 is a prominence signal,
        R5 is an association signal, and DBSCAN noise is a separate outlier flag. Use the
        trace, not the number alone.

        **Data.** Real mode uses Etherscan V2 normal transactions, per-address most-recent
        500, a per-address cache, and a train/test split. Negatives are sampled Kaggle
        `flag=0` addresses; context addresses come from random blocks. The sample is
        case-control, so precision/recall are sample metrics.

        **Deviations from the deck.** R2 is implemented as the outbound fan-out gate
        (`unique_counterparties ≥ 20`, `tx_count_out ≥ 20`, `fanout_ratio ≥ 0.5`), not the
        deck's `out_in_ratio > 5` + counterparties > 50. The deck version is recorded in
        each R2 trace as `r2_deck_equivalent`. On the current locked weights, R1+R2+R4 =
        60, not the slide-21 "65".

        **Ethics.** Public-chain labels are noisy and stale; OFAC/darklist snapshots have
        dates; Kaggle negatives are not verified non-fraud. Treat outputs as investigative
        leads with human review.
        """
    )
    st.code(
        "python run.py --mode sample\n"
        "python run.py --mode real --max-addresses 400\n"
        "streamlit run dashboard.py",
        language="bash",
    )
