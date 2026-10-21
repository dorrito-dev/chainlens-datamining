"""ChainLens Phase 5 Streamlit shell.

The wallet breakdown renders the persisted score trace. The graph tab renders a
NetworkX neighborhood; Neo4j is an optional showcase mirror and is deliberately
not imported by this app.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text
from streamlit_agraph import Config, Edge, Node, agraph

from chainlens.config import load_data_config, project_path
from chainlens.db import get_pg_engine
from chainlens.detection import load_detection_config
from chainlens.features import get_scaled_feature_columns
from chainlens.graph import build_graph, neighborhood_subgraph

TIER_COLORS = {
    "high": "#d62728",
    "medium": "#ffb300",
    "low": "#9e9e9e",
}
TIER_ROW_STYLES = {
    "high": "background-color: #d62728; color: white;",
    "medium": "background-color: #ffb300; color: #1a1a1a;",
    "low": "background-color: #9e9e9e; color: #1a1a1a;",
}
COMMUNITY_COLORS = [
    "#1f77b4",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#aec7e8",
    "#ffbb78",
]


@st.cache_resource(show_spinner="Building NetworkX graph…")
def _cached_graph(scaler_version: str, parquet_stamp: int):
    engine = get_pg_engine()
    return build_graph(engine, scaler_version=scaler_version)


def _short(address: str, length: int = 12) -> str:
    return f"{address[:6]}…{address[-4:]}" if len(address) > length else address


def _engine_or_error():
    try:
        engine = get_pg_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # pragma: no cover - exercised by manual failure mode
        st.error(
            "PostgreSQL is unavailable. Start it with `make up` and retry. "
            f"Details: {exc.__class__.__name__}"
        )
        st.stop()


@st.cache_data(ttl=10, show_spinner=False)
def _ranked_frame() -> pd.DataFrame:
    # Test/demo-only switch for the empty-state assertion; normal runs never set it.
    if os.getenv("CHAINLENS_FORCE_EMPTY_SCORES") == "1":
        return pd.DataFrame(
            columns=[
                "address",
                "score",
                "tier",
                "top_driver",
                "rules_version",
                "tx_count_total",
                "flags",
            ]
        )
    engine = get_pg_engine()
    query = text(
        """
        SELECT
            rs.address,
            rs.score,
            rs.tier,
            rs.top_driver,
            rs.rules_version,
            af.tx_count_total,
            COALESCE(string_agg(DISTINCT flags.signal_name, ', '), '') AS flags
        FROM risk_scores rs
        JOIN address_features af ON af.address = rs.address
        LEFT JOIN address_flags flags ON flags.address = rs.address
        GROUP BY
            rs.address, rs.score, rs.tier, rs.top_driver,
            rs.rules_version, af.tx_count_total
        ORDER BY rs.score DESC, af.tx_count_total DESC, rs.address ASC
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _flag_names() -> list[str]:
    engine = get_pg_engine()
    with engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT DISTINCT signal_name FROM address_flags "
                    "ORDER BY signal_name"
                )
            ).scalars()
        )


def _addresses_with_signal(signal_name: str) -> set[str]:
    engine = get_pg_engine()
    with engine.connect() as conn:
        return set(
            conn.execute(
                text(
                    "SELECT address FROM address_flags WHERE signal_name = :signal"
                ),
                {"signal": signal_name},
            ).scalars()
        )


def _all_addresses() -> list[str]:
    engine = get_pg_engine()
    with engine.connect() as conn:
        return list(
            conn.execute(
                text("SELECT address FROM risk_scores ORDER BY address")
            ).scalars()
        )


@st.cache_data(ttl=10, show_spinner=False)
def _neo4j_link(address: str, uri: str | None) -> str | None:
    """Return a browser link only when Neo4j is reachable and curated."""
    if not uri:
        return None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            uri,
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", ""),
            ),
            connection_timeout=1.0,
        )
        try:
            driver.verify_connectivity()
            with driver.session() as session:
                exists = session.run(
                    "MATCH (a:ChainLens {address: $address}) "
                    "RETURN count(a) AS n",
                    address=address,
                ).single()["n"]
        finally:
            driver.close()
    except Exception:  # noqa: BLE001 - Neo4j is deliberately optional
        return None
    if not exists:
        return None
    return os.getenv("NEO4J_BROWSER_URL", "http://localhost:7474/browser/")


def _tier_row_style(row: pd.Series) -> list[str]:
    style = TIER_ROW_STYLES.get(str(row.get("tier")), "")
    return [style if column == "tier" else "" for column in row.index]


def _render_ranked() -> None:
    frame = _ranked_frame()
    if frame.empty:
        st.info("No scores yet — run `make detect` then `make score`.")
        return

    tiers = sorted(frame["tier"].dropna().unique())
    signals = _flag_names()
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_tiers = st.multiselect("Tier", tiers, default=tiers)
    with col2:
        min_score = st.slider(
            "Minimum score",
            min_value=0,
            max_value=max(100, int(frame["score"].max())),
            value=0,
        )
    with col3:
        signal = st.selectbox("Signal", ["(all)", *signals])

    view = frame[frame["tier"].isin(selected_tiers) & (frame["score"] >= min_score)]
    if signal != "(all)":
        addresses = _addresses_with_signal(signal)
        view = view[view["address"].isin(addresses)]
    if view.empty:
        st.info("No addresses match the selected tier, score, and signal filters.")
        return

    display = view.reset_index(drop=True).copy()
    display["address"] = display["address"].map(_short)
    display = display.rename(
        columns={
            "address": "address",
            "score": "score",
            "tier": "tier",
            "top_driver": "top driver",
            "flags": "top flags",
            "tx_count_total": "tx count",
            "rules_version": "rules",
        }
    )
    st.caption(
        f"{len(display):,} of {len(frame):,} addresses · click a row to open "
        "that wallet in the breakdown tab"
    )
    columns = [
        "address",
        "score",
        "tier",
        "top driver",
        "top flags",
        "tx count",
        "rules",
    ]
    event = st.dataframe(
        display[columns].style.apply(_tier_row_style, axis=1),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ranked_table",
    )
    selected_rows = getattr(getattr(event, "selection", None), "rows", [])
    if selected_rows:
        chosen = str(view.iloc[selected_rows[0]]["address"])
        if chosen != st.session_state.get("selected_address"):
            st.session_state["selected_address"] = chosen
            st.rerun()


def _wallet_details(address: str) -> None:
    engine = get_pg_engine()
    with engine.connect() as conn:
        feature = pd.read_sql(
            text("SELECT * FROM address_features WHERE address = :address"),
            conn,
            params={"address": address},
        )
        flags = pd.read_sql(
            text(
                "SELECT signal_name, signal_value, evidence, created_at "
                "FROM address_flags WHERE address = :address "
                "ORDER BY signal_name"
            ),
            conn,
            params={"address": address},
        )
        score = pd.read_sql(
            text("SELECT * FROM risk_scores WHERE address = :address"),
            conn,
            params={"address": address},
        )
        trace_frame = pd.read_sql(
            text(
                """
                SELECT st.trace
                FROM score_trace st
                JOIN risk_scores rs
                  ON rs.address = st.address
                 AND rs.rules_version = st.rules_version
                WHERE st.address = :address
                """
            ),
            conn,
            params={"address": address},
        )
    if os.getenv("CHAINLENS_FORCE_MISSING_TRACE") == "1":
        trace_frame = trace_frame.iloc[0:0]

    st.subheader("Raw features")
    feature_table = feature.T.reset_index()
    feature_table.columns = ["feature", "value"]
    feature_table["value"] = feature_table["value"].astype(str)
    st.dataframe(feature_table, width="stretch", hide_index=True)

    st.subheader("Flags")
    if flags.empty:
        st.caption("No detection flags for this address.")
    else:
        display = flags.copy()
        display["evidence"] = display["evidence"].map(
            lambda value: json.dumps(value, indent=2, sort_keys=True, default=str)
            if isinstance(value, dict)
            else str(value)
        )
        st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Score trace")
    if score.empty:
        st.caption("No risk_scores row for this address.")
    elif score.iloc[0]["rules_version"] == "seed-phase4" or trace_frame.empty:
        version = str(score.iloc[0]["rules_version"])
        st.info(
            f"Trace not found for rules_version {version} — re-run `make score`."
        )
    else:
        trace = trace_frame.iloc[0]["trace"]
        if isinstance(trace, str):
            trace = json.loads(trace)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Score", int(trace["score"]))
        metric_columns[1].metric("Tier", str(trace["tier"]))
        metric_columns[2].metric("Raw score", int(trace["raw_score"]))
        metric_columns[3].metric("Top driver", trace["top_driver"] or "none")

        st.caption(
            f"rules={trace['rules_version']} ({trace['rules_version_label']}) · "
            f"scaler={trace['scaler_version']} · engine={trace['engine_version']} · "
            f"adjusted={trace['adjusted_raw_score']:.2f} · "
            f"clamped={trace['clamped']}"
        )
        downweight = trace["downweight"]
        if downweight["applied"]:
            st.warning(
                "Known-address downweight applied: "
                f"category={downweight['category']}, factor={downweight['factor']}, "
                f"{downweight['pre_downweight_raw']} → "
                f"{downweight['post_downweight_raw']:.2f}"
            )
        else:
            st.caption("Known-address downweight: not applied (factor 1.0).")

        rules = trace["rules"]
        rows = [
            {
                "rule": rule_id,
                "name": rule["name"],
                "hit": rule["hit"],
                "weight": rule["weight"],
                "contribution": rule["contribution"],
                "inputs": (
                    json.dumps(rule["inputs"], indent=2, sort_keys=True)
                    if rule["inputs"] is not None
                    else None
                ),
            }
            for rule_id, rule in rules.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.expander("Full trace (JSON)"):
            st.json(trace)
        neo4j_url = _neo4j_link(address, os.getenv("NEO4J_URI"))
        if neo4j_url:
            st.markdown(f"[Open in Neo4j Browser]({neo4j_url})")


def _node_metadata(addresses: list[str]) -> pd.DataFrame:
    engine = get_pg_engine()
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    rs.address,
                    rs.score,
                    rs.tier,
                    lc.signal_value AS louvain_community
                FROM risk_scores rs
                LEFT JOIN address_flags lc
                    ON lc.address = rs.address
                   AND lc.signal_name = 'louvain_community'
                WHERE rs.address = ANY(:addresses)
                """
            ),
            conn,
            params={"addresses": addresses},
        )


def _graph_node_color(tier: str | None, community: Any) -> str:
    if tier and tier != "unscored":
        return TIER_COLORS.get(str(tier), "#7f7f7f")
    if community is not None:
        try:
            return COMMUNITY_COLORS[int(community) % len(COMMUNITY_COLORS)]
        except (TypeError, ValueError):
            pass
    return "#7f7f7f"


def _render_graph(address: str) -> None:
    engine = get_pg_engine()
    try:
        _, scaler_version = get_scaled_feature_columns(engine)
    except Exception as exc:
        st.error(f"Scaler provenance unavailable: {exc}")
        return
    clean_path = project_path(load_data_config()["ingest"]["clean_parquet"])
    if not clean_path.exists():
        st.error("Clean Parquet is missing; run `make ingest` first.")
        return
    stamp = clean_path.stat().st_mtime_ns
    graph = _cached_graph(scaler_version, stamp)
    if address not in graph:
        st.warning("This address is not present in the current NetworkX graph.")
        return

    config = load_detection_config()["graph"]
    subgraph = neighborhood_subgraph(
        graph,
        address,
        hops=int(config.get("app_graph_hops", 2)),
        max_nodes=int(config.get("app_graph_node_cap", 200)),
    )
    node_addresses = subgraph["addresses"]
    metadata = _node_metadata(node_addresses).set_index("address")

    nodes = []
    for node_address in node_addresses:
        row = metadata.loc[node_address] if node_address in metadata.index else None
        tier = row["tier"] if row is not None else "unscored"
        community = row["louvain_community"] if row is not None else None
        score = int(row["score"]) if row is not None else 0
        nodes.append(
            Node(
                id=node_address,
                label=_short(node_address),
                title=f"{node_address}\ntier={tier}\nscore={score}",
                color=_graph_node_color(tier, community),
                size=14 + min(int(score), 20),
            )
        )

    edges = [
        Edge(
            source=edge["from"],
            target=edge["to"],
            label=f"{edge['tx_count']} tx",
            width=max(1.0, min(8.0, math.log1p(edge["tx_count"]))),
            title=f"{edge['tx_count']} tx · {edge['total_value_eth']:.4f} ETH",
        )
        for edge in subgraph["edges"]
    ]
    st.caption(
        f"{len(nodes)} nodes / {len(edges)} aggregated edges · "
        f"{subgraph['hops']} hops · cap {subgraph['node_cap']}"
    )
    if subgraph["cap_hit"]:
        st.warning("Node cap reached; showing the closest neighbours only.")
    agraph(
        nodes=nodes,
        edges=edges,
        config=Config(
            width=900,
            height=650,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
        ),
    )
    st.caption(
        "This view is built from NetworkX. Neo4j is an optional showcase mirror; "
        "the tab does not require it."
    )


def main() -> None:
    st.set_page_config(
        page_title="ChainLens",
        page_icon="🔍",
        layout="wide",
    )
    st.title("ChainLens")
    st.caption(
        "Phase 6 evaluation shell · locked-rule scores and audit traces · "
        "graph tab reads NetworkX; Neo4j is optional"
    )

    _engine_or_error()
    try:
        ranked = _ranked_frame()
    except Exception as exc:
        st.error(f"Could not read risk_scores: {exc.__class__.__name__}")
        st.stop()
    if ranked.empty:
        st.info("No scores yet — run `make detect` then `make score`.")
        st.stop()

    addresses = _all_addresses()
    selected_address = st.session_state.get("selected_address")
    if selected_address not in addresses:
        selected_address = addresses[0]
    focus = st.sidebar.selectbox(
        "Focus wallet",
        addresses,
        format_func=_short,
        index=addresses.index(selected_address),
    )
    st.session_state["selected_address"] = focus
    st.sidebar.caption(
        "Click a ranked-list row to jump here. Scores use locked R1–R5 "
        "weights; open the wallet breakdown for the reconstructable trace."
    )

    ranked_tab, wallet_tab, graph_tab = st.tabs(
        ["Ranked list", "Wallet breakdown", "Graph view"]
    )
    with ranked_tab:
        _render_ranked()
    with wallet_tab:
        _wallet_details(focus)
    with graph_tab:
        _render_graph(focus)


if __name__ == "__main__":
    main()
