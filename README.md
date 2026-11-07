# ChainLens 🔗

> **Blockchain Wallet Risk Intelligence Platform** — An explainable, heuristic + ML pipeline that ingests Ethereum transaction data, engineers behavioural features, detects suspicious patterns using five locked rules, and surfaces a prioritised analyst triage dashboard.

---

## Table of Contents

1. [What ChainLens Does](#1-what-chainlens-does)
2. [The Full Pipeline — Stage by Stage](#2-the-full-pipeline--stage-by-stage)
3. [Scoring: How Wallets Get a Risk Score](#3-scoring-how-wallets-get-a-risk-score)
4. [All Five Detection Rules Explained](#4-all-five-detection-rules-explained)
5. [Unsupervised Algorithms: DBSCAN, Apriori, Louvain, PageRank](#5-unsupervised-algorithms-dbscan-apriori-louvain-pagerank)
6. [Why Rules Sometimes Catch Nothing — Explained](#6-why-rules-sometimes-catch-nothing--explained)
7. [Sanctioned Entities & Community Detection](#7-sanctioned-entities--community-detection)
8. [Dashboard: Every Tab & Every Chart](#8-dashboard-every-tab--every-chart)
9. [Evaluation: How We Measure Quality](#9-evaluation-how-we-measure-quality)
10. [Quickstart](#10-quickstart)
11. [Configuration Reference](#11-configuration-reference)
12. [Output Artifacts](#12-output-artifacts)
13. [Architecture Overview](#13-architecture-overview)

---

## 1. What ChainLens Does

ChainLens is a **blockchain forensics and compliance intelligence system** for Ethereum. Given a set of wallet addresses — including known bad actors (sanctions lists, fraud reports, darklists) and known clean wallets — it:

1. **Fetches** all their on-chain transactions from the Etherscan API (or loads them from a sample fixture)
2. **Engineers** 14+ behavioural features per wallet (e.g. fan-out ratio, circular flow, round-number transfers)
3. **Detects** suspicious patterns using five transparent heuristic rules (R1–R5), two unsupervised ML algorithms (DBSCAN + Apriori), and one graph community detector (Louvain)
4. **Scores** every wallet 0–100 and assigns it to a risk tier: **High (≥ 60)**, **Medium (25–59)**, or **Low (0–24)**
5. **Evaluates** the model on a held-out test split, producing Precision / Recall / F1 and a leave-one-rule-out ablation study
6. **Visualises** everything in a read-only Streamlit analyst dashboard with 7 interactive tabs

The system is **fully explainable** — every rule hit is traced back to concrete evidence (which cycle path triggered R1, which sanctioned seed triggered R3, which itemset triggered R5). An analyst can reconstruct the entire decision from a single JSON "case file".

---

## 2. The Full Pipeline — Stage by Stage

```
Etherscan API / Sample Fixture
          │
          ▼
┌─────────────────────────────┐
│  Stage 1 · fetch_data.py    │  Ingestion
│  Rate-limited Etherscan V2  │  Per-address Parquet cache
│  client + label loading     │  Deterministic train/test split
└──────────────┬──────────────┘
               │ raw transactions.parquet
               ▼
┌─────────────────────────────┐
│  Stage 2 · features.py      │  Cleaning & Feature Engineering
│  DuckDB SQL: de-dup, filter │  14 features per wallet
│  Train-only StandardScaler  │  → wallet_features.parquet
└──────────────┬──────────────┘
               │ scaled features
               ▼
┌─────────────────────────────┐
│  Stage 3 · DuckDB warehouse │  In-memory SQL warehouse
│  (inside run.py)            │  All tables available to
│                             │  patterns, scoring, dashboard
└──────────────┬──────────────┘
               │ features + graph edges
               ▼
┌─────────────────────────────┐
│  Stage 4 · patterns.py      │  Pattern Detection
│  NetworkX DiGraph           │  R1: cycles (length 2–5)
│  DBSCAN clustering          │  R2: scatter→gather motif
│  Apriori association rules  │  R3: frozen sanctions + exposure
│  Louvain communities        │  R4: continuous PageRank
│  PageRank                   │  R5: Apriori match
└──────────────┬──────────────┘
               │ flags dict
               ▼
┌─────────────────────────────┐
│  Stage 5 · scoring.py       │  Composite Scoring
│  R1–R5 rule weights         │  raw_score = Σ (strength × weight)
│  Known-address downweight   │  score clamped 0–100
│  Audit trace JSON           │  → scores.parquet, traces.parquet
└──────────────┬──────────────┘
               │ scores + flags
               ▼
┌─────────────────────────────┐
│  Stage 6 · evaluate.py      │  Model Evaluation
│  Test-split P/R/F1          │  Leave-one-rule-out ablation
│  PR-AUC                     │  → evaluation.json, ablation.csv
└──────────────┬──────────────┘
               │ All artifacts written to data/output/
               ▼
┌─────────────────────────────┐
│  dashboard.py (read-only)   │  7-tab Streamlit workspace
│  Never re-runs pipeline     │  Pure visualisation of artifacts
└─────────────────────────────┘
```

### Key Design Decisions

| Decision | Reason |
|---|---|
| **DuckDB for feature engineering** | Handles wei→ETH decimal arithmetic without float64 overflow; deterministic SQL |
| **Train-only scaling** | Scaler is fit on the training split to prevent data leakage to the test split |
| **Locked rule weights** | Weights `[25, 20, 30, 15, 20]` are locked in `rules.yaml`; a hash guards against accidental changes |
| **Known-address downweight (0.3×)** | Exchanges, bridges, mixers, and contracts appear in many transactions but are not suspicious; their raw score is multiplied by 0.3 |
| **Dashboard never imports patterns/scoring** | The dashboard is strictly read-only; it cannot mutate results, ensuring audit integrity |

---

## 3. Scoring: How Wallets Get a Risk Score

Every wallet receives a **composite risk score from 0 to 100** based on which rules it triggers.

### Rule Weights (Locked)

| Rule | Signal | Weight | What it detects |
|---|---|---|---|
| R1 | `cycle_member` | **25 pts** | Circular fund flows (money laundering typology) |
| R2 | `fanout` | **20 pts** | Temporal scatter→gather smurf motif (structuring) |
| R3 | `sanctioned_1hop` | **30 pts** | Direct link to frozen OFAC/sanctioned address, scaled by exposure |
| R4 | `pagerank_top5` | **15 pts** | Continuous upper-tail network centrality |
| R5 | `apriori_match` | **20 pts** | Behavioural profile matching known bad actors |
| **Max raw** | | **110 pts** | (all five at full strength) |

The maximum raw score is 110 but the final score is **hard-clamped to 100**.

R1, R2 and R5 are binary. R3 and R4 carry an evidence-derived `strength` in
`[0, 1]` and contribute `weight × strength`, so a small sanctioned exposure or
a mid-tail PageRank rank earns proportionally fewer points. `raw_score` can
therefore be fractional; the final score is still rounded after downweighting.

### Tier Thresholds

```
  0 ─────────────────────────────────────────────── 100
  │           │                    │                  │
  └── Low ────┴───── Medium ───────┴──── High ────────┘
      0–24         25–59                 60–100
```

### Known-Address Downweight

If a wallet is in `config/known_addresses.json` under a category like `exchange`, `bridge`, `mixer`, or `contract`, its raw score is multiplied by **0.3** before clamping. This prevents Uniswap, Binance, and similar high-volume addresses from being falsely flagged.

### Scoring Examples

```
Example A (Full-strength binary hits):
Wallet 0xabc...def interacts directly with a sanctioned address (R3 hit, full exposure: strength = 1.0 → 30 pts)
and triggers the scatter→gather smurf motif (R2 hit: strength = 1.0 → 20 pts)

  raw_score = 30.0 + 20.0 = 50.0
  known_category = None → factor = 1.0
  final_score = min(50, 100) = 50  →  MEDIUM tier
  top_driver = R3

Example B (Continuous evidence strength & benign downweighting):
Wallet 0x123...456 has partial sanctioned exposure (R3: 25% inflow volume, strength = 0.25 → 7.5 pts)
and high PageRank centrality (R4: 94th percentile, strength = 0.40 → 6.0 pts)

  raw_score = 7.5 + 6.0 = 13.5
  known_category = None → factor = 1.0
  final_score = round(13.5) = 14  →  LOW tier

Example C (Known-entity downweighting):
Known exchange hot wallet in known_addresses.json triggers R2 fan-out (20 pts) and R4 PageRank (15 pts):
  raw_score = 35.0
  known_category = "exchange" → factor = 0.3
  adjusted_raw_score = 35.0 × 0.3 = 10.5
  final_score = round(10.5) = 11  →  LOW tier (suppressed false positive)
```

---

## 4. All Five Detection Rules Explained

### R1 · Circular Flow Detection (Weight: 25)

**What it looks for:** Directed transaction cycles of length **2–5** where money travels in a loop back to its origin.

**Real-world pattern:** In layering schemes, funds move A → B → C → A to obscure the origin. ChainLens detects this using **Strongly Connected Components (SCC)** in the transaction graph, then enumerates simple cycles within each SCC. Length-2 cycles capture the two-account self-trade pairs documented by Victor & Weintraud.

**The constraints that make it precise:**
- Cycle length must be between **2 and 5 hops** (longer cycles are too common)
- All edge values within the cycle must be within **±2% of each other** (ETH transfers differ only by gas-fee noise)
- The entire cycle must complete within **24 hours**
- SCCs larger than 5,000 nodes are skipped for performance

---

### R2 · Outbound Fan-Out / Smurf Motif (Weight: 20)

**What it looks for:** The **scatter→gather** motif from Starnini et al.: a wallet disperses funds to many intermediaries, and those intermediaries forward funds onwards **after** the scatter, within a 72-hour window.

**The scatter gate (all must pass):**
1. `unique_counterparties >= 20`
2. `tx_count_out >= 20`
3. `fanout_ratio >= 0.5`

**The gather gate (must also pass):**
- At least 2 intermediaries (or 10% of scattering counterparties) forward funds after the scatter edge
- At least 1 distinct collector receives those forwarded funds
- Gather edges occur after their scatter edge and within 72 hours

A wallet that only scatters is a *candidate*, not a confirmed smurf; it receives no R2 points.

---

### R3 · Sanctioned Entity 1-Hop (Weight: 30)

**What it looks for:** Any wallet that directly sent to, or directly received from, a **frozen OFAC-sanctioned address** (or a labelled sanctioned seed).

**Seeds are split-independent:** `config/sanctions.json` is a frozen OFAC snapshot, unioned with all labelled sanctioned addresses. Train/test membership is irrelevant to R3.

**Evidence and severity:**
- Transfers below **0.01 ETH** are ignored (dusting-attack defence)
- `inflow_exposure_fraction` and `outflow_exposure_fraction` measure how much of the wallet's volume touched a sanctioned seed
- R3 strength is `max(exposure, 0.10)` so a confirmed direct link always retains at least 10% of the 30-point weight
- Evidence lists every seed, direction, and ETH amount

---

### R4 · PageRank Network Prominence (Weight: 15)

**What it looks for:** Wallets in the **continuous upper tail of value-weighted PageRank** across the entire transaction graph.

Parameters: `alpha = 0.85`, weighted by `total_value_eth`. Wallets at the **90th percentile or above** earn a continuous strength `(percentile − 90) / 10`; contributions below 0.05 strength are discarded. The 95th percentile is still recorded as a diagnostic `top5` marker.

**Known-address exclusion:** Addresses in `known_addresses.json` under `exchange`, `bridge`, `mixer`, or `contract` are never R4-flagged, because high PageRank often means a legitimate high-volume hub.

---

### R5 · Apriori Behavioural Association Match (Weight: 20)

**What it looks for:** Wallets that match **frequent itemset association rules** learned from the behavioural fingerprints of known bad actors in the training split.

Full details in the next section.

---

## 5. Unsupervised Algorithms: DBSCAN, Apriori, Louvain, PageRank

### Apriori (Rule R5) — Association Rule Mining

**What is Apriori?**
Apriori is a classic algorithm for finding **frequent patterns in boolean data**. Originally developed for market basket analysis ("customers who buy bread also buy butter"), here it is applied to wallet behaviour profiles.

**How ChainLens uses it:**

1. **Booleanise features:** Each wallet is represented as 9 boolean items:
   - `high_fanout` — unique_counterparties >= 20
   - `high_tx_count` — total transactions >= 10
   - `low_value` — median transfer < 0.1 ETH
   - `high_value` — total value >= 10 ETH
   - `many_round` — >= 50% of transfers are whole-ETH amounts
   - `is_contract` — address created a contract
   - `recent_account` — first seen within 7 days of the window end
   - `high_pagerank` — in the PageRank diagnostic top 5% (95th percentile)
   - `dbscan_noise` — flagged as an outlier by DBSCAN

2. **Mine frequent itemsets** on the **training split only** with `min_support = 0.05`

3. **Generate association rules** targeting `flagged` as consequent:
   - Only rules with `lift >= 2.0` and `confidence >= 0.8` are kept

4. **Apply to all wallets:** Any wallet whose feature set satisfies a rule's antecedent gets flagged

**Key metrics:**
- **Support:** Fraction of training wallets having this combination (e.g., 0.07 = 7%)
- **Confidence:** Of wallets with this combination, what fraction were bad actors (e.g., 0.85 = 85%)
- **Lift:** How much more likely a wallet with this combination is to be flagged vs base rate (lift=2 means 2× more likely)

---

### DBSCAN — Density-Based Spatial Clustering

**What is DBSCAN?**
DBSCAN clusters data points by density. Points in dense regions form clusters; isolated points are labelled **noise (-1)**. Unlike K-Means, DBSCAN does not require specifying the number of clusters.

**How ChainLens uses it:**
- Input: 5 scaled wallet features (`total_value_eth_scaled`, `tx_count_out_scaled`, `tx_count_in_scaled`, `unique_counterparties_scaled`, `fanout_ratio_scaled`)
- Parameters: `eps = 1.0122` (neighbourhood radius), `min_samples = 5`

**What each label means:**
- **Cluster (0, 1, 2, ...):** Wallet belongs to a dense behavioural group → `dbscan_cluster` flag (display only, not scored)
- **Noise (-1):** Wallet is a behavioural outlier → `dbscan_noise` flag (used as an Apriori item, feeding into R5)

**DBSCAN does not directly score wallets.** It provides a feature (`dbscan_noise`) that feeds into Apriori.

---

### Louvain Community Detection

**What is Louvain?**
Louvain is a modularity-maximising graph clustering algorithm. It partitions the transaction graph into densely connected communities — groups of wallets that transact heavily with each other and sparsely with the outside world.

**How ChainLens uses it:**
- Run on the **undirected** version of the transaction graph
- Weighted by `total_value_eth`
- Communities are numbered stably by sorting on smallest member address

**Louvain does NOT score wallets** — it is a **display-only** signal for analyst investigation. You can see which community a wallet belongs to in the Network and Wallet Dossier tabs.

---

### Value-Weighted PageRank

PageRank assigns each node an importance score based on how many important nodes point to it. In a transaction graph, it captures "who receives large flows from influential addresses".

- Standard PageRank with `alpha = 0.85`
- Edge weight = `total_value_eth`
- Continuous tail begins at the **90th percentile**; strength = `(percentile − 90) / 10`
- Contribution = `15 × strength` (minimum strength 0.05)
- The 95th percentile is retained as a diagnostic `top5` marker for reporting
- Known exchange/bridge/mixer/contract addresses are excluded before flagging

---

## 6. Why Rules Sometimes Catch Nothing — Explained

This is a very important operational reality.

### The Sample Dataset Limitation

When running in `--mode sample`, ChainLens uses a **fixture dataset** with synthetic or limited real transactions. This dataset has sparse inter-wallet connections and may not contain enough inter-address transactions for cycles to form.

### Rule-by-Rule Zero-Hit Reasons

| Rule | Why Zero Hits Happens |
|---|---|
| **R1 Circular Flow** | The transaction subgraph needs directed cycles of length 2–5 with values within ±2% and 24h. In sparse or synthetic data, this rarely occurs naturally. |
| **R2 Fan-Out** | A wallet needs ≥20 outbound transactions to ≥20 unique addresses **and** a temporal scatter→gather continuation. Small or synthetic cohorts often scatter without any gather phase. |
| **R3 Sanctioned 1-Hop** | Requires a frozen sanctions seed to have a ≥0.01 ETH edge to another wallet in the graph. If no sanctioned seed appears as a graph node, zero flags are produced. |
| **R4 PageRank** | Flags the continuous tail above the 90th percentile; on tiny graphs the percentile arithmetic produces edge effects, and known-address exclusion can suppress the only high-centrality node. |
| **R5 Apriori** | Needs: (a) >=10 labelled training positives, (b) patterns that exceed `min_support = 0.05` with `lift >= 2.0` and `confidence >= 0.8`. All three can fail on small or homogeneous datasets. |

### The Compound Rule Effect

The High tier (>=60) requires **at least two rules to fire simultaneously**:
- R3 alone (30 pts) → Low tier
- R2 + R5 (20+20 = 40 pts) → Medium tier
- R1 + R3 + R2 (25+30+20 = 75 pts) → High tier ✅

**In sample mode, it is completely normal for the High tier to be empty.** The Medium tier is the active investigative queue in sample mode.

---

## 7. Sanctioned Entities & Community Detection

### How Sanctioned Addresses Work

ChainLens loads a **frozen sanctions snapshot** from `config/sanctions.json` (currently 120 OFAC SDN Ethereum addresses) and unions it with every labelled `sanctioned` address. Train/test split membership is irrelevant.

Seed sources include:
- OFAC SDN Ethereum addresses (frozen snapshot in `config/sanctions.json`)
- Labelled sanctioned addresses from the label store
- Community darklists and Kaggle fraud labels remain separate label classes, not R3 seeds

### What "1-Hop" Means

```
   Sanctioned Seed (training only)
        │
        │ ← 1 hop (direct transaction edge) →
        │
   Suspicious Wallet (flagged by R3)
```

A wallet is flagged if it directly sent ETH to, or directly received **at least 0.01 ETH** from, a sanctioned address. Two hops away is NOT flagged (too many false positives), and sub-0.01 ETH dust transfers are ignored. The rule reports the sanctioned exposure fraction and scales its 30-point contribution accordingly.

### Community Detection and Sanctioned Clusters

The Louvain communities tab lets analysts answer: **"Is this sanctioned address part of a larger cluster of colluding wallets?"**

If a sanctioned address falls into community #7 with 15 other wallets, those 15 wallets are also in community #7 and visible in the dashboard. The analyst can then investigate whether those co-members also have other flags.

**Louvain + R3 together are powerful:** R3 flags direct contacts, Louvain reveals the broader community structure.

### Known-Address Downweight

```
known_addresses.json → category "exchange" → score × 0.3

e.g., Uniswap router:
  raw_score = 40 (R2 fan-out + R4 pagerank)
  factor = 0.3 (exchange)
  final_score = int(round(40 × 0.3)) = 12 → LOW tier
```

---

## 8. Dashboard: Every Tab & Every Chart

The dashboard (`dashboard.py`) is a **strictly read-only** Streamlit app. Launch with:

```bash
uv run streamlit run dashboard.py
```

### Sidebar

- **Navigation:** Radio button selecting one of 7 views
- **Quick Wallet Jump:** Enter any `0x` address to jump directly to its dossier
- **Cohort Size:** Total wallet count and run mode (SAMPLE / REAL)
- **Watchlist count:** Number of wallets starred for monitoring
- **Notices:** Data quality warnings (version mismatches, missing artifacts)

---

### Tab 1: Overview

**Purpose:** Executive summary of a pipeline run.

**Metrics row (6 cards):**

| Card | What it shows |
|---|---|
| Analyzed Wallets | Total wallets scored in this run |
| High Risk (≥60) | Count in the High tier (red) |
| Medium Risk (25–59) | Count in the Medium tier (amber) |
| Low Risk (0–24) | Count in the Low tier (slate) |
| Flagged Wallets | Wallets triggering >=1 signal |
| Test Avg Precision | Area under Precision-Recall curve on held-out test split |

**Chart 1 — Risk Score Distribution (bar chart):**
- X-axis: Composite score 0–100
- Y-axis: Number of wallets
- Color: Red for High, amber for Medium, slate for Low
- A spike at score 0 is normal (wallets with no rule hits). An empty High bucket in sample mode is expected.

**Chart 2 — Signal & Rule Hit Frequencies (horizontal bar chart):**
- Each bar = one signal (R1–R5, DBSCAN noise, DBSCAN cluster, Louvain community)
- X-axis: Number of wallets triggering that signal
- R4 PageRank should always show ~5% of wallets. R3 will be zero if no sanctioned seeds exist.

**Priority Triage Queue:**
- Top 10 wallets with score >= 25, sorted by score descending
- Clicking "Dossier →" navigates to the Wallet Dossier tab for that address

---

### Tab 2: Triage

**Purpose:** Investigative queue — filter and triage the full wallet cohort.

**Filter controls:**

| Filter | Options |
|---|---|
| Risk Tier | High, Medium, Low (multiselect) |
| Score Range | Slider 0–100 |
| Triggered Rule | Any of R1–R5, DBSCAN noise (multiselect — AND logic) |
| Split Partition | All / train / test |
| Known Entity Category | exchange, bridge, mixer, contract |
| Search Address or Label | Free text substring match |
| Flagged wallets only | Show only wallets with >=1 signal |
| Watchlist only | Show only starred wallets |

**Result table columns:** TIER, ADDRESS, SCORE (raw score), DRIVER (top rule), SIGNALS (count), KNOWN/SPLIT, LABEL/SOURCE, ACTION (Inspect → Dossier)

**Export:** "Download Filtered Wallets (CSV)"

---

### Tab 3: Wallet Dossier

**Purpose:** Deep-dive forensic analysis of a single wallet address.

**Banner:** Full address, tier badge, known-category chip, Etherscan link, Add/Remove Watchlist button

**Rule Evidence Cards (5 cards, one per rule):**
Each shows: `🚨 TRIGGERED (+N pts)` or `⚪ NOT TRIGGERED`, plus expandable evidence JSON

| Card | Evidence shown |
|---|---|
| R1 Circular Flow | Cycle count, cycle paths (lengths 2–5, ±2% value tolerance), SCC size, duration |
| R2 Scatter–Gather | Unique counterparties, outbound transactions, fanout ratio, gather intermediaries, collectors, gather fraction, temporal ordering (72h window) |
| R3 Sanctioned 1-Hop | Seed addresses, direction (in/out/both), ETH amounts, dust filter (≥0.01 ETH), inflow/outflow exposure fraction, evidence strength |
| R4 PageRank | Value-weighted PageRank score, percentile rank (≥90th pct), diagnostic top-5% marker, continuous strength, known entity hub filter |
| R5 Apriori | Antecedent itemset, support, confidence, lift, matched boolean features |

**Score Contribution Breakdown (horizontal bar chart):**
- One bar per rule: red if triggered, grey if not

**Sub-tabs within the Dossier:**

- **📜 Transaction Evidence:** Scatter plot (timestamp vs ETH value, colored by direction) + transactions table
- **📊 Behavioral Features:** All 14 features with raw value and cohort percentile
- **🕸️ Graph Neighbors:** Table of direct 1-hop transaction edges
- **💾 Raw Case File (JSON):** Full audit trace with download button

---

### Tab 4: Network

**Purpose:** Interactive transaction subgraph visualization.

**Controls:** Focal Address, Max Hops (1 or 2), Max Nodes (10–80), Color Nodes By

**Color modes:**
| Mode | What colors mean |
|---|---|
| Risk Tier | Red=High, Amber=Medium, Slate=Low |
| Primary Flag | Orange=cycle, Yellow=fan-out, Red=sanctioned, Purple=PageRank, Sky=Apriori, Grey=noise |
| Louvain Community | Each community gets a distinct color |
| DBSCAN Cluster | Red=noise outlier, Blue=dense cluster |

---

### Tab 5: Patterns

**Purpose:** Inspect the raw output of each detection algorithm.

Sub-tabs: **R1 · Circular Flow**, **R2 · Scatter–Gather**, **R3 · Sanctioned 1-Hop**, **R4 · Continuous PageRank**, **R5 · Apriori Rules**, **DBSCAN Density**, **Louvain Communities**

Each rule sub-tab shows a table of flagged wallets merged with their risk score and tier.

The **DBSCAN tab** shows noise outlier count, cluster count, and a table of noise wallets.

The **Louvain tab** shows community count and a bar chart of the top 15 community sizes.

---

### Tab 6: Evaluation

**Purpose:** Empirical model validation on the held-out test split.

**Operating Points Comparison (two columns):**
- **High Threshold (score >= 60):** Precision, Recall, F1 + Confusion matrix heatmap (blue)
- **Medium Threshold (score >= 25):** Precision, Recall, F1 + Confusion matrix heatmap (green)

**Leave-One-Rule-Out Ablation Study:**
- Table: rule, affected wallets, baseline F1, ablated F1, delta F1, delta precision, delta recall, tier changes
- Bar chart: X=ablated rule, Y=delta F1. Red bars = rule is valuable, Green bars = rule adds noise.

---

### Tab 7: Data & Export

**Purpose:** Audit trail, artifact inventory, and data export.

- **Artifact Inventory table:** All output files with size, last modified, and presence status
- **Provenance & Execution Hashes:** Full `run_info.json` with git commit, package versions, timestamps, API call counts, rules SHA-256
- **Export Bundle:** "📦 Download Complete Artifact Archive (.zip)"

---

## 9. Evaluation: How We Measure Quality

### Train/Test Split

Addresses are assigned to **train** or **test** splits deterministically by address hash — not randomly at runtime. The same address always gets the same split assignment.

### Metrics

- **Precision** = of wallets we called suspicious, what fraction really were bad actors
- **Recall** = of actual bad actors, what fraction did we catch
- **F1** = harmonic mean of precision and recall
- **Average Precision (PR-AUC)** = area under the Precision-Recall curve

Evaluated at two thresholds:
- **High (score >= 60):** High precision focus
- **Medium (score >= 25):** Higher recall focus

### Ablation Study

Drops one rule at a time (sets its weight to 0), re-scores, and re-evaluates. Measures the **marginal value** of each rule.

```
R3 ablated: ΔF1 = -0.180 → R3 is the most important rule
R4 ablated: ΔF1 = +0.005 → R4 adds slight noise on this dataset
```

---

## 10. Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- Etherscan API key (free tier, from https://etherscan.io/myapikey) — only needed for `--mode real`

### Install

```bash
git clone <repo-url>
cd ChainLens
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env and add: ETHERSCAN_API_KEY=your_key_here
```

### Interactive Console (Recommended)

ChainLens includes an interactive terminal runner to execute pipelines, browse DuckDB tables, manage the dashboard, and run checks from a single menu:

```bash
uv run python full_runner.py
```

### Run Sample Pipeline (No API Key Needed)

```bash
uv run python run.py --mode sample
```

Uses the fixture data in `data/sample/` — no internet connection or API key required.

### Run Real Pipeline

```bash
# Full real-data fetch (requires ETHERSCAN_API_KEY in .env)
uv run python run.py --mode real --max-addresses 200

# Skip re-fetching if transactions already cached
uv run python run.py --mode real --skip-fetch

# Force refresh cache
uv run python run.py --mode real --refresh
```

### Launch Dashboard

```bash
uv run streamlit run dashboard.py
# Opens at http://localhost:8501
```

### Run Tests and Quality Assurance

```bash
# Run complete test suite (45 unit & integration tests)
uv run pytest tests/ -q

# Run Ruff linter
uv run ruff check .
```

---

## 11. Configuration Reference

### `config/rules.yaml` — Rule Weights and Thresholds

```yaml
rules_version_label: "v0-deck"
max_score: 100

tiers:
  low_max: 24        # 0–24 = Low
  medium_max: 59     # 25–59 = Medium
  high_min: 60       # 60–100 = High

downweight:
  known_address_factor: 0.3
  known_address_categories: [exchange, bridge, mixer, contract]

rules:
  - id: R1   weight: 25   # length_min: 2, max: 5, value_tolerance: 0.02, time_window_hours: 24
  - id: R2   weight: 20   # scatter gates + require_temporal_ordering: true, gather_window_hours: 72
  - id: R3   weight: 30   # sanctions_path, min_value_eth: 0.01, min_direct_strength: 0.10
  - id: R4   weight: 15   # continuous: true, min_percentile: 90, min_strength: 0.05
  - id: R5   weight: 20   # min_support: 0.05, min_lift: 2.0, min_confidence: 0.8
```

> ⚠️ **Rule weights are locked.** The pipeline validates that weights are exactly `[25, 20, 30, 15, 20]` at startup. Changing them causes a `RuntimeError`.

### `config/detection.yaml` — Algorithm Hyperparameters

```yaml
dbscan:
  eps: 1.0122         # 95th percentile of 5-NN distance
  min_samples: 5

louvain:
  seed: 42
  weight: total_value_eth
  resolution: 1.0

pagerank:
  alpha: 0.85
  max_iter: 100
  tol: 1.0e-6

cycles:
  max_scc_size: 5000
  enum_cap: 5000
  per_scc_cap: 1000
```

### `config/known_addresses.json`

```json
[
  {"address": "0xa0b86991c6218...", "category": "exchange", "note": "USDC issuer"},
  {"address": "0x7a250d5630b4...", "category": "contract", "note": "Uniswap v2 router"}
]
```

---

## 12. Output Artifacts

All written to `data/output/` after a successful pipeline run:

| File | Format | Contents |
|---|---|---|
| `chainlens.duckdb` | DuckDB | Unified relational analytical warehouse containing all pipeline tables |
| `scores.parquet` | Parquet | One row per wallet: address, raw_score, score, tier, top_driver, rules_version |
| `scores.csv` | CSV | Same as above in CSV format |
| `traces.parquet` | Parquet | Full per-wallet audit trace JSON: all rule hits, contributions, downweight details |
| `traces.csv` | CSV | Same as above |
| `flags.parquet` | Parquet | All signal flags: address, signal_name, signal_value, evidence JSON |
| `graph_edges.parquet` | Parquet | Aggregated transaction graph: from, to, tx_count, total_value_eth, timestamps |
| `wallet_features.parquet` | Parquet | 14 engineered features + 5 scaled features per wallet |
| `labels.parquet` | Parquet | Ground truth labels with source and confidence |
| `splits.parquet` | Parquet | Train/test assignment per address |
| `known_addresses.parquet` | Parquet | Snapshot of known entity registry used in this run |
| `transactions_clean.parquet` | Parquet | De-duplicated, filtered transaction table |
| `context_blocks.json` | JSON | Sampled block heights used for context address ingestion |
| `evaluation.json` | JSON | P/R/F1, PR-AUC, confusion matrix at both operating thresholds |
| `evaluation.md` | Markdown | Human-readable evaluation summary |
| `ablation.csv` | CSV | Leave-one-rule-out ablation results per rule |
| `run_info.json` | JSON | Full provenance: git commit, package versions, timestamps, API call counts, hashes |

---

## 13. Architecture Overview

```
ChainLens/
├── full_runner.py          # Interactive CLI terminal console & runner
├── run.py                  # Pipeline orchestrator
├── fetch_data.py           # Stage 1: Etherscan ingestion + label loading
├── features.py             # Stage 2: Feature engineering (DuckDB SQL)
├── patterns.py             # Stage 4: R1–R5 detection + DBSCAN + Louvain + Apriori
├── scoring.py              # Stage 5: Composite scoring + audit trace
├── evaluate.py             # Stage 6: P/R/F1 + ablation
├── dashboard.py            # Read-only Streamlit analyst workspace
├── config/
│   ├── rules.yaml          # Locked rule weights, thresholds, and paper refinements
│   ├── detection.yaml      # Algorithm hyperparameters
│   ├── data.yaml           # Data paths and sampling settings
│   ├── sanctions.json      # Frozen OFAC sanctions snapshot (split-independent)
│   └── known_addresses.json # Known entity registry
├── data/
│   ├── sample/             # Fixture data (transactions, labels, known addresses)
│   │   ├── transactions.parquet
│   │   ├── labels_fixture.csv
│   │   └── known_fixture.csv
│   └── output/             # All pipeline artifacts (including chainlens.duckdb)
└── tests/                  # pytest test suite (45 tests)
```

### Module Dependency Map

```
run.py
  ├── fetch_data.py     (stage 1)
  ├── features.py       (stage 2-3)
  ├── patterns.py       (stage 4)
  │     ├── networkx   (graph algorithms)
  │     ├── sklearn    (DBSCAN)
  │     └── mlxtend    (Apriori)
  ├── scoring.py        (stage 5)
  └── evaluate.py       (stage 6)

dashboard.py
  └── reads data/output/ artifacts ONLY — no imports from above modules
```

---

## Frequently Asked Questions

**Q: Why is the High tier empty?**
In sample mode, triggering two or more rules simultaneously is rare by construction. The Medium tier is the active investigative queue. See §6.

**Q: Why does R5 (Apriori) show "no frequent itemsets"?**
There are not enough labelled training positives (bad actors) to mine stable association rules. You need at least ~10 confirmed positives in the training split before Apriori can find patterns clearing `min_support = 0.05` + `lift >= 2.0`.

**Q: Can I change the rule weights?**
No — the weights are intentionally locked at `[25, 20, 30, 15, 20]`. Changing them causes a `RuntimeError` on startup. This prevents model drift without a formal re-review.

**Q: What does "downweighted" mean on a wallet?**
The wallet is in `known_addresses.json` under a benign category (exchange, bridge, etc). Its raw score is multiplied by 0.3 before clamping. A raw score of 40 becomes a final score of 12 (Low tier).

**Q: Can the dashboard be used without running the pipeline?**
No — the dashboard requires at least `scores.parquet` in `data/output/`. Run `uv run python run.py --mode sample` first.

**Q: How is the train/test split determined?**
Deterministically by hashing the wallet address with a fixed seed. The same address always gets the same split assignment across runs.

---

*ChainLens v2 — Blockchain forensics & compliance intelligence platform*
