# ChainLens: System Architecture, Design, Methodology, and Requirements Specification

> **Document Version:** 2.0.0  
> **Target System:** ChainLens Ethereum Risk Intelligence Platform  
> **Status:** Active / Production Specification  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Design](#2-system-architecture--design)
   - [2.1 Architectural Pattern](#21-architectural-pattern)
   - [2.2 High-Level Architecture Diagram](#22-high-level-architecture-diagram)
   - [2.3 Component Decomposition](#23-component-decomposition)
   - [2.4 Data Architecture & Storage Schema](#24-data-architecture--storage-schema)
   - [2.5 Security, Provenance & Integrity Design](#25-security-provenance--integrity-design)
3. [Methodology](#3-methodology)
   - [3.1 Ingestion & Preprocessing Methodology](#31-ingestion--preprocessing-methodology)
   - [3.2 Behavioral Feature Engineering Methodology](#32-behavioral-feature-engineering-methodology)
   - [3.3 Partitioning & Anti-Leakage Scaling Methodology](#33-partitioning--anti-leakage-scaling-methodology)
   - [3.4 Graph & Pattern Detection Algorithms](#34-graph--pattern-detection-algorithms)
     - [Rule R1: Circular Flow & SCC Graph Traversal](#rule-r1-circular-flow-detection)
     - [Rule R2: Outbound Fan-Out Smurfing / Structuring](#rule-r2-outbound-fan-out)
     - [Rule R3: Sanctioned Entity 1-Hop Proximity](#rule-r3-sanctioned-entity-1-hop)
     - [Rule R4: Network Prominence (Value-Weighted PageRank)](#rule-r4-value-weighted-pagerank-continuous-tail)
     - [Unsupervised Clustering: DBSCAN Density Clustering](#unsupervised-dbscan-density-clustering)
     - [Community Partitioning: Louvain Modularity Maximization](#community-partitioning-louvain-modularity)
     - [Rule R5: Apriori Association Rule Mining](#rule-r5-apriori-association-rule-mining)
   - [3.5 Composite Scoring & Risk Tiering Methodology](#35-composite-scoring--risk-tiering-methodology)
   - [3.6 Explainability & Forensic Auditability](#36-explainability--forensic-auditability)
   - [3.7 Empirical Evaluation & Leave-One-Rule-Out Ablation](#37-empirical-evaluation--leave-one-rule-out-ablation)
4. [System Requirements](#4-system-requirements)
   - [4.1 Functional Requirements](#41-functional-requirements)
   - [4.2 Non-Functional Requirements](#42-non-functional-requirements)
   - [4.3 Hardware Requirements](#43-hardware-requirements)
   - [4.4 Software & Environmental Requirements](#44-software--environmental-requirements)
5. [Technologies Used & Architectural Rationale](#5-technologies-used--architectural-rationale)
   - [5.1 Core Programming Language & Runtime](#51-core-programming-language--runtime)
   - [5.2 Analytical Database & Columnar Storage](#52-analytical-database--columnar-storage)
   - [5.3 Graph Analytics & Algorithms](#53-graph-analytics--algorithms)
   - [5.4 Machine Learning & Data Mining](#54-machine-learning--data-mining)
   - [5.5 Visualization & Analyst Dashboard](#55-visualization--analyst-dashboard)
   - [5.6 Quality Assurance & Developer Tooling](#56-quality-assurance--developer-tooling)
   - [5.7 Technology Selection Matrix](#57-technology-selection-matrix)

---

## 1. Executive Summary

**ChainLens** is an explainable, enterprise-grade blockchain forensics, anti-money laundering (AML), and compliance intelligence system designed for Ethereum transaction graphs. The platform bridges the divide between black-box deep learning models (which are legally and operationally uninterpretable) and naive static rules (which produce unsustainable false positive rates).

ChainLens implements a **hybrid heuristic-statistical-graph intelligence pipeline**:
- Ingests raw Ethereum block transaction traces via rate-limited blockchain indexers.
- Models high-precision financial flows with exact 18-decimal fixed-point precision.
- Extracts 14+ multidimensional behavioral metrics per entity.
- Executes five locked forensic detection heuristics ($R_1$–$R_5$) combined with two unsupervised learning algorithms (DBSCAN and Apriori) and graph community clustering (Louvain).
- Produces a composite risk score (0–100) and partitions entities into deterministic risk tiers (**High**, **Medium**, **Low**).
- Yields a complete, auditable forensic JSON case file per entity, backed by an empirical evaluation framework and a read-only Streamlit investigative dashboard.

---

## 2. System Architecture & Design

### 2.1 Architectural Pattern

ChainLens utilizes a **Decoupled Pipeline and Read-Only Presentation Architecture**. The system enforces a strict unidirectional data flow:

```
[Ingestion Tier] ──> [Feature Tier] ──> [Analytics Tier] ──> [Scoring Tier] ──> [Evaluation Tier]
                                                                                       │
                                                                   [Parquet / JSON Storage Layer]
                                                                                       │
                                                                         [Read-Only Dashboard]
```

#### Core Design Invariants:
1. **Separation of Computation and Visualization:** `dashboard.py` does not import or execute any computational stages (`features.py`, `patterns.py`, `scoring.py`). It functions purely as an analytical display layer reading static Parquet and JSON artifacts. This guarantees audit integrity and sub-second UI responsiveness.
2. **Deterministic Reproducibility:** Dataset splits, algorithm seeds, and rules are cryptographically anchored. Rerunning the pipeline on identical data yields bit-for-bit identical risk scores and case files.
3. **Locked Rule Contract:** Rule weights ($R_1: 25, R_2: 20, R_3: 30, R_4: 15, R_5: 20$) are locked via configuration schema validation and runtime integrity checks. Modifying weights without authorization triggers an immediate pipeline halt.
4. **Anti-Leakage Data Isolation:** Model scaling and rule mining are fitted exclusively on the `train` partition. Test partition entities remain completely unseen until formal scoring and evaluation.

---

### 2.2 High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Data Sources
        ES[Etherscan API V2 / Fixtures]
        SAN[OFAC SDN & Darklist Labels]
        KNOWN[Known Entity Registry JSON]
    end

    subgraph Stage 1: Ingestion & Invariant Enforcement [fetch_data.py]
        ING[Rate-Limited Ingestion Engine]
        SPLIT[Deterministic Address Hash Splitter (SHA-256)]
        NORM[Schema Normalization & Cache Engine]
    end

    subgraph Stage 2 & 3: Feature Engineering [features.py]
        DUCK[DuckDB In-Memory OLAP Engine]
        DEDUP[Deduplication & EVM Validation]
        FEAT[14 Behavioral Feature Synthesizer]
        SCALE[Train-Only Standard & Log1p Scaler]
    end

    subgraph Stage 4: Pattern Detection Engine [patterns.py]
        GRAPH[NetworkX DiGraph Builder]
        R1[R1: Tarjan SCC & Cycles 2–5]
        R2[R2: Scatter–Gather Smurf Motif]
        R3[R3: Frozen Sanctions + Exposure]
        R4[R4: Continuous PageRank Tail]
        DB[Unsupervised DBSCAN Outlier Detector]
        LOUV[Louvain Modularity Community Partition]
        R5[R5: Apriori Association Rule Miner]
    end

    subgraph Stage 5: Scoring & Trace Engine [scoring.py]
        SCORE[Additive Weighted Scoring Engine]
        DOWN[Benign Entity Downweight Factor (0.3x)]
        CLAMP[Hard Clamping (0-100) & Tier Classifier]
        TRACE[Per-Wallet Forensic JSON Trace Generator]
    end

    subgraph Stage 6: Evaluation & Quality Assurance [evaluate.py]
        EVAL[P/R/F1 & PR-AUC Matrix Calculator]
        ABL[Leave-One-Rule-Out Ablation Engine]
    end

    subgraph Storage Layer [data/output/ Parquet & JSON]
        ART_SCORE[(scores.parquet / traces.parquet)]
        ART_FEAT[(wallet_features.parquet / graph_edges.parquet)]
        ART_EVAL[(evaluation.json / ablation.csv)]
        ART_PROV[(run_info.json)]
    end

    subgraph Presentation Tier [dashboard.py]
        DASH[Streamlit Multi-Tab Analyst Workspace]
    end

    ES --> ING
    SAN --> ING
    KNOWN --> ING
    ING --> SPLIT
    SPLIT --> NORM
    NORM --> DUCK
    DUCK --> DEDUP
    DEDUP --> FEAT
    FEAT --> SCALE
    SCALE --> GRAPH
    SCALE --> DB
    GRAPH --> R1
    GRAPH --> R2
    GRAPH --> R3
    GRAPH --> R4
    GRAPH --> LOUV
    DB --> R5
    R4 --> R5
    R1 & R2 & R3 & R4 & R5 --> SCORE
    KNOWN --> SCORE
    SCORE --> DOWN
    DOWN --> CLAMP
    CLAMP --> TRACE
    TRACE --> EVAL
    TRACE --> ABL
    TRACE --> Storage Layer
    FEAT --> Storage Layer
    EVAL --> Storage Layer
    Storage Layer --> DASH
```

---

### 2.3 Component Decomposition

The system is organized into modular, single-responsibility components:

| Module | Purpose | Primary Inputs | Primary Outputs |
|---|---|---|---|
| [`full_runner.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/full_runner.py) | Interactive CLI Runner & Manager | User interactive input, DuckDB | Workflow execution, database querying, test runs |
| [`fetch_data.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/fetch_data.py) | Ingestion & Caching | Etherscan API, `.env`, `data/sample/` | `raw_transactions.parquet`, `splits.parquet`, `labels.parquet` |
| [`features.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/features.py) | Data Cleaning & Feature Generation | Raw transactions, DuckDB connection | `transactions_clean.parquet`, `wallet_features.parquet`, `scaler.json` |
| [`patterns.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/patterns.py) | Graph & ML Pattern Extraction | Clean transactions, scaled features, detection configs | `flags.parquet`, `graph_edges.parquet`, `louvain_communities`, `apriori_rules.json` |
| [`scoring.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/scoring.py) | Risk Scoring & Trace Generation | Flags, feature set, `rules.yaml`, `known_addresses.json` | `scores.parquet`, `traces.parquet`, `audit_case_files.json` |
| [`evaluate.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/evaluate.py) | Evaluation & Ablation Testing | Scores, ground-truth labels, splits | `evaluation.json`, `evaluation.md`, `ablation.csv` |
| [`run.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/run.py) | Pipeline Orchestration & CLI | CLI parameters, configuration files | Pipeline execution flow, provenance metadata (`run_info.json`) |
| [`dashboard.py`](file:///home/codez/Desktop/Chain-Analysis/ChainLens-v2/dashboard.py) | Read-Only Analyst Workspace | Static output artifacts (`data/output/`) | Interactive 7-tab Streamlit dashboard |

---

### 2.4 Data Architecture & Storage Schema

ChainLens employs Apache Parquet and an embedded DuckDB database file as its primary persistence standard, providing high-performance columnar compression, type-strict schema preservation, and fast vectorized reading.

```
data/
├── sample/                            <- Static fixture environment (offline / CI)
│   ├── transactions.parquet           <- Pre-collected sample transaction graph
│   ├── labels_fixture.csv             <- Ground truth labels for testing
│   └── known_fixture.csv              <- Known benign addresses snapshot
└── output/                            <- Dynamic pipeline artifact directory
    ├── chainlens.duckdb               <- Unified relational analytical warehouse
    ├── transactions_clean.parquet     <- De-duplicated, normalized transactions
    ├── wallet_features.parquet        <- 14 raw + 5 scaled features per address
    ├── graph_edges.parquet            <- Directed aggregated flow edges
    ├── flags.parquet                  <- Long-format signal activations
    ├── scores.parquet / .csv          <- Address risk scores, tiers, top drivers
    ├── traces.parquet / .csv          <- Comprehensive per-wallet forensic traces
    ├── splits.parquet                 <- Deterministic train/test assignments
    ├── labels.parquet                 <- Harmonized ground-truth annotations
    ├── known_addresses.parquet        <- Registry snapshot of downweighted entities
    ├── context_blocks.json            <- Sampled block heights for context addresses
    ├── evaluation.json / .md          <- Classification metrics (P, R, F1, PR-AUC)
    ├── ablation.csv                   <- Leave-one-rule-out impact statistics
    └── run_info.json                  <- System provenance, package hashes, timestamps
```

#### Core Data Schemas:

##### 1. Clean Transactions Schema (`transactions_clean.parquet` & DuckDB `transactions`)
- `tx_hash` (`VARCHAR`): 66-character lowercase Ethereum transaction hash.
- `block_number` (`BIGINT`): Block index in the canonical chain.
- `block_timestamp` (`TIMESTAMPTZ`): UTC execution timestamp.
- `from_address` (`VARCHAR`): 42-character lowercase hex address of sender.
- `to_address` (`VARCHAR`): 42-character lowercase hex address of recipient.
- `value_wei` (`VARCHAR`): Exact unsigned 256-bit integer string.
- `value_eth` (`DECIMAL(38,18)`): Exact fractional Ether amount.
- `receipt_status` (`INTEGER`): EVM status (1 = success; 0 = reverted).
- `is_error` (`INTEGER`): Transaction error flag.

##### 2. Scored Wallets Schema (`scores.parquet` & DuckDB `risk_scores`)
- `address` (`VARCHAR`): Target Ethereum address (primary key).
- `raw_score` (`DOUBLE`): Unclamped additive rule score sum (continuous strengths supported).
- `score` (`INTEGER`): Clamped final risk score (0 to 100).
- `tier` (`VARCHAR`): Risk classification (`High`, `Medium`, `Low`).
- `top_driver` (`VARCHAR`): Rule ID contributing the highest score share.
- `rules_version` (`VARCHAR`): Configured rule contract version label.

##### 3. Forensic Trace Schema (`traces.parquet` & DuckDB `score_trace`)
- `address` (`VARCHAR`): Target address.
- `raw_score` (`DOUBLE`): Sum of fired rule contributions before downweighting.
- `adjusted_raw_score` (`DOUBLE`): Raw score scaled by downweight factor.
- `final_score` (`INTEGER`): Post-downweight, clamped integer score (0–100).
- `tier` (`VARCHAR`): Assigned risk tier.
- `rules` (`JSON` / `STRUCT`): Per-rule breakdown including `hit` (`BOOLEAN`), `weight` (`INTEGER`), `strength` (`DOUBLE`), `contribution` (`DOUBLE`), and evidentiary inputs.
- `downweight` (`STRUCT`): Downweight factor (0.3 if known entity; 1.0 otherwise), category, and status.
- `clamped` (`BOOLEAN`): True if score exceeded 100 before clamping.
- `trace_json` (`VARCHAR`): Full serialized case file JSON.

---

### 2.5 Security, Provenance & Integrity Design

1. **Cryptographic Rule Freezing:** `rules.yaml` defines a versioned rule contract. The orchestrator computes a SHA-256 digest of the rule configuration on startup and validates that weights equal `[25, 20, 30, 15, 20]`. Any manual modification aborts execution immediately.
2. **Provenance Audit Trail (`run_info.json`):** Every execution captures git commit hash, dirty-tree status, Python interpreter version, exact package version locks (`duckdb`, `pandas`, `scikit-learn`, `networkx`, `mlxtend`), execution duration, start/end timestamps, and input data hashes.
3. **Strict Read-Only Dashboard:** The presentation tier has zero capability to alter analytical outputs, mutate database state, or trigger network ingestion.

---

## 3. Methodology

### 3.1 Ingestion & Preprocessing Methodology

The ingestion subsystem (`fetch_data.py`) gathers on-chain transaction data while adhering to API rate limits and structural constraints:

1. **Dual-Mode Operation:**
   - `--mode sample`: Uses static, zero-network Parquet fixtures from `data/sample/` for CI/CD, unit testing, and offline audits.
   - `--mode real`: Invokes the Etherscan V2 API across target address cohorts using per-address Parquet disk caches.
2. **Adaptive Rate Limiting & Backoff:** Implements token-bucket rate limiting with exponential backoff on HTTP 429 and 5xx status codes.
3. **Transaction De-duplication:** Transactions between two tracked cohort wallets appear twice (once in the sender's trace and once in the recipient's). Deduplication is performed across `tx_hash` partitioned by block ordering:
   $$\text{QUALIFY } \text{row\_number}() \text{ OVER (PARTITION BY tx\_hash ORDER BY block\_number) } = 1$$
4. **Validation & Filtering Invariants:**
   - Excludes reverted transactions (`receipt_status = 1` AND `is_error = 0`).
   - Validates 40-hex-character EVM address regular expressions: `^0x[0-9a-f]{40}$`.
   - Filters self-transfers (`from_address <> to_address`).
   - Normalizes all addresses to lowercase.

---

### 3.2 Behavioral Feature Engineering Methodology

Traditional AML systems often suffer from floating-point inaccuracies when parsing Ethereum's 18-decimal fixed-point denominations. ChainLens calculates all transactions using DuckDB's native `DECIMAL(38,18)` types to preserve exact precision:

$$\text{value\_eth} = \left(\text{value\_wei} \mathbin{/\mkern-6mu/} 10^{18}\right) + \frac{\text{value\_wei} \pmod{10^{18}}}{10^{18}}$$

Fourteen discrete behavioral features are engineered via a single unified SQL query over the transaction graph:

| Feature Identifier | SQL Definition / Calculation Logic | Forensic Significance |
|---|---|---|
| `tx_count_out` | $\sum \mathbf{1}_{\text{from\_address} = a}$ | Outbound transfer frequency |
| `tx_count_in` | $\sum \mathbf{1}_{\text{to\_address} = a}$ | Inbound funding frequency |
| `tx_count_total` | `tx_count_out` + `tx_count_in` | Gross transaction velocity |
| `unique_counterparties` | $|\{ \text{to\_address} \mid \text{from\_address} = a \}|$ | Dispersion degree of outbound funds |
| `unique_senders` | $|\{ \text{from\_address} \mid \text{to\_address} = a \}|$ | Aggregation degree of inbound funds |
| `fanout_ratio` | $\frac{\text{unique\_counterparties}}{\max(1, \text{tx\_count\_out})}$ | Dispersion behavior ($1.0 = $ fully non-repeating) |
| `out_in_ratio` | $\frac{\text{tx\_count\_out}}{\max(1, \text{tx\_count\_in})}$ | Velocity asymmetry (draining vs. accumulating) |
| `total_value_eth` | $\sum \text{value\_eth}_{\text{out}}$ | Gross outbound transfer volume |
| `avg_value_eth` | $\text{Mean}(\text{value\_eth}_{\text{out}})$ | Average payment size |
| `median_value_eth` | $\text{Median}(\text{value\_eth}_{\text{out}})$ | Outlier-resistant typical transfer magnitude |
| `net_flow_eth` | $\sum \text{value\_eth}_{\text{in}} - \sum \text{value\_eth}_{\text{out}}$ | Net capital retention or liquidation |
| `is_contract` | $a \in \text{receipt\_contract\_addresses}$ | Entity categorization (smart contract vs. EOA) |
| `active_days` | $|\{ \text{Date}(\text{block\_timestamp}) \mid \text{involved}(a) \}|$ | Temporal persistence across calendar days |
| `round_number_ratio` | $\frac{|\{ \text{tx} \mid \text{value} = \lfloor\text{value}\rfloor \}|}{\max(1, \text{tx\_count\_out})}$ | Heuristic marker for human or algorithmic transfers |

---

### 3.3 Partitioning & Anti-Leakage Scaling Methodology

#### Deterministic Address Partitioning
To guarantee reproducible cohort separation without cross-partition contamination, wallets are mapped to either `train` or `test` partitions via a SHA-256 hash modulo operation:

$$\text{hash\_val} = \text{int}\Big(\text{SHA256}(a \parallel \text{salt})[:8], 16\Big)$$

$$\text{split} = \begin{cases} \text{test}, & \text{if } (\text{hash\_val} \bmod 100) < (\text{test\_ratio} \times 100) \\ \text{train}, & \text{otherwise} \end{cases}$$

#### Anti-Leakage Scaling
Standard ML models often suffer from data leakage when feature scalers are fitted on the entire dataset. ChainLens strictly isolates scaling:
1. Five continuous features are selected for normalization: `total_value_eth`, `tx_count_out`, `tx_count_in`, `unique_counterparties`, and `fanout_ratio`.
2. Features exhibiting high positive skew undergo log-plus-one transformation:
   $$x' = \ln(1 + x)$$
3. The `StandardScaler` calculates mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ **exclusively over wallets in the `train` split**:
   $$z = \frac{x' - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. The learned parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are applied across the entire dataset. The scaler state is saved to `scaler.json` for auditing.

---

### 3.4 Graph & Pattern Detection Algorithms

```
                          ┌─────────────────────────┐
                          │ Transaction Subgraph G  │
                          └────────────┬────────────┘
                                       │
         ┌───────────────┬─────────────┼─────────────┬────────────────┐
         │               │             │             │                │
         ▼               ▼             ▼             ▼                ▼
   ┌───────────┐   ┌───────────┐ ┌───────────┐ ┌───────────┐    ┌───────────┐
   │ Tarjan's  │   │ 3-Gate    │ │ Sanction  │ │ Weighted  │    │ DBSCAN    │
   │ SCC Graph │   │ Outbound  │ │ 1-Hop     │ │ PageRank  │    │ Density   │
   │ Search    │   │ Threshold │ │ Neighbor  │ │ 95th Pct  │    │ Features  │
   └─────┬─────┘   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    └─────┬─────┘
         │               │             │             │                │
         ▼               ▼             ▼             ▼                ▼
     [Rule R1]       [Rule R2]     [Rule R3]     [Rule R4]     [Outlier Noise]
     (25 pts)        (20 pts)      (30 pts)      (15 pts)             │
         │               │             │             │                ▼
         │               │             │             │          ┌───────────┐
         │               │             │             │          │ Apriori   │
         │               │             │             │          │ Rule Mine │
         │               │             │             │          └─────┬─────┘
         │               │             │             │                │
         │               │             │             │                ▼
         │               │             │             │            [Rule R5]
         │               │             │             │            (20 pts)
         ▼               ▼             ▼             ▼                ▼
   └──────────────────────────────────────────────────────────────────────┘
                         Composite Scoring Engine (0-100)
```

#### Rule R1: Circular Flow Detection
- **Typology:** Wash trading, layering, and peeling chains where capital moves through intermediary accounts before returning to the originator.
- **Algorithm:**
  1. Identifies Strongly Connected Components (SCCs) using Tarjan's algorithm on directed graph $G = (V, E)$.
  2. Bounded cycle search isolates cycles of length **$2 \le k \le 5$** (length-2 self-trade pairs per Victor & Weintraud).
  3. Applies uniform wash tolerance: the value of all edges in the cycle must fall within **$\pm 2\%$** of the initial transfer:
     $$\frac{\max_{e \in C} \text{value}(e) - \min_{e \in C} \text{value}(e)}{\max_{e \in C} \text{value}(e)} \le 0.02$$
  4. Temporal windowing: the delta between the first and last transaction in the cycle must not exceed 24 hours.
- **Complexity Protections:** SCCs containing $> 5,000$ nodes are capped to prevent combinatorial path explosion; cycles are canonicalised to avoid rotation duplicates.
- **Assigned Weight:** **25 points**.

#### Rule R2: Scatter–Gather Smurf Motif
- **Typology:** Smurfing and structuring schemes involving rapid outbound dispersion to avoid single-transaction threshold reporting.
- **Scatter Criteria:** A strict three-gate logical conjunction:
  $$\text{Gate}_1: \text{unique\_counterparties} \ge 20$$
  $$\text{Gate}_2: \text{tx\_count\_out} \ge 20$$
  $$\text{Gate}_3: \text{fanout\_ratio} \ge 0.5$$
- **Gather Criteria (Starnini et al.):**
  - At least 2 intermediaries, or 10% of scattering counterparties, must forward funds onwards.
  - At least 1 distinct collector must receive those forwarded funds.
  - Gather edges must start after their scatter edge and within a 72-hour window.
- **Assigned Weight:** **20 points** (full weight only when the ordered motif is confirmed).

#### Rule R3: Sanctioned Entity 1-Hop
- **Typology:** Direct exposure to OFAC-sanctioned addresses, known exploiters, or blacklisted mixing pools.
- **Algorithm:**
  - Seeds come from the frozen `config/sanctions.json` OFAC snapshot unioned with all labelled sanctioned addresses; **train/test membership is irrelevant**.
  - Ignores transfers below **0.01 ETH** to defeat dusting attacks.
  - Queries immediate predecessor/successor relationships:
    $$(u, v) \in E \quad \text{where } u \in \text{Sanctioned} \lor v \in \text{Sanctioned}$$
  - Computes `inflow_exposure_fraction` and `outflow_exposure_fraction`, then sets R3 strength to `max(exposure, 0.10)`.
- **Scope Restriction:** Strictly limited to 1-hop transactions. Multi-hop propagation is handled via Louvain community clustering to limit false positives.
- **Assigned Weight:** **30 points × strength**.

#### Rule R4: Value-Weighted PageRank (Continuous Tail)
- **Typology:** Hub identification in fund-routing rings and dominant money mules.
- **Algorithm:** PageRank over directed graph $G$ using transaction value weights:
  $$\mathbf{PR}(u) = \frac{1 - \alpha}{|V|} + \alpha \sum_{v \in M(u)} \frac{\mathbf{PR}(v) \cdot w(v, u)}{\sum_{k \in N_{\text{out}}(v)} w(v, k)}$$
  Where $\alpha = 0.85$, $w(v, u) = \text{total\_value\_eth}(v \to u)$, and convergence tolerance $\tau = 1.0 \times 10^{-6}$.
- **Activation:** Continuous upper-tail strength from the 90th percentile:
  $$\text{strength}(u) = \operatorname{clamp}\left(\frac{\text{percentile}(u) - 90}{10}, 0, 1\right)$$
  Contributions below 0.05 strength are discarded; the 95th percentile is retained as a diagnostic `top5` marker.
- **Known-address exclusion:** Exchange, bridge, mixer, and contract addresses are never R4-flagged.
- **Assigned Weight:** **15 points × strength**.

#### Unsupervised DBSCAN Density Clustering
- **Typology:** High-dimensional behavioral outlier identification without requiring a predefined cluster count.
- **Parameters:** Configured with $\varepsilon = 1.0122$ (empirically derived from the 95th percentile 5-NN distance) and $\text{min\_samples} = 5$.
- **Feature Space:** 5 normalized features: `total_value_eth_scaled`, `tx_count_out_scaled`, `tx_count_in_scaled`, `unique_counterparties_scaled`, and `fanout_ratio_scaled`.
- **Classification:** Wallets labeled $-1$ are categorized as `dbscan_noise`. This status serves as an input signal for Apriori association rule mining.

#### Community Partitioning: Louvain Modularity
- **Typology:** Subgraph modularity clustering to identify colluding rings and sybil clusters.
- **Algorithm:** Modularity maximization on the undirected graph weighted by total transaction volume:
  $$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
- **Role:** Non-scoring analyst signal surfaced in the Network and Dossier tabs.

#### Rule R5: Apriori Association Rule Mining
- **Typology:** Behavioral profiling based on combinations of subtle risk indicators.
- **Algorithm:**
  1. Binarizes each address into a transaction profile across 9 boolean items:
     - `high_fanout`: $\text{unique\_counterparties} \ge 20$
     - `high_tx_count`: $\text{tx\_count\_total} \ge 10$
     - `low_value`: $\text{median\_value\_eth} < 0.1$
     - `high_value`: $\text{total\_value\_eth} \ge 10$
     - `many_round`: $\text{round\_number\_ratio} \ge 0.5$
     - `is_contract`: $\text{is\_contract} = \text{True}$
     - `recent_account`: active within the final 7 days of the observation window
     - `high_pagerank`: wallet in the diagnostic top 5% of PageRank ($\ge 95\text{th percentile}$)
     - `dbscan_noise`: wallet flagged as a DBSCAN outlier
  2. Generates frequent itemsets on **training split positives only** with $\text{min\_support} = 0.05$.
  3. Derives association rules where consequent is `flagged = True`.
  4. Keeps rules satisfying both $\text{confidence} \ge 0.80$ and $\text{lift} \ge 2.0$:
     $$\text{Confidence}(X \to Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
     $$\text{Lift}(X \to Y) = \frac{\text{Confidence}(X \to Y)}{\text{Support}(Y)}$$
  5. Any cohort entity matching the antecedent $X$ of an active rule triggers R5.
- **Assigned Weight:** **20 points**.

---

### 3.5 Composite Scoring & Risk Tiering Methodology

```
Rule Weights:
  R1 (Circular Flow):       25 pts
  R2 (Scatter–Gather):      20 pts
  R3 (Sanctioned 1-Hop):    30 pts
  R4 (Continuous PageRank): 15 pts
  R5 (Apriori Rule Match):  20 pts
                           ───────
  Max Possible Raw Score:  110 pts
```

#### Scoring Formulation
1. **Raw Score Calculation:** R1, R2 and R5 are binary; R3 and R4 carry an evidence strength $s_i \in [0,1]$:
   $$\text{raw\_score}(a) = \sum_{i=1}^{5} w_i \cdot s_i(a), \qquad s_i(a) = 1 \text{ for binary rules}$$
2. **Benign Entity Downweighting:** High-volume institutional entities (e.g., Uniswap routers, exchange hot wallets) often display structural patterns that trigger rules like fan-out or high PageRank. If an address is listed in `known_addresses.json` under categories such as `exchange`, `bridge`, `mixer`, or `contract`, its raw score is scaled by a downweight factor of $0.3\times$:
   $$\text{adjusted\_score}(a) = \begin{cases} \text{round}\big(\text{raw\_score}(a) \times 0.3\big), & \text{if } a \in \text{KnownRegistry} \\ \text{raw\_score}(a), & \text{otherwise} \end{cases}$$
3. **Clamping:** The final score is clamped to the range $[0, 100]$:
   $$\text{score}(a) = \min\big(100, \max(0, \text{adjusted\_score}(a))\big)$$

#### Risk Tier Partitioning
Entities are partitioned into three operational compliance queues:

$$\text{Tier}(a) = \begin{cases} 
\mathbf{High}, & \text{if } 60 \le \text{score}(a) \le 100 \\ 
\mathbf{Medium}, & \text{if } 25 \le \text{score}(a) \le 59 \\ 
\mathbf{Low}, & \text{if } 0 \le \text{score}(a) \le 24 
\end{cases}$$

> **Compound Rule Effect:**
> Achieving a **High** risk tier ($\ge 60$) requires triggering at least two rules concurrently (e.g., $R_3 + R_1 = 30 + 25 = 55 \to \text{Medium}$; $R_3 + R_1 + R_2 = 30 + 25 + 20 = 75 \to \text{High}$). This multi-signal requirement helps suppress false-positive escalations.

---

### 3.6 Explainability & Forensic Auditability

ChainLens creates a self-contained JSON forensic audit case file for every scored wallet. This file includes:
1. Complete rule activation histories and points awarded.
2. Concrete graph and statistical evidence (e.g., specific cycle paths for R1; counterparty address sets for R2; seed addresses for R3; PageRank percentiles for R4; itemset antecedents for R5).
3. Adjustments and rationale for downweighting benign entities.
4. Links to on-chain transaction hashes, block numbers, and timestamps.

This ensures analysts and compliance officers can trace risk scores back to underlying evidence without relying on black-box model inferences.

---

### 3.7 Empirical Evaluation & Leave-One-Rule-Out Ablation

The evaluation framework (`evaluate.py`) assesses pipeline quality on the held-out test split:

1. **Classification Metrics:** Evaluated across both the High ($\ge 60$) and Medium ($\ge 25$) operating thresholds:
   - **Precision:** $\frac{TP}{TP + FP}$
   - **Recall:** $\frac{TP}{TP + FN}$
   - **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
   - **Average Precision (PR-AUC):** Area under the Precision-Recall curve.
2. **Leave-One-Rule-Out Ablation:** The evaluation engine iteratively sets each rule's weight to zero ($w_i = 0$), recomputes all wallet scores, and recalculates test set F1 scores:
   $$\Delta \text{F1}_i = \text{F1}_{\text{ablated}, i} - \text{F1}_{\text{baseline}}$$
   - $\Delta \text{F1} < 0$: Identifies rules that provide positive detection value (removing the rule reduces model performance).
   - $\Delta \text{F1} > 0$: Highlights potential sources of noise or false positives on the evaluation dataset.

---

## 4. System Requirements

### 4.1 Functional Requirements

| Identifier | Requirement Specification |
|---|---|
| **FR-01** | Ingest Ethereum transactions from Etherscan V2 API or local Parquet sample fixtures. |
| **FR-02** | Cache on-chain transaction histories locally in Parquet format by address. |
| **FR-03** | Deterministically assign wallets to train/test partitions using seeded address hashes. |
| **FR-04** | Deduplicate mirrored sender/receiver transactions and filter out reverted operations. |
| **FR-05** | Compute 14 behavioral features per wallet using DuckDB fixed-point decimal arithmetic. |
| **FR-06** | Normalize features using parameters fitted strictly on the training partition. |
| **FR-07** | Construct directed flow graphs and detect circular cycles ($2 \le k \le 5$) with ±2% value tolerance within 24-hour windows. |
| **FR-08** | Detect scatter→gather smurf motifs through temporal ordering of fan-out and onward transfers. |
| **FR-09** | Detect ≥0.01 ETH 1-hop exposure to frozen OFAC sanctions seeds, independent of train/test split, and report exposure fractions. |
| **FR-10** | Calculate value-weighted PageRank and score the continuous upper tail (≥90th percentile), excluding known hubs. |
| **FR-11** | Identify behavioral outliers using unsupervised DBSCAN density clustering. |
| **FR-12** | Partition transaction subgraphs into communities using Louvain modularity optimization. |
| **FR-13** | Mine association rules with Apriori on training split bad actors and flag matching wallets. |
| **FR-14** | Calculate composite risk scores (0–100) and assign entities to High, Medium, or Low tiers. |
| **FR-15** | Apply a $0.3\times$ downweight factor to verified benign entities (exchanges, bridges, contracts). |
| **FR-16** | Export full per-wallet forensic traces, evidentiary JSONs, and summary Parquet files. |
| **FR-17** | Compute Precision, Recall, F1, and PR-AUC, and execute leave-one-rule-out ablation testing. |
| **FR-18** | Provide an interactive CLI terminal runner (`full_runner.py`) and a read-only Streamlit dashboard with seven interactive analytical tabs. |

---

### 4.2 Non-Functional Requirements

| Category | Requirement Specification |
|---|---|
| **Explainability** | Every scored wallet must provide an audit trace detailing all triggered rules, evidence, and points. |
| **Determinism** | Identical inputs and random seeds must produce bit-for-bit identical scores, tiers, and evaluation metrics across runs. |
| **Integrity & Security** | Rule weights are locked at `[25, 20, 30, 15, 20]`. Any configuration drift halts execution. |
| **Data Isolation** | Feature scaling and rule mining must never read test split data prior to scoring. |
| **Performance** | In-memory DuckDB processing must execute feature engineering for 500+ wallets in $< 5.0$ seconds. |
| **Decoupling** | The Streamlit dashboard must operate strictly read-only, reading Parquet artifacts without importing pipeline logic. |
| **Precision** | Financial accounting must use fixed-point `DECIMAL(38,18)` without float64 precision loss. |

---

### 4.3 Hardware Requirements

#### Minimum Operational Footprint (Offline / Sample Mode / CI)
- **Processor:** 64-bit Dual-Core x86_64 or ARM64 (Apple Silicon / AWS Graviton).
- **System Memory (RAM):** 4 GB.
- **Persistent Storage:** 2 GB available SSD disk space.
- **Network Interface:** Optional (not required for sample mode or unit tests).

#### Recommended Production Footprint (Large Cohort / Real-Data Mode)
- **Processor:** Quad-Core Intel Core i7 / AMD Ryzen 7 / Xeon / EPYC (or Apple M-series).
- **System Memory (RAM):** 16 GB to 32 GB DDR4/DDR5 (accommodates large NetworkX in-memory graphs).
- **Persistent Storage:** 50 GB NVMe SSD (high IOPS for reading and writing large Parquet caches).
- **Network Interface:** Dedicated broadband connection ($\ge 50\text{ Mbps}$) for Etherscan API ingestion.

---

### 4.4 Software & Environmental Requirements

- **Operating System:** Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+, Arch), macOS 12+, or Windows 11 with WSL2.
- **Python Runtime:** Python `3.11.x` or `3.12.x` (64-bit).
- **Package & Environment Manager:** `uv` ($\ge 0.4.0$) or standard Python `pip` / `virtualenv`.
- **Compiler Requirements:** C/C++ compiler toolchain (GCC or Clang) if building native packages from source.
- **External Network Services:** Etherscan API V2 Developer Account (free or pro tier) for `--mode real`.
- **Client Web Browser:** Modern Chromium, Firefox, or WebKit browser for the Streamlit dashboard interface.

---

## 5. Technologies Used & Architectural Rationale

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit & Plotly Dashboard                    │
├───────────────────────────────────┬────────────────────────────────────┤
│       Machine Learning & Mining   │       Graph & Network Analytics    │
│       • scikit-learn (DBSCAN)     │       • NetworkX (SCC, Cycles, PR) │
│       • MLxtend (Apriori Rules)   │       • python-louvain (Modularity)│
├───────────────────────────────────┴────────────────────────────────────┤
│                       Analytical Data Processing                       │
│                       • DuckDB OLAP SQL Engine                         │
│                       • Apache Arrow / PyArrow Parquet                 │
│                       • NumPy & Pandas DataFrames                      │
├────────────────────────────────────────────────────────────────────────┤
│                      Runtime Infrastructure & Core                     │
│                      • Python 3.11+ / uv Package Engine                │
│                      • Pytest & Ruff Quality Assurance                 │
│                      • Requests & Python-Dotenv Networking             │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Core Programming Language & Runtime

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **Python** | `>=3.11, <3.13` | Base System Language | Selected for its strong data science ecosystem, clear syntax for compliance code, and performance improvements in Python 3.11+. |
| **uv** | Latest | Package & Toolchain Manager | High-performance Rust-based Python package manager. Provides fast dependency resolution and deterministic lockfile generation (`uv.lock`). |

---

### 5.2 Analytical Database & Columnar Storage

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **DuckDB** | `>=1.0.0` | In-Memory OLAP SQL Engine | Embedded analytical database used in `features.py`. Supports exact arithmetic via `DECIMAL(38,18)`, fast aggregation, and zero-copy data exchange with Arrow and Pandas. |
| **Apache Arrow / PyArrow** | `>=14.0.0` | Columnar Memory & Storage Format | High-performance columnar data standard used for all `data/output/*.parquet` artifacts. Preserves types and optimizes disk usage. |
| **Pandas** | `>=2.1.0, <3.0.0` | In-Memory Data Transformation | Provides familiar tabular data structures for scoring matrices, rule transformations, and evaluation reporting. |
| **NumPy** | `>=1.26.0, <3.0.0` | Vectorized Linear Algebra | Handles numerical transformations, log-scaling, array masking, and matrix calculations. |

---

### 5.3 Graph Analytics & Algorithms

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **NetworkX** | `>=3.2, <4.0` | Graph Modeling & Analysis | Python graph library used for building directed transaction topologies, Tarjan's Strongly Connected Components (SCC), cycle searches, and PageRank calculations. |
| **python-louvain** | Integrated | Graph Community Detection | Implements the Louvain modularity algorithm on undirected transaction graphs to detect community clusters. |

---

### 5.4 Machine Learning & Data Mining

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **scikit-learn** | `>=1.4.0, <2.0.0` | Unsupervised Density Clustering | Powers DBSCAN spatial clustering to flag behavioral outliers (`dbscan_noise`), which then feed into Rule R5. |
| **MLxtend** | `>=0.23.0, <1.0.0` | Association Rule Mining | Provides Apriori frequent itemset generation and association rule extraction (`lift`, `confidence`) across binarized entity profiles. |

---

### 5.5 Visualization & Analyst Dashboard

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **Streamlit** | `>=1.50.0, <2.0.0` | Investigative Analyst Dashboard | Framework for the read-only 7-tab analyst dashboard. Operates without custom frontend builds and renders data from disk artifacts. |
| **Plotly** | `>=5.20.0, <8.0.0` | Interactive Charting & Subgraph Visualizer | Renders interactive network subgraphs, risk distributions, confusion matrices, and ablation charts. |

---

### 5.6 Quality Assurance & Developer Tooling

| Technology | Version Lock | Architectural Role | Architectural Rationale |
|---|---|---|---|
| **pytest** | `>=8.0.0` | Automated Testing Suite | Provides regression and integration testing across all pipeline stages, edge cases, and locked-rule validation. |
| **Ruff** | `>=0.5.0` | Static Linting & Code Quality | High-performance linter and formatter enforcing PEP 8 standards, import sorting, and code cleanliness. |
| **PyYAML** | `>=6.0` | Configuration Serialization | Parses `rules.yaml`, `detection.yaml`, and `data.yaml` files. |
| **requests** | `>=2.31.0, <3.0.0` | HTTP Client Library | Handles communication with the Etherscan V2 API, supporting headers, timeouts, and error handling. |
| **python-dotenv** | `>=1.0.0, <2.0.0` | Environment Management | Loads API keys and configurations from local `.env` files. |

---

### 5.7 Technology Selection Matrix

| Design Requirement | Chosen Solution | Alternative Evaluated | Why the Alternative Was Rejected |
|---|---|---|---|
| **Exact 18-decimal wei arithmetic** | DuckDB `DECIMAL(38,18)` | Python `float64` / NumPy | `float64` precision breaks on values exceeding 15 significant digits, introducing rounding errors in financial transactions. |
| **Fast embedded OLAP** | DuckDB in-memory | SQLite / PostgreSQL | SQLite lacks native columnar analytical primitives; PostgreSQL introduces external database server overhead. |
| **Artifact persistence** | Apache Parquet | CSV files | CSV does not preserve data types, requires larger disk footprints, and exhibits slower read performance on large datasets. |
| **Rule pattern mining** | MLxtend Apriori | FP-Growth / Deep Learning | Black-box neural models lack legal and operational explainability. Apriori produces human-readable, auditable association rules. |
| **Outlier detection** | Scikit-Learn DBSCAN | K-Means Clustering | K-Means forces every data point into a cluster and requires predefining $K$, whereas DBSCAN isolates true noise points without prior cluster assumptions. |
| **Analyst dashboard** | Streamlit + Plotly | React / FastAPI | Full-stack web frameworks add deployment complexity. Streamlit delivers a lightweight, read-only UI tailored for data analysts. |

---

## 6. Document Metadata & Authorization

- **System Name:** ChainLens Forensic Intelligence System
- **Current Version:** `v2.0.0` (Rules Contract: `v0-deck`)
- **Author:** ChainLens Engineering & Compliance Architecture Team
- **Approval Status:** Locked Production Reference Specification
