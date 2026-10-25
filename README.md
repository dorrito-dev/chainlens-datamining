# ChainLens

**Auditable Ethereum wallet risk scoring.** Seven pipeline files, real Etherscan V2
transactions, five transparent rules, graph analytics, a leakage-safe evaluation, and a
read-only Plotly dashboard.

> A ChainLens score is an investigative lead, not proof of wrongdoing. Read the trace.

---

## What it does

ChainLens ingests Ethereum normal transactions for labelled wallets, their counterparties,
Kaggle negatives, and random-block context addresses. It builds a transaction graph, applies
five rules plus DBSCAN, Apriori, and Louvain, and emits a 0–100 score with an auditable
per-rule trace. A held-out test split produces precision, recall, F1, average precision, and
a leave-one-rule-out ablation.

The implementation is deliberately small:

| File | Responsibility | Main output |
|---|---|---|
| `fetch_data.py` | Etherscan V2 client, per-address cache/resume, label loading, negatives, context sampling, train/test split | `transactions_raw.parquet`, `labels.parquet`, `splits.parquet` |
| `features.py` | DuckDB cleaning, exact ETH accounting, wallet features, train-only scaler | `transactions_clean.parquet`, `wallet_features.parquet`, `feature_scaler` |
| `patterns.py` | Graph build, R1 cycles, R2 fan-out, R3 sanctioned 1-hop, R4 PageRank, DBSCAN, Apriori (R5), Louvain | `graph_edges.parquet`, `flags.parquet` |
| `scoring.py` | Locked weights, tiers, clamp, known-address down-weight, trace contract | `scores.parquet`, `traces.parquet` |
| `evaluate.py` | Test-split P/R/F1, insufficient-data guard, secondary operating point, ablation | `evaluation.json`, `evaluation.md`, `ablation.csv` |
| `run.py` | CLI orchestrator for sample/real modes | all artifacts, `run_info.json`, `chainlens.duckdb` |
| `dashboard.py` | Read-only Streamlit + Plotly dashboard | wallet explorer, graph, evaluation, CSV export |

```
.env ──▶ fetch_data.py ──▶ features.py ──▶ patterns.py ──▶ scoring.py ──▶ evaluate.py
                                                                  │
                                                                  ▼
                                                        data/output/ ──▶ dashboard.py
                                      run.py orchestrates the stages
```

`dashboard.py` never recomputes scores, flags, or metrics. It reads only `data/output/`.

---

## Quickstart

### 1. Install

Python 3.11+ is required (the code uses `X | None` type syntax). The development environment
for this run used Python 3.14, but the target is 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the offline sample first

The sample fixture needs **no API key** and is the fastest smoke test:

```bash
python run.py --mode sample
streamlit run dashboard.py
```

Open the URL Streamlit prints (normally `http://localhost:8501`).

### 3. Configure the real Etherscan key

```bash
cp .env.example .env
# edit .env and replace your_etherscan_api_key_here
```

`.env` is gitignored. The key is read from `ETHERSCAN_API_KEY`, sent only to
`https://api.etherscan.io/v2/api`, and is never written to `run_info.json` or any output
artifact. `data/output/` and `data/cache/` are also gitignored.

### 4. Run real data

```bash
# Development sample (recommended first real run: ~600 addresses, ~4 minutes)
python run.py --mode real --max-addresses 400 --max-txs 150 --context-max 200

# Full configured scope (all labels, most-recent 500 tx/address)
python run.py --mode real

# Re-run analytics/evaluation without re-fetching (uses cached raw + labels + split)
python run.py --mode real --skip-fetch
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--mode sample` | Offline fixture; no network, no key required |
| `--mode real` | Etherscan V2 ingestion |
| `--max-addresses N` | Development cap: roughly half labelled positives, half Kaggle negatives |
| `--max-txs N` | Override the per-address most-recent transaction cap (default 500) |
| `--context-max N` | Cap random-block context addresses (default 500) |
| `--no-context` | Disable random-block context sampling |
| `--skip-fetch` | Reuse `transactions_raw.parquet`, `labels.parquet`, `splits.parquet` |
| `--output DIR` | Write artifacts somewhere other than `data/output` |

The Etherscan free tier is rate-limited to **3 calls/second**; the client throttles to that
rate, retries rate-limit responses, and caches each address by chain, offset, and block
window. A cache entry from a 150-transaction run is not silently reused by a 500-transaction
run.

### 5. Open the dashboard

```bash
streamlit run dashboard.py
# or point it at another artifact directory:
CHAINLENS_OUTPUT_DIR=/tmp/chainlens_run streamlit run dashboard.py
```

Tabs: **Overview**, **Wallets**, **Network**, **Evaluation**, **About**. The wallet table is
selectable; the selected wallet shows its rule trace, contribution bars, evidence JSON, and a
JSON download. Scores and traces can be exported as CSV/JSON. The network tab colours nodes by
the already-computed tier; NetworkX supplies only a spring layout.

---

## Scoring contract

Weights are **locked** to the deck:

| Rule | Signal | Weight | Meaning |
|---|---|---:|---|
| R1 | Circular flow | 25 | 3–5 hop cycle with edge values within 5% and the cycle inside 24 hours |
| R2 | Fan-out | 20 | Many distinct outbound counterparties with a high outbound share |
| R3 | Sanctioned 1-hop | 30 | One hop from a **train-split** sanctioned seed, in either direction |
| R4 | PageRank top 5% | 15 | Value-weighted graph prominence ≥ the 95th percentile |
| R5 | Apriori match | 20 | Behavioural itemset match to train-split flagged wallets, confidence ≥ 0.8 |

```
raw_score   = sum(weight for each hit)                  # maximum 110
adjusted    = raw_score * 0.3 if a known exchange/bridge/mixer/contract
score       = min(100, round(adjusted))                 # hard clamp
tier        = low 0–24 | medium 25–59 | high 60–100
```

The known-address down-weight is applied **before** the clamp and tiering. The final score is
always an integer. The raw maximum is 110, so a wallet hitting all five rules is clamped from
110 to 100; `clamped: true` is recorded in its trace.

### The slide-21 “65” discrepancy

The deck's slide-21 example says R1 + R2 + R4 = 65. Under the locked weights it is:

```
R1 25 + R2 20 + R4 15 = 60
```

Therefore the example is High (60 ≥ 60), but the value **60**, not 65, is what the code
asserts. The test suite checks `raw_score == 60` and `tier == "high"`; it does not hard-code 65.
Resolving 65 requires either changing a locked weight or adding a signal, which the project
intentionally did not do.

### Trace contract

Every wallet has a JSON trace:

```json
{
  "address": "0x…",
  "raw_score": 60,
  "adjusted_raw_score": 60.0,
  "score": 60,
  "tier": "high",
  "top_driver": "R1",
  "clamped": false,
  "downweight": {"applied": false, "factor": 1.0, "category": null},
  "rules": {
    "R1": {"hit": true, "weight": 25, "contribution": 25, "inputs": {}, "gate": null},
    "R2": {"hit": false, "weight": 20, "contribution": 0, "inputs": {}, "gate": null}
  },
  "rules_version": "…",
  "scaler_version": "…"
}
```

`validate_trace()` checks that contributions sum to `raw_score`, the clamp is consistent, and
the tier matches the score. The dashboard displays this trace, not a hidden model.

---

## Rules in detail

### R1 — circular flow (25)

Finds elementary cycles of length 3–5 inside strongly connected components. A cycle qualifies
only when every edge value is within 5% of the cycle minimum and the time span from the first
edge to the last edge is at most 24 hours. SCCs larger than `max_scc_size` are skipped as a
scalability guard; enumeration is capped. The trace records the cycle count and cycle paths.

### R2 — outbound fan-out (20)

Implemented gate:

```
unique_counterparties >= 20
AND tx_count_out >= 20
AND unique_counterparties / tx_count_out >= 0.5
```

The deck's `out_in_ratio > 5` and `counterparties > 50` are recorded in each R2 trace as
`r2_deck_equivalent`. This is a deliberate, documented semantic deviation: the implemented
gate is the one the fixture and slide-21 example were built against.

### R3 — sanctioned 1-hop (30)

Seeds are **sanctioned addresses in the train split only**. Both successors and predecessors
of a seed are flagged. Darklist addresses remain positive labels for R5, but are not R3 seeds. Test-split sanctioned addresses are never seeds. The trace
records the seed, direction (`in`/`out`), and seed count.

### R4 — PageRank top 5% (15)

Value-weighted PageRank with `alpha=0.85`, damping and a `1e-12` floor on zero-value edges.
Addresses at or above the 95th percentile are flagged. PageRank is a prominence/centrality
signal, not a fraud probability; exchanges and token contracts can legitimately rank high,
which is why the known-address down-weight exists.

### R5 — Apriori match (20)

The labelled train cohort is converted to boolean items:

```
high_fanout, high_tx_count, low_value, high_value, many_round,
is_contract, recent_account, high_pagerank, dbscan_noise
```

Apriori mines rules with `min_support=0.05`, `min_lift=2.0`; only rules with consequent
`flagged` and antecedents of size ≤ 3 are kept. At match time, confidence must be ≥ 0.8. The
mining table contains **only labelled train addresses**; scoring applies the learned rules to
every wallet. Mining the full 9k-address graph instead diluted `flagged` support to 1.1% and
made `min_support=0.05` impossible — that bug was fixed and is covered by a test.

`dbscan_noise` is an Apriori item, so DBSCAN is connected to scoring via R5. DBSCAN is not
scored directly, and its cluster labels are not risk labels.

### DBSCAN

DBSCAN runs on `total_value_eth`, `tx_count_out`, `tx_count_in`, `unique_counterparties`, and
`fanout_ratio`, all scaled on the train split, for wallets with at least 3 transactions.
`eps=1.0122` is the 95th percentile of the 5-NN distance on the current feature cohort (the
previous `3.195323` value produced one cluster and zero noise). Clusters feed the
`dbscan_cluster` display flag and `dbscan_noise` feeds R5.

### Louvain

`nx.community.louvain_communities` runs on the undirected projection, seeded and weighted by
`total_value_eth`. The community id is a display/context flag only; it is **not scored**.

---

## Data and sampling

### Transactions

- Etherscan V2 `account/txlist`, Ethereum mainnet (`chainid=1`).
- Per-address **most recent 500** normal transactions (configurable); the actual block range is
  recorded in `run_info.json`.
- Failed transactions and self-transfers are removed; contract-creation rows are retained where
  useful; values are handled as `DECIMAL(38,18)` ETH, never binary floats for accounting.
- A per-address Parquet + metadata cache enables resume. The cache key includes chain, offset,
  start block, and end block.

### Labels

| Source | Use |
|---|---|
| `data/labels/ofac_eth.txt` | Sanctioned positives (snapshot dated in `SOURCES.json`) |
| `data/labels/darklist.json` | Darklist positives |
| `data/labels/kaggle_fraud.csv` | `flag=1` positives and sampled `flag=0` negatives (7,662 available) |

Negatives are sampled deterministically (`negative_count: 2000`, seed 42). An address that is
positive in any source is never used as a negative. Context addresses are sampled from 20
random blocks (active addresses, unlabelled) and are excluded from test metrics.

### Split and leakage

- 70/30 stratified train/test split by label stratum, fixed seed 42.
- R3 seeds are train-only.
- R5 mines on labelled train addresses only; test labels never enter mining.
- The scaler is fitted on the train split only and reused for all addresses.
- Test addresses with no fetched activity are dropped and counted in `evaluation.json`.
- The sample is **case-control** (labels enriched), so P/R/F1 are sample metrics, not
  population prevalence estimates.

---

## Evaluation

Primary operating point is **High ≥ 60**. The code also reports a secondary **Medium ≥ 25**
point, because the High tier can be empty while the ranking still carries signal. Average
precision is computed over the test set.

Guards:

- Fewer than 3 test positives **or** fewer than 3 test negatives → `insufficient: true` and
  precision/recall/F1 are `null`, never a fabricated 0 or 1.
- Fixture labels are excluded by default.
- `evaluation.json` records test counts, dropped addresses, TP/FP/TN/FN, and both operating
  points.
- Leave-one-rule-out ablation removes each rule from the score, recomputes tiers, and reports
  `affected_wallets`, `delta_f1`, and `tier_changes`. If F1 is undefined, the ablation still
  reports tier changes.

---

## Results

### Offline sample (committed fixture, no network)

| Wallets | Tiers | R1 | R2 | R3 | R4 | R5 | DBSCAN | Louvain | Test labels |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 377 | 371 low · 6 medium · 0 high | 1 cycle (3 addresses) | 2 | 2 | 19 | 4 rules / 25 matches | 3 clusters · 19 noise | 38 communities | 1 positive / 3 negatives → insufficient |

The sample is for correctness and UI smoke testing, not evidence.

### Real Etherscan V2 runs

#### Development run (600 addresses)

`python run.py --mode real --max-addresses 400 --max-txs 150 --context-max 200`

| Wallets | Tiers | R1 | R2 | R3 | R4 | R5 | DBSCAN | Louvain | Test positives / negatives |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 9,061 | 8,968 low · 93 medium · 0 high | 0 | 40 | 63 | 454 | 0 rules (best lift 1.450 < 2.0) | 4 clusters · 36 noise | 2,613 communities | 46 / 60 |

| Operating point | Precision | Recall | F1 | TP / FP / TN / FN |
|---|---:|---:|---:|---|
| High ≥ 60 (primary) | 0.000 | 0.000 | 0.000 | 0 / 0 / 60 / 46 |
| Medium ≥ 25 (secondary) | 0.429 | 0.065 | 0.113 | 3 / 4 / 56 / 43 |

Average precision: **0.493** (case-control test sample; base positive rate 0.434).

#### 1,000-address comparison

`python run.py --mode real --max-addresses 1000 --max-txs 500 --context-max 200`

This run requested 1,200 addresses (500 positives, 500 Kaggle negatives, 200 random-block
context addresses), fetched 117,709 raw rows, and produced 25,779 scored wallets (counterparties
included). The fetch used 1,035 cache hits plus 165 live Etherscan calls; the final analytics
were then reproduced with `--skip-fetch`.

| Wallets | Tiers | R1 | R2 | R3 | R4 | R5 | DBSCAN | Louvain | Test positives / negatives |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 25,779 | 25,290 low · 489 medium · 0 high | 0 | 68 | 442 | 1,370 | 0 rules (best lift 1.407 < 2.0) | 2 clusters · 32 noise | 8,879 communities | 118 / 150 |

| Operating point | Precision | Recall | F1 | TP / FP / TN / FN |
|---|---:|---:|---:|---|
| High ≥ 60 (primary) | 0.000 | 0.000 | 0.000 | 0 / 0 / 150 / 118 |
| Medium ≥ 25 (secondary) | 0.462 | 0.051 | 0.092 | 6 / 7 / 143 / 112 |

Average precision: **0.522**; 32 test positives were dropped because they had no fetched
activity. The 1,000-address run is a larger comparison, not a controlled improvement over the
600-address run: it uses a different transaction cap and a larger known-address seed.

**How to read these results.** Both high tiers are empty, so high-tier F1 is 0 and precision is
undefined in practice. The medium tier has small but non-zero recall. Average precision is only
slightly above the case-control base rate (about 0.44). This is an honest negative result for
these labels, not a tuned success:

- The Kaggle labels are from 2020–2021, while the graph is built from each address's most
  recent 2026 transactions. A labelled 2021 fraud address that is now inactive has no graph
  signal; 14 positives were dropped in the 600 run and 32 in the 1,000 run.
- R3 sanctions are train-only. In the 1,000 run R3 flags 442 addresses, but only 1 of the 118
  active test positives is a 1-hop neighbour of a train sanction seed. R2 hits 5 test positives;
  R4 flags 61 test positives but contributes only 15 points.
- R1 found no qualifying 3–5 cycles. The graph does contain 447 reciprocal 2-cycles, which the
  rule deliberately excludes.
- R5's strict gate (`min_lift=2.0`, confidence 0.8) yields zero rules; the best observed lift is
  1.407–1.450. The threshold was **not** silently relaxed. The sample fixture still exercises
  R5 with four rules.
- Without R1/R3/R5 on most test wallets, no test wallet reaches 60. The top raw score is 50
  (R3 + R2); the top-scoring addresses are still exchange/token/router infrastructure and are
  not all covered by the documented known-address seed.
- The full configured label set is ~4,300 addresses and requires roughly 25–30 minutes at
  3 req/s; the command is `python run.py --mode real` (optionally with a larger cache).

---

## Deviations from the deck / review

| Item | Deck / review | Implemented | Why |
|---|---|---|---|
| R1 + R2 + R4 example | 65 | **60** | Locked weights: 25 + 20 + 15 = 60; 60 still clears High |
| R2 gate | `out_in_ratio > 5` AND counterparties > 50 | `unique_counterparties ≥ 20`, `tx_count_out ≥ 20`, `fanout_ratio ≥ 0.5`; deck condition in trace | Preserves tested v0 behaviour; deviation is visible, not silent |
| Analysis window | One fixed week | Per-address most recent 500 tx; actual range recorded | Labels span years; a fixed week would empty most labelled addresses |
| R5 mining cohort | “train only” | Labelled train cohort only | Mining all graph addresses diluted support below `min_support` |
| R5 lift | 2.0 | 2.0 (unchanged); 0 rules on real dev data | Strict result reported; best observed lift 1.407–1.450 |
| DBSCAN `eps` | 3.195323 | 1.0122 | 95th percentile 5-NN distance; old value gave 1 cluster/0 noise |
| Evaluation | High only | High primary + Medium secondary + average precision | High tier can be empty while ranking still has signal |
| Known addresses | Small seed | Added major token/router infrastructure | Reduces obvious R3/R4 false positives; still explicitly non-exhaustive |
| Storage | Postgres + Neo4j | DuckDB file + NetworkX | Fits in memory; removes services with no analytical benefit |
| Cache scope | Implicit | Requested addresses only | Prevents stale cache entries from an earlier, larger sample leaking into a smaller run |
| Tests | 0 or 36 | 7 test files / 36 tests | One focused suite per stage, including slide-21 arithmetic |

### What the project deliberately does **not** claim

- That a high score is proof of fraud.
- That Kaggle `flag=0` addresses are verified non-fraud.
- That the OFAC/darklist snapshots are current.
- That the case-control P/R/F1 generalizes to Ethereum's population.
- That 60 is 65, or that the deck's R2 gate is identical to the implemented gate.

---

## Tests and CI

```bash
ruff check .
python -m pytest -q
```

The 7 test files cover:

| File | Focus |
|---|---|
| `tests/test_fetch.py` | Normalisation, cache/resume/refresh, error recovery, labels, split, context, HTTP client, key loader |
| `tests/test_features.py` | Deduplication, failed/self filters, DECIMAL ETH, hand-computed features, train-only scaler, schema guard |
| `tests/test_patterns.py` | R1 value/time/length filters, R2 deck diagnostic, PageRank zero floor, DBSCAN outlier, leakage-safe Apriori + `dbscan_noise`, Louvain, train-only R3 |
| `tests/test_rules.py` | Locked weights, tier boundaries, R1–R5 hits, R5 confidence gate, trace validation, config rejection |
| `tests/test_scoring.py` | Slide-21 60/High, down-weight, clamp, summary, trace round-trip |
| `tests/test_evaluation.py` | High metrics, insufficient guard, fixture exclusion, ablation deltas, artifact writing |
| `tests/test_pipeline.py` | End-to-end sample contract, DuckDB tables, no-key artifact check, dashboard AppTest |

`tests/test_pipeline.py` runs the sample pipeline once per session and checks 377 wallets, all
required artifacts, the DuckDB warehouse tables, trace validity, absence of the API key in
text artifacts, and a Streamlit `AppTest` smoke run with zero exceptions. No test needs the
network.

CI (`.github/workflows/ci.yml`) installs `requirements.txt` on Python 3.11, runs `ruff check .`
and `pytest -q`; it has no Postgres, Neo4j, or Docker services.

---

## Repository layout

```
ChainLens/
├── config/
│   ├── rules.yaml              # locked weights, tiers, R2/R5 params
│   ├── detection.yaml          # DBSCAN/PageRank/Louvain/cycle params
│   ├── data.yaml               # paths, sampling, window, Etherscan
│   └── known_addresses.json    # documented seed list
├── data/
│   ├── labels/                 # OFAC, darklist, Kaggle
│   ├── sample/                 # offline transaction + label fixtures
│   ├── cache/                  # gitignored per-address cache
│   └── output/                 # gitignored artifacts
├── tests/                      # 7 focused test files
├── fetch_data.py
├── features.py
├── patterns.py
├── scoring.py
├── evaluate.py
├── run.py
├── dashboard.py
├── simplification_plan.md      # v2 design/decision record
├── requirements.txt
└── pyproject.toml
```

---

## Output artifacts

| File | Contents |
|---|---|
| `transactions_raw.parquet` | Normalised Etherscan rows |
| `transactions_clean.parquet` | Deduplicated, filtered, DECIMAL-ETH transactions |
| `wallet_features.parquet` | Wallet features + train-scaled columns |
| `graph_edges.parquet` | Aggregated directed edges with value/count/time |
| `flags.parquet` | Long-form rule/DBSCAN/PageRank/Louvain signals |
| `scores.parquet` | Address, raw score, final score, tier, top driver, versions |
| `traces.parquet` | One JSON trace per scored wallet |
| `evaluation.json` / `.md` | Test metrics, operating points, ablation summary |
| `ablation.csv` | Leave-one-rule-out table |
| `run_info.json` | Mode, seed, window, counts, versions, hashes, git commit |
| `chainlens.duckdb` | Warehouse: transactions, features, scaler, labels, splits, flags, scores, traces, edges |
| `scores.csv` / `traces.csv` | Flat exports for the dashboard/CSV workflow |

---

## Security and ethics

- The Etherscan key lives only in `.env`; `.env` is gitignored. The key is never printed,
  embedded in logs, or written to `run_info.json`. `run_info.json` contains only
  `api_key_present: true/false`.
- The test suite checks that a configured key does not appear in text artifacts.
- `data/cache/` contains public on-chain transaction data and is gitignored; `data/output/`
  is gitignored.
- Rotate the key immediately if it is ever committed or pasted into a screenshot.
- Labels are noisy and time-stamped. OFAC/darklist entries can be stale or overbroad; Kaggle
  negatives are not verified clean. Use the trace and human review, not the score alone.
- R4 and R5 are association/prominence signals, not causal fraud evidence. DBSCAN noise is an
  outlier flag, not guilt.
- Known-address down-weighting is a small, documented seed list; it cannot cover every
  exchange, bridge, router, or token contract.

---

## Viva-ready questions

**Why seven files rather than one engine?**
One file per stage gives a clear input/output contract, keeps each module small enough to
explain, and prevents a `utils.py` dumping ground. The stages still run in one process.

**Why DuckDB?**
The dataset fits in memory, DuckDB gives SQL cleaning and an inspectable warehouse file, and
it removes Postgres/Neo4j services that added ops cost without analytical value.

**Why does the real run have no High-tier predictions?**
Because no active test wallet hit 60 under the locked weights: R1 found no qualifying cycles,
R3 test wallets were not 1-hop from train sanction seeds, and R5 produced no rules at
`min_lift=2.0`. The honest result is F1 0 at High; the Medium operating point and AP are
reported alongside it.

**Why is R5 empty on real data?**
The best observed lift among flagged-consequent rules is 1.407–1.450, below the configured 2.0.
Lowering the gate would make the number look better but would silently change the rule
contract. The sample fixture still demonstrates R5 with four rules.

**How is leakage prevented?**
R3 seeds are train-only; R5 mines only labelled train addresses; the scaler is fitted on train;
test labels are used only in `evaluate.py`; and fixture labels are excluded from real metrics.

**How does DBSCAN affect the score?**
Only through R5: `dbscan_noise` is a boolean Apriori item. DBSCAN itself is not a weighted
rule, and its cluster id is a display flag.

**Why 60, not 65?**
Locked weights: R1 25 + R2 20 + R4 15 = 60. The slide's 65 cannot be reproduced without
changing a weight or adding a signal; the test asserts 60 and High.

**What would you do next?**
Use fresher labels aligned to the transaction window, run the full ~4,300-address scope with
500 tx/address, calibrate R5 with a validation split (not the test split), and add an
exchange/bridge router registry with dated provenance.

---

## License / academic use

This is an academic project. Public-chain data is used for detection research; labels are
attributed in `data/labels/SOURCES.json`. Do not deploy the scores as automated enforcement.
