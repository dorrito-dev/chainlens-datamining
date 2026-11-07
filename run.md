# ChainLens — run commands

All commands run from the repo root: `/home/codez/Desktop/Chain-Analysis/ChainLens-v2`.

Interactive menu (fetch, cache rebuild, sample, datasets, dashboard, checks):

```bash
uv run python full_runner.py
```

## 1. Setup

```bash
uv sync
cp .env.example .env          # then set ETHERSCAN_API_KEY
```

## 2. Full run (real data)

```bash
uv run python run.py --mode real
```

Re-run using the on-disk cache (no API calls):

```bash
uv run python run.py --mode real --skip-fetch
```

Force re-fetch:

```bash
uv run python run.py --mode real --refresh
```

Smaller run / different caps:

```bash
uv run python run.py --mode real --max-addresses 200 --max-txs 200
```

## 3. Sample run (no API key)

```bash
uv run python run.py --mode sample
```

## 4. Verify outputs

```bash
cat data/output/run_info.json | uv run python -m json.tool
cat data/output/evaluation.md
cat data/output/ablation.csv
ls -lh data/output/
```

## 5. Dashboard

```bash
uv run streamlit run dashboard.py --server.headless true --server.port 8501
```

Stop it:

```bash
pkill -f "streamlit run dashboard.py"
```

## 6. Tests and lint

```bash
uv run ruff check patterns.py scoring.py run.py dashboard.py tests/
uv run pytest tests/ -q
```

## 7. View datasets

Shell helper (bash):

```bash
q() { uv run python -c "import duckdb,sys; duckdb.connect('data/output/chainlens.duckdb', read_only=True).sql(sys.argv[1]).show()" "$1"; }
```

List tables and row counts:

```bash
q "show tables"
q "select table_name, estimated_size from duckdb_tables() order by table_name"
```

Preview any table:

```bash
q "describe risk_scores"
q "select * from risk_scores using sample 10 rows"
```

### Pipeline stages

```bash
# raw fetch (no value_eth)
q "select * from transactions_raw using sample 5 rows"

# cleaned transactions
q "select * from transactions using sample 5 rows"

# wallet features
q "select address, tx_count_total, fanout_ratio, total_value_eth_scaled from wallet_features limit 10"

# per-wallet signals (long format)
q "select signal_name, count(*) n from wallet_flags group by 1 order by n desc"
q "select * from wallet_flags where address='0x9a86deb3b58b18dc192b5ad31066dc6fe3e7c83d'"

# graph edges
q "select * from graph_edges order by total_value_eth desc limit 10"

# labels / splits / known addresses / scaler
q "select source, label, count(*) from labels group by 1,2 order by 3 desc"
q "select split, count(*) from splits group by 1"
q "select * from known_addresses"
q "select * from feature_scaler"
```

### Scores and explanations

```bash
q "select tier, count(*) n, round(avg(score),1) avg_score from risk_scores group by 1 order by n desc"
q "select address, score, tier, top_driver from risk_scores order by score desc limit 20"
q "select address, raw_score, score, top_driver from risk_scores where address='0x9a86deb3b58b18dc192b5ad31066dc6fe3e7c83d'"
q "select trace_json from score_trace where address='0x9a86deb3b58b18dc192b5ad31066dc6fe3e7c83d'"
```

Flat per-rule breakdown:

```bash
uv run python -c "
import pandas as pd; pd.set_option('display.width', 250)
t = pd.read_csv('data/output/traces.csv', dtype={'address': str})
print(t[t.address == '0x9a86deb3b58b18dc192b5ad31066dc6fe3e7c83d'].T)
"
```

### Raw inputs

```bash
head -5 data/labels/ofac_eth.txt
column -s, -t < data/labels/kaggle_fraud.csv | head -20
uv run python -m json.tool data/labels/darklist.json | less
cat data/labels/SOURCES.json
cat config/sanctions.json
cat config/rules.yaml
```

Etherscan cache (per-address parquet + meta.json):

```bash
uv run python -c "
import pandas as pd; pd.set_option('display.width', 200)
print(pd.read_parquet('data/cache/addresses/0x0000000000000000000000000000000000000000.parquet').head(10))
"
cat data/cache/addresses/0x0000000000000000000000000000000000000000.meta.json
ls data/cache/addresses | wc -l
```

### Files directly (outside DuckDB)

```bash
libreoffice --calc data/output/scores.csv
libreoffice --calc data/output/traces.csv

uv run python -c "
import pandas as pd
df = pd.read_parquet('data/output/scores.parquet')
print(df.nlargest(20, 'score'))
"
```

Any parquet through the same helper:

```bash
q "describe select * from 'data/output/flags.parquet'"
q "select * from 'data/output/flags.parquet' limit 10"
```

### Output artifacts

| File | Content |
|---|---|
| `transactions_raw.parquet` | fetched rows, 379,023 |
| `transactions_clean.parquet` | deduped/validated, 368,113 |
| `wallet_features.parquet` | 81,756 wallets, raw + scaled features |
| `flags.parquet` | 103,671 wallet×signal rows |
| `scores.parquet` / `.csv` | 81,756 scores, tier, driver |
| `traces.parquet` / `.csv` | per-rule hits and contributions |
| `graph_edges.parquet` | 99,706 edges |
| `labels.parquet` | 5,446 labels |
| `splits.parquet` | 4,817 train/test rows |
| `known_addresses.parquet` | 15 excluded addresses |
| `evaluation.json` / `.md` | P/R/F1 and average precision |
| `ablation.csv` | per-rule score contribution |
| `run_info.json` | run metadata + pattern counters |
| `context_blocks.json` | sampled block heights |
| `chainlens.duckdb` | all of the above in one file |

`data/processed/` is stale sample-mode output; ignore it.
