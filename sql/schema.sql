-- ChainLens PostgreSQL schema contracts (Phase 1).
-- Addresses are stored lowercase. Wei values use NUMERIC, never BIGINT.
-- All statements are idempotent so `make schema` can be re-run safely.

CREATE TABLE IF NOT EXISTS labels (
    address     TEXT NOT NULL,
    label       TEXT NOT NULL,          -- e.g. 'sanctioned', 'fraud', 'darklist'
    source      TEXT NOT NULL,          -- 'ofac' | 'darklist' | 'kaggle'
    confidence  NUMERIC(3,2) DEFAULT 1.0,
    as_of       DATE NOT NULL,          -- sanction lists change; store snapshot date
    PRIMARY KEY (address, source, label)
);

CREATE TABLE IF NOT EXISTS label_splits (
    address     TEXT PRIMARY KEY,
    split       TEXT NOT NULL CHECK (split IN ('train','test')),
    seed        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS known_addresses (
    address     TEXT PRIMARY KEY,
    category    TEXT NOT NULL CHECK (category IN
                ('exchange','bridge','mixer','contract','other')),
    note        TEXT
);

-- Feature table only. Labels and `flagged` deliberately live elsewhere.
CREATE TABLE IF NOT EXISTS address_features (
    address              TEXT PRIMARY KEY,
    tx_count_out         INTEGER,
    tx_count_in          INTEGER,
    tx_count_total       INTEGER,
    unique_counterparties INTEGER,
    fanout_ratio         NUMERIC(18,10),
    total_value_eth      NUMERIC(38,18),
    avg_value_eth        NUMERIC(38,18),
    median_value_eth     NUMERIC(38,18),
    net_flow_eth         NUMERIC(38,18),   -- NOT "balance"; window net flow only
    token_transfer_count INTEGER,
    is_contract          BOOLEAN,
    first_seen           TIMESTAMPTZ,
    last_seen            TIMESTAMPTZ,
    active_days          INTEGER,
    round_number_ratio   NUMERIC(6,4),
    -- scaled counterparts for DBSCAN (log1p + StandardScaler output)
    total_value_eth_scaled   NUMERIC(10,6),
    tx_count_out_scaled      NUMERIC(10,6),
    tx_count_in_scaled       NUMERIC(10,6),
    unique_counterparties_scaled NUMERIC(10,6),
    fanout_ratio_scaled      NUMERIC(10,6),
    scaler_version       TEXT
);

-- Phase 3 additions. ALTERs are idempotent for databases created in Phase 1/2.
ALTER TABLE address_features ADD COLUMN IF NOT EXISTS tx_count_total INTEGER;
ALTER TABLE address_features ADD COLUMN IF NOT EXISTS scaler_version TEXT;
ALTER TABLE address_features
    ALTER COLUMN fanout_ratio TYPE NUMERIC(18,10);

CREATE TABLE IF NOT EXISTS feature_scaler (
    feature_name     TEXT PRIMARY KEY,
    transform        TEXT NOT NULL,       -- 'log1p_std' | 'std'
    mean             NUMERIC NOT NULL,
    std              NUMERIC NOT NULL,
    fitted_on_split  TEXT NOT NULL CHECK (fitted_on_split IN ('train')),
    row_count        INTEGER NOT NULL,
    scaler_version   TEXT NOT NULL,
    fitted_at        TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE feature_scaler
    DROP CONSTRAINT IF EXISTS feature_scaler_fitted_on_split_check;
ALTER TABLE feature_scaler
    ADD CONSTRAINT feature_scaler_fitted_on_split_check
    CHECK (fitted_on_split IN ('train', 'train_fallback_full_std'));

CREATE TABLE IF NOT EXISTS feature_scaler_dropped (
    feature_name     TEXT PRIMARY KEY,
    reason           TEXT NOT NULL,
    dropped_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS address_flags (
    address       TEXT NOT NULL,
    signal_name   TEXT NOT NULL,   -- 'dbscan_cluster','pagerank','cycle','fanout','sanctioned_1hop','apriori','louvain'
    signal_value  NUMERIC,
    evidence      JSONB,
    created_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (address, signal_name)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    address        TEXT PRIMARY KEY,
    raw_score      INTEGER NOT NULL,   -- sum of rule contributions (can exceed 100)
    score          INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    tier           TEXT CHECK (tier IN ('low','medium','high','unscored')),
    top_driver     TEXT,
    rules_version  TEXT NOT NULL,
    scored_at      TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE risk_scores
    DROP CONSTRAINT IF EXISTS risk_scores_tier_check;
ALTER TABLE risk_scores
    ADD CONSTRAINT risk_scores_tier_check
    CHECK (tier IN ('low','medium','high','unscored'));

CREATE TABLE IF NOT EXISTS score_trace (
    address        TEXT NOT NULL,
    rules_version  TEXT NOT NULL,
    trace          JSONB NOT NULL,   -- per-rule: hit, inputs, weight, contribution
    created_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (address, rules_version)
);

CREATE TABLE IF NOT EXISTS data_manifest (
    file_path    TEXT PRIMARY KEY,
    sha256       TEXT NOT NULL,
    row_count    INTEGER,
    block_start  BIGINT,
    block_end    BIGINT,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Phase 2 additions: record BigQuery dry-run cost and gate decisions without
-- inventing a second manifest table. ADD COLUMN IF NOT EXISTS keeps this
-- migration idempotent for databases created by Phase 1.
ALTER TABLE data_manifest ADD COLUMN IF NOT EXISTS bytes_processed BIGINT;
ALTER TABLE data_manifest ADD COLUMN IF NOT EXISTS estimated_usd NUMERIC(18,6);
ALTER TABLE data_manifest ADD COLUMN IF NOT EXISTS metadata JSONB;

-- ---------------------------------------------------------------------------
-- Phase 7 operations metadata.
--
-- pipeline_runs is operational, NOT analytical. It is deliberately absent from
-- fact_transactions and the star schema: it records UI/CLI stage invocations so
-- the operations console is stateless across browser refreshes and restarts.
--
-- `stage` includes 'admin' for destructive Admin-page actions. Those rows are
-- written after the action succeeds (same transaction where possible) and use
-- `error` as a human-readable description of what was done.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               SERIAL PRIMARY KEY,
    stage            TEXT NOT NULL CHECK (stage IN
                       ('sample','ingest','features','detect','score','admin')),
    mode             TEXT NOT NULL CHECK (mode IN ('sample','real')),
    status           TEXT NOT NULL CHECK (status IN ('running','success','failed')),
    started_at       TIMESTAMPTZ DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    exit_code        INTEGER,
    row_counts       JSONB,
    version_snapshot JSONB,
    log_path         TEXT,
    error            TEXT,
    triggered_by     TEXT DEFAULT 'ui'
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started
    ON pipeline_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stage
    ON pipeline_runs(stage, status);
