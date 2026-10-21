-- ChainLens Phase 6 star schema (reporting presentation artifact).
-- The pipeline's critical path remains address_features / address_flags /
-- risk_scores / score_trace. This schema is populated by the one-off
-- pipeline/07_star_schema.py script only.

CREATE TABLE IF NOT EXISTS fact_transactions (
    tx_hash         TEXT PRIMARY KEY,
    block_number    BIGINT NOT NULL,
    block_timestamp TIMESTAMPTZ NOT NULL,
    from_address    TEXT NOT NULL,
    to_address      TEXT NOT NULL,
    value_eth       NUMERIC(38,18) NOT NULL,   -- DECIMAL, not float; wei-safe
    gas             BIGINT,
    gas_price       NUMERIC(78,0),
    is_contract_to  BOOLEAN,
    receipt_status  SMALLINT
);

CREATE INDEX IF NOT EXISTS fact_transactions_block_number_idx
    ON fact_transactions (block_number);
CREATE INDEX IF NOT EXISTS fact_transactions_from_address_idx
    ON fact_transactions (from_address);
CREATE INDEX IF NOT EXISTS fact_transactions_to_address_idx
    ON fact_transactions (to_address);

CREATE TABLE IF NOT EXISTS dim_address (
    address        TEXT PRIMARY KEY,
    is_contract    BOOLEAN,
    is_known       BOOLEAN,
    known_category TEXT,
    first_seen     TIMESTAMPTZ,
    last_seen      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dim_label (
    address    TEXT NOT NULL,
    label      TEXT NOT NULL,
    source     TEXT NOT NULL,
    confidence NUMERIC(3,2),
    as_of      DATE NOT NULL,
    PRIMARY KEY (address, source, label)
);

CREATE INDEX IF NOT EXISTS dim_label_address_idx ON dim_label (address);
