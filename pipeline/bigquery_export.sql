-- ChainLens Phase 2 BigQuery export (label-anchored mode).
--
-- COMMITTED ONLY. This file is never executed automatically. Before running:
--   1. Run the overlap gate (pipeline/00_overlap.py) and record its decision.
--   2. Run pipeline/00_preflight.py (dry run) and check bytes scanned + USD.
--   3. Only then execute this script once with the parameters below.
--
-- Required query parameters (passed by the one-off export runner):
--   @block_start, @block_end           candidate window from config/data.yaml
--   @ref_block_start, @ref_block_end   wider range for 1-2 hop expansion
--   @label_addresses                   ARRAY<STRING> of all ground-truth labels
--
-- Check the "bytes scanned" estimate in the BigQuery console/dry run before
-- running. `receipt_status = 1` keeps successful transactions only; reverted or
-- failed transactions have status 0 (field available post-Byzantium).
-- Value is exported as STRING so wei never passes through INT64/FLOAT64.

EXPORT DATA
  OPTIONS (
    uri = 'gs://YOUR_BUCKET/chainlens/transactions-*.parquet',
    format = 'PARQUET',
    overwrite = true
  )
AS
WITH seeds AS (
  SELECT DISTINCT LOWER(address) AS address
  FROM UNNEST(@label_addresses) AS address
),
hop1 AS (
  SELECT DISTINCT
    IF(t.from_address IN (SELECT address FROM seeds), t.to_address, t.from_address) AS address
  FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
  WHERE t.block_number BETWEEN @ref_block_start AND @ref_block_end
    AND (
      t.from_address IN (SELECT address FROM seeds)
      OR t.to_address IN (SELECT address FROM seeds)
    )
),
hop2 AS (
  SELECT DISTINCT
    IF(t.from_address IN (SELECT address FROM hop1), t.to_address, t.from_address) AS address
  FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
  WHERE t.block_number BETWEEN @ref_block_start AND @ref_block_end
    AND (
      t.from_address IN (SELECT address FROM hop1)
      OR t.to_address IN (SELECT address FROM hop1)
    )
),
targets AS (
  SELECT address FROM seeds
  UNION DISTINCT
  SELECT address FROM hop1
  UNION DISTINCT
  SELECT address FROM hop2
)
SELECT
  `hash`                    AS tx_hash,
  block_number,
  block_timestamp,
  from_address,
  to_address,
  CAST(value AS STRING)     AS value_wei,
  gas,
  gas_price,
  receipt_status,
  receipt_contract_address,
  transaction_index
FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
WHERE t.block_number BETWEEN @block_start AND @block_end
  AND t.receipt_status = 1
  AND (
    t.from_address IN (SELECT address FROM targets)
    OR t.to_address IN (SELECT address FROM targets)
  );

-- Contract addresses for the same block range. Phase 2 left-joins this against
-- transaction counterparties to populate `is_contract`; EOAs are excluded by
-- the clean step, which removes any address seen as a transaction sender.
EXPORT DATA
  OPTIONS (
    uri = 'gs://YOUR_BUCKET/chainlens/contracts-*.parquet',
    format = 'PARQUET',
    overwrite = true
  )
AS
SELECT
  address,
  block_number,
  block_timestamp
FROM `bigquery-public-data.crypto_ethereum.contracts`
WHERE block_number BETWEEN @block_start AND @block_end;
