COMPOSE ?= docker compose
APP_ENV = $(if $(CHAINLENS_SAMPLE_MODE),-e CHAINLENS_SAMPLE_MODE=$(CHAINLENS_SAMPLE_MODE),)
APP_EXEC ?= $(COMPOSE) exec -T $(APP_ENV) app
PYTHON ?= python
RUNNER ?= scripts/run.sh

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: up down doctor doctor-local overlap overlap-local preflight preflight-local \
	schema schema-local sample sample-local ingest ingest-local features features-local \
	features-check features-check-local detect detect-local score score-local score-check \
	score-check-local baseline baseline-local ablate ablate-local star-schema star-schema-local \
	graph-load graph-load-local graph-reset graph-reset-local app app-local \
	test test-local lint lint-local clean

up: ## Build the app image, start all services, and wait for health
	$(COMPOSE) up -d --build --wait

down: ## Stop containers; named volumes are preserved
	$(COMPOSE) down

doctor: ## Check env, Neo4j secret sync, and both database connections
	$(RUNNER) doctor

doctor-local: ## Doctor with host Python (used by CI)
	CHAINLENS_RUNNER=local $(RUNNER) doctor

overlap: ## Run the label-overlap gate and record the window decision
	$(RUNNER) overlap --update-config

overlap-local: ## Overlap gate with host Python (CI uses sample mode)
	CHAINLENS_RUNNER=local $(RUNNER) overlap --update-config

preflight: ## BigQuery dry-run cost gate (manual, requires credentials)
	$(RUNNER) preflight

preflight-local: ## Preflight with host Python
	CHAINLENS_RUNNER=local $(RUNNER) preflight

schema: ## Apply sql/schema.sql inside the app container (idempotent)
	$(APP_EXEC) python -c "from chainlens.db import apply_schema, get_pg_engine; apply_schema(get_pg_engine())"

schema-local: ## Apply schema on the host Python (used by CI)
	$(PYTHON) -c "from chainlens.db import apply_schema, get_pg_engine; apply_schema(get_pg_engine())"

sample: ## Generate sample + seed Postgres inside the app container
	$(APP_EXEC) python pipeline/00_generate_sample.py

sample-local: ## Generate sample + seed Postgres with host Python (used by CI)
	$(PYTHON) pipeline/00_generate_sample.py

ingest: ## Run labels, known, split, clean through scripts/run.sh
	$(RUNNER) ingest

ingest-local: ## Same shared entrypoint, forced to host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) ingest

features: ## Build address_features with a train-only scaler
	$(RUNNER) features

features-local: ## Same feature build, forced to host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) features

features-check: ## Print feature/scaler provenance and evaluable-positive count
	$(RUNNER) features-check

features-check-local: ## Feature check with host Python
	CHAINLENS_RUNNER=local $(RUNNER) features-check

detect: ## Run DBSCAN, graph signals, train-only Apriori, and seed scores
	$(RUNNER) detect

detect-local: ## Phase 4 detection with host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) detect

score: ## Apply locked R1-R5 weights and write current scores + score traces
	$(RUNNER) score

score-local: ## Phase 5 scoring with host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) score

score-check: ## Verify score coverage, versions, and no unscored rows
	$(RUNNER) score-check

score-check-local: ## Score postcondition check with host Python
	CHAINLENS_RUNNER=local $(RUNNER) score-check

baseline: ## Snapshot current scores/traces for Phase 6 ablation
	$(RUNNER) baseline

baseline-local: ## Baseline snapshot with host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) baseline

ablate: ## Run the per-rule ablation detection test against the baseline
	$(RUNNER) ablate

ablate-local: ## Ablation with host Python (CI)
	CHAINLENS_RUNNER=local $(RUNNER) ablate

star-schema: ## One-off: populate the reporting star schema
	$(RUNNER) star-schema

star-schema-local: ## One-off star schema with host Python
	CHAINLENS_RUNNER=local $(RUNNER) star-schema

graph-load: ## Rebuild the curated Neo4j showcase subgraph (idempotent)
	$(RUNNER) graph-load

graph-load-local: ## Curated Neo4j load with host Python
	CHAINLENS_RUNNER=local $(RUNNER) graph-load

graph-reset: ## Delete only :ChainLens nodes/relationships from Neo4j
	$(RUNNER) graph-reset

graph-reset-local: ## Delete :ChainLens nodes with host Python
	CHAINLENS_RUNNER=local $(RUNNER) graph-reset

app: ## Launch Streamlit on http://localhost:8501 (NetworkX graph tab)
	$(RUNNER) app

app-local: ## Launch Streamlit with host Python
	CHAINLENS_RUNNER=local $(RUNNER) app

test: ## Run pytest inside the app container
	$(APP_EXEC) python -m pytest -q

test-local: ## Run pytest with host Python (used by CI)
	$(PYTHON) -m pytest -q

lint: ## Run ruff inside the app container
	$(APP_EXEC) python -m ruff check .

lint-local: ## Run ruff with host Python (used by CI)
	$(PYTHON) -m ruff check .

clean: ## Stop containers AND delete named volumes (destructive)
	@echo "WARNING: this deletes the pgdata and neo4jdata volumes."
	@printf "Type 'yes' to continue: " && read answer && [ "$$answer" = "yes" ] || (echo "Aborted."; exit 1)
	$(COMPOSE) down -v
