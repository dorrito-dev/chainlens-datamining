#!/usr/bin/env bash
# Shared entrypoint for Phase 2-4 pipeline stages.
#
# The actual command for each stage is written once here. The runner is chosen
# from the environment:
#   - inside the app container (/.dockerenv exists) -> python
#   - CHAINLENS_RUNNER=local                       -> python
#   - otherwise                                    -> docker compose exec app python
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run.sh <doctor|overlap|preflight|labels|known|split|clean|ingest|features|features-check|cluster|graph|apriori|seed-scores|detect|score|score-check|baseline|ablate|star-schema|graph-load|graph-reset|app> [args...]" >&2
  exit 2
fi

STAGE="$1"
shift

DOCKER_EXEC_ENV=()
for name in CHAINLENS_SAMPLE_MODE BQ_MAX_BYTES BIGQUERY_PROJECT; do
  if [[ -n "${!name:-}" ]]; then
    DOCKER_EXEC_ENV+=(-e "${name}=${!name}")
  fi
done

if [[ "${CHAINLENS_RUNNER:-}" == "local" ]]; then
  RUNNER=(python)
elif [[ -f /.dockerenv ]] || [[ "${CHAINLENS_RUNNER:-}" == "container" ]]; then
  RUNNER=(python)
else
  RUNNER=(docker compose exec -T "${DOCKER_EXEC_ENV[@]}" app python)
fi

run_stage() {
  local stage="$1"
  case "$stage" in
    labels) "${RUNNER[@]}" pipeline/01_ingest.py labels ;;
    known) "${RUNNER[@]}" pipeline/01_ingest.py known ;;
    split) "${RUNNER[@]}" pipeline/01_ingest.py split ;;
    clean) "${RUNNER[@]}" pipeline/02_clean.py ;;
    cluster) "${RUNNER[@]}" pipeline/03_cluster.py ;;
    graph) "${RUNNER[@]}" pipeline/04_graph.py ;;
    apriori) "${RUNNER[@]}" pipeline/05_apriori.py ;;
    seed-scores) "${RUNNER[@]}" pipeline/04b_seed_scores.py ;;
    graph-load) "${RUNNER[@]}" pipeline/06_graph_load.py ;;
    graph-reset) "${RUNNER[@]}" pipeline/06b_graph_reset.py ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
}

case "$STAGE" in
  doctor)
    "${RUNNER[@]}" scripts/doctor.py "$@"
    ;;
  overlap)
    "${RUNNER[@]}" pipeline/00_overlap.py "$@"
    ;;
  preflight)
    "${RUNNER[@]}" pipeline/00_preflight.py "$@"
    ;;
  features)
    "${RUNNER[@]}" pipeline/03_features.py "$@"
    ;;
  features-check)
    "${RUNNER[@]}" pipeline/03_features.py --check "$@"
    ;;
  detect)
    for stage in cluster graph apriori seed-scores; do
      echo "==> $stage"
      run_stage "$stage"
    done
    ;;
  score)
    "${RUNNER[@]}" pipeline/06_score.py "$@"
    ;;
  score-check)
    "${RUNNER[@]}" pipeline/06_score.py --check "$@"
    ;;
  baseline)
    "${RUNNER[@]}" pipeline/07_baseline.py "$@"
    ;;
  ablate)
    "${RUNNER[@]}" pipeline/07_ablation.py "$@"
    ;;
  star-schema)
    "${RUNNER[@]}" pipeline/07_star_schema.py "$@"
    ;;
  app)
    "${RUNNER[@]}" -m streamlit run app/streamlit_app.py \
      --server.address 0.0.0.0 --server.port 8501 --server.headless true "$@"
    ;;
  graph-load|graph-reset|cluster|graph|apriori|seed-scores)
    run_stage "$STAGE" "$@"
    ;;
  ingest)
    for stage in labels known split clean; do
      echo "==> $stage"
      run_stage "$stage"
    done
    ;;
  labels|known|split|clean)
    run_stage "$STAGE" "$@"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
