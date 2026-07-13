#!/bin/bash
set -euo pipefail

INIT_FROM=""
CONFIGS=()
EXPAND_OUTPUT_DIR=".generated/downstream_suites"
RESULT_OUTPUT_DIR=".generated/downstream_results/sequence_$$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --config) CONFIGS+=("$2"); shift 2 ;;
    --expand-output-dir) EXPAND_OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--init-from PRETRAIN_RUN_OR_CKPT] --config cfg_or_suite.yaml [--config cfg2.yaml ...]"
      echo ""
      echo "Each --config may be either a normal per-task downstream YAML or a"
      echo "downstream_suite YAML expanded into one generated config per task."
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "ERROR: at least one --config is required" >&2
  exit 1
fi

EXPANDED_CONFIGS=()
for CFG in "${CONFIGS[@]}"; do
  EXPANDED_OUTPUT="$(python3 scripts/expand_downstream_suite.py \
    --config "$CFG" \
    --output-dir "$EXPAND_OUTPUT_DIR")" || {
      echo "ERROR: failed to expand downstream config: $CFG" >&2
      exit 1
    }
  while IFS= read -r EXPANDED; do
    [[ -n "$EXPANDED" ]] && EXPANDED_CONFIGS+=("$EXPANDED")
  done <<< "$EXPANDED_OUTPUT"
done
CONFIGS=("${EXPANDED_CONFIGS[@]}")

PIN_COMMAND=(python3 scripts/downstream_protocol.py pin-source)
if [[ -n "$INIT_FROM" ]]; then
  PIN_COMMAND+=(--source "$INIT_FROM")
fi
for CFG in "${CONFIGS[@]}"; do
  PIN_COMMAND+=(--config "$CFG")
done
PIN_OUTPUT="$("${PIN_COMMAND[@]}")" || {
  echo "ERROR: failed to pin the downstream source checkpoint" >&2
  exit 1
}
IFS=$'\t' read -r SOURCE_REQUESTED SOURCE_RESOLVED SOURCE_STEP <<< "$PIN_OUTPUT"
if [[ -z "$SOURCE_REQUESTED" || -z "$SOURCE_RESOLVED" || -z "$SOURCE_STEP" ]]; then
  echo "ERROR: invalid pinned source response: $PIN_OUTPUT" >&2
  exit 1
fi

mkdir -p "$RESULT_OUTPUT_DIR"
RESULT_FILES=()

echo "============================================================"
echo "[sequence] SOURCE PINNED"
echo "[sequence] source_checkpoint_requested=$SOURCE_REQUESTED"
echo "[sequence] source_checkpoint_resolved=$SOURCE_RESOLVED"
echo "[sequence] source_checkpoint_step=$SOURCE_STEP"
echo "[sequence] source_checkpoint_resolved_once=true"
echo "[sequence] task_source_policy=pinned_same_checkpoint"
echo "============================================================"

i=0
for CFG in "${CONFIGS[@]}"; do
  i=$((i + 1))
  RESULT_FILE="$RESULT_OUTPUT_DIR/task_${i}.json"
  RESULT_FILES+=("$RESULT_FILE")
  echo "============================================================"
  echo "[sequence] START ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] source_checkpoint_resolved=$SOURCE_RESOLVED"
  echo "[sequence] source_checkpoint_step=$SOURCE_STEP"
  echo "[sequence] task_source_policy=pinned_same_checkpoint"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
  python3 scripts/downstream_finetune_jax.py \
    --config "$CFG" \
    --init-from "$SOURCE_RESOLVED" \
    --source-requested "$SOURCE_REQUESTED" \
    --expected-source-path "$SOURCE_RESOLVED" \
    --expected-source-step "$SOURCE_STEP" \
    --result-json "$RESULT_FILE"
  if [[ ! -s "$RESULT_FILE" ]]; then
    echo "ERROR: task did not produce a result JSON: $RESULT_FILE" >&2
    exit 1
  fi
  echo "============================================================"
  echo "[sequence] DONE ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
done
echo "[sequence] ALL DONE (${#CONFIGS[@]} configs)"
SUMMARY_COMMAND=(python3 scripts/downstream_protocol.py summary)
for RESULT_FILE in "${RESULT_FILES[@]}"; do
  SUMMARY_COMMAND+=(--result-json "$RESULT_FILE")
done
"${SUMMARY_COMMAND[@]}"
