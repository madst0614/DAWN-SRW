#!/bin/bash
set -euo pipefail

INIT_FROM=""
CONFIGS=()
EXPAND_OUTPUT_DIR=".generated/downstream_suites"

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

i=0
for CFG in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo "============================================================"
  echo "[sequence] START ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] common init-from: ${INIT_FROM:-<none>}"
  echo "[sequence] policy: independent transfer from the same source checkpoint"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
  if [[ -n "$INIT_FROM" ]]; then
    python3 scripts/downstream_finetune_jax.py \
      --config "$CFG" \
      --init-from "$INIT_FROM"
  else
    python3 scripts/downstream_finetune_jax.py \
      --config "$CFG"
  fi
  echo "============================================================"
  echo "[sequence] DONE ${i}/${#CONFIGS[@]} config: $CFG"
  echo "[sequence] time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
done
echo "[sequence] ALL DONE (${#CONFIGS[@]} configs)"
