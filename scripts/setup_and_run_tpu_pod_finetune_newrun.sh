#!/bin/bash
# =============================================================================
# Runs on each TPU worker. Starts fine-tune/new-run in tmux.
# Assumes launch_tpu_pod_finetune_newrun.sh has already force-synced the repo.
# =============================================================================
set -euo pipefail

BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
INIT_FROM="${INIT_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
WORK_DIR="$HOME/dawn-spatial"
TRAIN_SCRIPT="scripts/train_jax_finetune_newrun.py"

echo "============================================"
echo "Host $(hostname) — fine-tune/new-run setup"
echo "  Branch: $BRANCH"
echo "  Config: $CONFIG"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Resume: ${RESUME_FROM:-<none>}"
echo "============================================"

cd "$WORK_DIR"

echo "[verify] repo HEAD: $(git rev-parse --short HEAD) $(git log -1 --oneline)"
if grep -R "train_jax_init_newrun" scripts/launch_tpu_pod_finetune_newrun.sh scripts/setup_and_run_tpu_pod_finetune_newrun.sh 2>/dev/null; then
  echo "ERROR: finetune launcher/setup still references train_jax_init_newrun.py" >&2
  exit 2
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "ERROR: missing $TRAIN_SCRIPT" >&2
  exit 3
fi

echo "[1/3] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

echo "[2/3] Preparing TPU/XLA environment..."
XLA_DUMP_DIR="${XLA_DUMP_DIR:-/tmp/xla_dump_train}"
mkdir -p "$XLA_DUMP_DIR"
export XLA_DUMP_DIR
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
if [ -z "${XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
else
  export XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
fi

echo "[3/3] Starting tmux session 'train'..."
tmux kill-session -t train 2>/dev/null || true
pkill -f train_jax_init_newrun.py 2>/dev/null || true
pkill -f train_jax_finetune_newrun.py 2>/dev/null || true

CMD=(python3 "$TRAIN_SCRIPT" --config "$CONFIG")
if [ -n "$INIT_FROM" ]; then
  CMD+=(--init-from "$INIT_FROM")
fi
if [ -n "$RESUME_FROM" ]; then
  CMD+=(--resume-from "$RESUME_FROM")
fi
# shellcheck disable=SC2206
EXTRA_ARGS=( $TRAIN_ARGS )
CMD+=("${EXTRA_ARGS[@]}")

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "  Command: $CMD_STR"
echo "  Log: ~/train.log"

tmux new-session -d -s train \
  "cd '$WORK_DIR'; export XLA_DUMP_DIR='$XLA_DUMP_DIR'; export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; export XLA_FLAGS='$XLA_FLAGS'; $CMD_STR 2>&1 | tee ~/train.log; echo 'Training finished. Press enter to close.'; read"

echo "tmux session 'train' started."
echo "Attach: tmux attach -t train"
echo "Monitor: tail -f ~/train.log"
