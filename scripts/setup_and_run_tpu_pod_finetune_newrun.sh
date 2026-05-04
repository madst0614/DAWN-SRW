#!/bin/bash
set -euo pipefail

GH_TOKEN="${GH_TOKEN:-}"
if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi
BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
INIT_FROM="${INIT_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
WORK_DIR="$HOME/dawn-spatial"

if [ -n "$INIT_FROM" ] && [ -n "$RESUME_FROM" ]; then
    echo "ERROR: INIT_FROM and RESUME_FROM are mutually exclusive" >&2
    exit 1
fi

echo "============================================"
echo "Host $(hostname) — Finetune/new-run TPU setup"
echo "  Branch: $BRANCH"
echo "  Config: $CONFIG"
if [ -n "$INIT_FROM" ]; then echo "  Init from: $INIT_FROM"; fi
if [ -n "$RESUME_FROM" ]; then echo "  Resume from: $RESUME_FROM"; fi
echo "============================================"

echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

echo "[2/4] Syncing repo (branch: $BRANCH)..."
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    git fetch origin "$BRANCH" --depth 1
    git checkout -B "$BRANCH" FETCH_HEAD
else
    cd "$HOME"
    rm -rf dawn-spatial
    git clone -b "$BRANCH" --single-branch --depth 1 "$REPO_URL" dawn-spatial
    cd dawn-spatial
fi

echo "[3/4] Skipping standalone JAX TPU preflight; training process will initialize the slice."

echo "[4/4] Starting finetune/new-run in tmux session 'train'..."
cd "$WORK_DIR"
tmux kill-session -t train 2>/dev/null || true

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

CMD="python3 scripts/train_jax_finetune_newrun.py --config '$CONFIG'"
if [ -n "$INIT_FROM" ]; then
    CMD="$CMD --init-from '$INIT_FROM'"
fi
if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume-from '$RESUME_FROM'"
fi
CMD="$CMD $TRAIN_ARGS"

tmux new-session -d -s train \
    "export XLA_DUMP_DIR='$XLA_DUMP_DIR'; export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; export XLA_FLAGS='$XLA_FLAGS'; $CMD 2>&1 | tee ~/train.log; echo 'Training finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Monitor: tail -f ~/train.log"
