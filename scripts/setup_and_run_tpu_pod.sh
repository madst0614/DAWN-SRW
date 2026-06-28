#!/bin/bash
# =============================================================================
# TPU Pod Setup + Training Script (runs on each worker)
# =============================================================================
# Expects BRANCH and CONFIG passed as environment variables from the launcher.
#
# Usage (direct):
#   BRANCH=main CONFIG=configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml \
#     bash scripts/setup_and_run_tpu_pod.sh
#
# Usually invoked via launch_tpu_pod.sh which sets env vars automatically.
# =============================================================================

set -euo pipefail

GH_TOKEN="${GH_TOKEN:-}"
if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/DAWN-SRW.git"
else
    REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
fi
BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_jax.py}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
WORK_DIR="$HOME/DAWN-SRW"
export PYTHONUNBUFFERED=1

echo "=== TPU worker startup ==="
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date -Is)"
echo "PWD=$(pwd)"
echo "BRANCH=$(git branch --show-current 2>/dev/null || true)"
echo "COMMIT=$(git rev-parse HEAD 2>/dev/null || true)"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "CONFIG=$CONFIG"
echo "TRAIN_ARGS=$TRAIN_ARGS"
echo "PYTHON=$(which python3)"
echo "PYTHON_VERSION=$(python3 --version)"

echo "============================================"
echo "Host $(hostname) -- Setting up TPU Pod training"
echo "  Branch: $BRANCH"
echo "  Config: $CONFIG"
echo "  Train script: $TRAIN_SCRIPT"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "CONFIG=$CONFIG"
echo "TRAIN_ARGS=$TRAIN_ARGS"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date -Is)"
echo "============================================"

# 1. Install dependencies (all workers)
echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax orbax-checkpoint==0.11.24 numpy pyyaml gcsfs conllu transformers matplotlib -q
python3 -c "import importlib.metadata as m; print('  orbax-checkpoint version: ' + m.version('orbax-checkpoint'))"

# 2. Deploy code via git (clone or update)
echo "[2/4] Syncing repo (branch: $BRANCH)..."
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
        echo "  Already on branch $BRANCH, pulling latest..."
        git pull origin "$BRANCH" --ff-only || true
    else
        echo "  Switching to branch $BRANCH..."
        git fetch origin "$BRANCH" --depth 1
        git checkout -B "$BRANCH" FETCH_HEAD
    fi
else
    echo "  Fresh clone (branch: $BRANCH)..."
    rm -rf "$WORK_DIR"
    git clone -b "$BRANCH" --single-branch --depth 1 "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

echo "=== TPU worker repo ready ==="
echo "PWD=$(pwd)"
echo "BRANCH=$(git branch --show-current 2>/dev/null || true)"
echo "COMMIT=$(git rev-parse HEAD 2>/dev/null || true)"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "CONFIG=$CONFIG"
echo "TRAIN_ARGS=$TRAIN_ARGS"

# 3. Skip standalone JAX preflight.
#
# On multi-host TPU pods a short-lived standalone JAX process can initialize
# PJRT, print device info, and then abort during teardown with:
#   GetSliceInfo can only be invoked after a slice is built...
# The real training process below performs the same backend/device checks and
# keeps the slice alive, so avoid opening a throwaway slice here.
echo "[3/4] Skipping standalone JAX TPU preflight; train_jax.py will verify devices."

# 4. Launch training in tmux (survives SSH disconnect)
echo "[4/4] Starting training in tmux session 'train'..."
echo "  Config: $CONFIG"
echo "  Train script: $TRAIN_SCRIPT"
echo "  Train args: ${TRAIN_ARGS:-}"
echo "  Host: $(hostname)"
echo "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Log: ~/train.log"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "CONFIG=$CONFIG"
echo "TRAIN_ARGS=$TRAIN_ARGS"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date -Is)"

cd "$WORK_DIR"

# Kill existing train session if any
tmux kill-session -t train 2>/dev/null || true

# Keep default logging quiet for normal runs. XLA HLO dumps are opt-in for
# OOM/debug runs because they can create large amounts of text.
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

XLA_DUMP_EXPORT=""
XLA_FLAGS_EXPORT=""
if [ "${ENABLE_XLA_DUMP:-0}" = "1" ]; then
    XLA_DUMP_DIR="${XLA_DUMP_DIR:-/tmp/xla_dump_train}"
    mkdir -p "$XLA_DUMP_DIR"
    export XLA_DUMP_DIR
    if [ -z "${XLA_FLAGS:-}" ]; then
        export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
    else
        export XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
    fi
    XLA_DUMP_EXPORT="export XLA_DUMP_DIR='$XLA_DUMP_DIR'; "
    XLA_FLAGS_EXPORT="export XLA_FLAGS='$XLA_FLAGS'; "
    echo "  XLA HLO dump: enabled ($XLA_DUMP_DIR)"
else
    unset XLA_DUMP_DIR
    if [ -n "${XLA_FLAGS:-}" ]; then
        XLA_FLAGS_EXPORT="export XLA_FLAGS='$XLA_FLAGS'; "
    fi
    echo "  XLA HLO dump: disabled (set ENABLE_XLA_DUMP=1 to enable)"
fi

# Start new tmux session running training, tee to ~/train.log
tmux new-session -d -s train \
    "${XLA_DUMP_EXPORT}export PYTHONUNBUFFERED=1; export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; ${XLA_FLAGS_EXPORT}{ echo '=== TPU training process startup ==='; echo \"TRAIN_SCRIPT=$TRAIN_SCRIPT\"; echo \"CONFIG=$CONFIG\"; echo \"TRAIN_ARGS=$TRAIN_ARGS\"; echo \"HOSTNAME=\$(hostname)\"; echo \"DATE=\$(date -Is)\"; echo \"PYTHONUNBUFFERED=\$PYTHONUNBUFFERED\"; python3 -u \"$TRAIN_SCRIPT\" --config \"$CONFIG\" $TRAIN_ARGS; } 2>&1 | tee ~/train.log; echo 'Training finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"
