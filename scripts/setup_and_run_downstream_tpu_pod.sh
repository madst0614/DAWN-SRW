#!/bin/bash
set -euo pipefail

BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIGS="${CONFIGS:?ERROR: CONFIGS env var not set}"
INIT_FROM="${INIT_FROM:-}"
DOWNSTREAM_RUN_ID="${DOWNSTREAM_RUN_ID:?ERROR: DOWNSTREAM_RUN_ID env var not set}"
WORK_DIR="$HOME/dawn-spatial"

IFS='|' read -r -a CONFIG_ARRAY <<< "$CONFIGS"

resolve_config_path() {
  local p="$1"
  if [ -f "$p" ]; then
    printf '%s\n' "$p"
  elif [ -f "${p}.yaml" ]; then
    printf '%s\n' "${p}.yaml"
  elif [ -f "${p}.yml" ]; then
    printf '%s\n' "${p}.yml"
  else
    printf '%s\n' "$p"
  fi
}

echo "============================================"
echo "Host $(hostname) — Setting up downstream TPU training"
echo "  Branch:    $BRANCH"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Run ID:    $DOWNSTREAM_RUN_ID"
echo "  Configs:   ${CONFIG_ARRAY[*]}"
echo "============================================"

cd "$WORK_DIR"

echo "[1/4] Installing dependencies..."
python3 -m pip install --upgrade pip -q
python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
python3 -m pip install -U flax optax numpy pyyaml transformers sentencepiece huggingface_hub google-cloud-storage "orbax-checkpoint==0.11.24" -q
# Pin datasets/pyarrow to a compatible pair. New pyarrow removed PyExtensionType,
# which older datasets imports at startup.  gcsfs and fsspec must be pinned as
# a compatible pair, otherwise downstream GCS checkpoint I/O can fail later.
python3 -m pip install --force-reinstall --no-cache-dir "pyarrow==20.0.0" "datasets==2.19.2" "fsspec==2024.3.1" "gcsfs==2024.3.1" -q
python3 - <<'PYCHK'
import sys
import datasets
import transformers
import pyarrow as pa
print(f"  Python: {sys.executable}", flush=True)
print(f"  datasets: {datasets.__version__}", flush=True)
print(f"  pyarrow: {pa.__version__}", flush=True)
print(f"  transformers: {transformers.__version__}", flush=True)
PYCHK

echo "[2/4] Verifying downstream files..."
test -f scripts/downstream_finetune_jax.py || { echo "missing scripts/downstream_finetune_jax.py" >&2; exit 2; }
test -f scripts/run_downstream_sequence.sh || { echo "missing scripts/run_downstream_sequence.sh" >&2; exit 2; }
test -f scripts/expand_downstream_suite.py || { echo "missing scripts/expand_downstream_suite.py" >&2; exit 2; }
test -f scripts/downstream_protocol.py || { echo "missing scripts/downstream_protocol.py" >&2; exit 2; }
RESOLVED_CONFIG_ARRAY=()
for c in "${CONFIG_ARRAY[@]}"; do
  rc="$(resolve_config_path "$c")"
  test -f "$rc" || { echo "missing config: $c" >&2; exit 2; }
  RESOLVED_CONFIG_ARRAY+=("$rc")
done
CONFIG_ARRAY=("${RESOLVED_CONFIG_ARRAY[@]}")

echo "[3/4] Skipping standalone JAX TPU preflight."

echo "[4/4] Starting downstream sequence in tmux session 'train'..."
tmux kill-session -t train 2>/dev/null || true
pkill -f downstream_finetune_jax.py 2>/dev/null || true

XLA_DUMP_DIR="${XLA_DUMP_DIR:-/tmp/xla_dump_downstream}"
mkdir -p "$XLA_DUMP_DIR"
export XLA_DUMP_DIR
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export DOWNSTREAM_JAX_DISTRIBUTED="${DOWNSTREAM_JAX_DISTRIBUTED:-1}"
if [ -z "${XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
else
  export XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
fi

CMD="bash scripts/run_downstream_sequence.sh --run-id '$DOWNSTREAM_RUN_ID'"
if [ -n "$INIT_FROM" ]; then
  CMD="$CMD --init-from '$INIT_FROM'"
fi
for c in "${CONFIG_ARRAY[@]}"; do
  CMD="$CMD --config '$c'"
done

echo "  Command: $CMD"
echo "  Log: ~/train.log"

tmux new-session -d -s train \
  "export XLA_DUMP_DIR='$XLA_DUMP_DIR'; \
   export JAX_TRACEBACK_FILTERING='$JAX_TRACEBACK_FILTERING'; \
   export JAX_LOG_COMPILES='$JAX_LOG_COMPILES'; \
   export TF_CPP_MIN_LOG_LEVEL='$TF_CPP_MIN_LOG_LEVEL'; \
   export DOWNSTREAM_JAX_DISTRIBUTED='$DOWNSTREAM_JAX_DISTRIBUTED'; \
   export DOWNSTREAM_RUN_ID='$DOWNSTREAM_RUN_ID'; \
   export XLA_FLAGS='$XLA_FLAGS'; \
   cd '$WORK_DIR'; \
   $CMD 2>&1 | tee ~/train.log; \
   echo 'Downstream sequence finished. Press enter to close.'; read"

echo "  tmux session 'train' started."
echo "  Attach:  tmux attach -t train"
echo "  Monitor: tail -f ~/train.log"
