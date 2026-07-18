#!/bin/bash
set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
INIT_FROM=""
OUTPUT_DIR=""
GH_TOKEN=""
BATCH_SIZE="32"
LIMIT=""
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --token) GH_TOKEN="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 --tpu NAME --init-from RUN_OR_CONCRETE_STEP [options]

Required:
  --tpu NAME
  --init-from PATH             Orbax run/checkpoints directory or step

Reproducibility:
  --output-dir PATH            Local or gs:// result directory
  --limit N                    Smoke only; omitted means full comparable run
  --batch-size N               Fixed global batch (default: 32)

Cloud:
  --zone ZONE                  Default: $ZONE
  --project PROJECT            Default: $PROJECT
  --branch BRANCH              Default: $BRANCH
  --token TOKEN                Optional GitHub token
  --foreground                 Do not detach into tmux
EOF
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TPU_NAME" ]]; then echo "ERROR: --tpu is required" >&2; exit 2; fi
if [[ -z "$INIT_FROM" ]]; then echo "ERROR: --init-from is required" >&2; exit 2; fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --batch-size must be a positive integer" >&2; exit 2
fi
if [[ -n "$LIMIT" ]] && ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --limit must be a positive integer" >&2; exit 2
fi

require_remote_visible_file() {
  local path="$1"
  if ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "ERROR: $path is not tracked; TPU workers cannot check it out." >&2
    exit 2
  fi
  if ! git diff --quiet -- "$path" || ! git diff --cached --quiet -- "$path"; then
    echo "ERROR: $path has uncommitted changes. Commit and push first." >&2
    exit 2
  fi
}

for file in \
  scripts/zero_shot_eval_jax.py \
  scripts/launch_zero_shot_eval_tpu_pod.sh \
  dawn/__init__.py \
  dawn/eval/__init__.py \
  dawn/eval/lm_eval_dawn_adapter.py \
  dawn/eval/jax_runtime.py \
  dawn/eval/zero_shot_protocol.py \
  models/baseline_transformer_jax.py \
  models/dawn_srw_v4166.py \
  models/dawn_srw_v4171.py \
  models/vocab_parallel.py \
  requirements_zero_shot_eval.txt; do
  require_remote_visible_file "$file"
done

if [[ -n "$GH_TOKEN" ]]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
  REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="gs://dawn-tpu-data-c4/zero_shot_results/${TPU_NAME}_${STAMP}"
fi

echo "============================================"
echo "Launching DAWN stock zero-shot evaluation"
echo "  TPU:                $TPU_NAME"
echo "  Zone/project:       $ZONE / $PROJECT"
echo "  Branch:             $BRANCH"
echo "  Init from:          $INIT_FROM"
echo "  Output:             $OUTPUT_DIR"
echo "  Tokenizer:          source pretokenization metadata"
echo "  Global batch:       $BATCH_SIZE"
echo "  Limit:              ${LIMIT:-<full>}"
echo "============================================"

gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
INIT_FROM='${INIT_FROM}'
OUTPUT_DIR='${OUTPUT_DIR}'
BATCH_SIZE='${BATCH_SIZE}'
LIMIT='${LIMIT}'
FOREGROUND='${FOREGROUND}'
WORK_DIR="\$HOME/dawn-spatial"
REMOTE_LOG="\$HOME/train.log"

tmux kill-session -t train 2>/dev/null || true
: > "\$REMOTE_LOG"
exec > >(tee -a "\$REMOTE_LOG") 2>&1

if [ -d "\$WORK_DIR/.git" ]; then
  cd "\$WORK_DIR"
  git fetch origin "\$BRANCH" --depth 1
  git checkout -B "\$BRANCH" FETCH_HEAD
  git reset --hard FETCH_HEAD
  git clean -fd
else
  rm -rf "\$WORK_DIR"
  git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
  cd "\$WORK_DIR"
fi

python3 -m pip install --upgrade pip -q
python3 -m pip install \
  "jax[tpu]==0.6.2" \
  "flax==0.10.7" \
  "optax==0.2.8" \
  "numpy==2.2.6" \
  pyyaml sentencepiece google-cloud-storage \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
python3 -m pip install -r requirements_zero_shot_eval.txt -q
python3 - <<'PYCHK'
from importlib import metadata
import datasets, flax, fsspec, jax, optax, orbax.checkpoint, pyarrow, transformers
assert jax.__version__ == '0.6.2'
assert metadata.version('flax') == '0.10.7'
assert metadata.version('optax') == '0.2.8'
assert metadata.version('numpy') == '2.2.6'
assert metadata.version('lm_eval') == '0.4.2'
assert metadata.version('orbax-checkpoint') == '0.11.24'
assert metadata.version('transformers') == '4.40.2'
assert metadata.version('tokenizers') == '0.19.1'
assert metadata.version('huggingface-hub') == '0.36.2'
assert metadata.version('datasets') == '2.19.2'
assert metadata.version('pyarrow') == '20.0.0'
assert metadata.version('fsspec') == '2024.3.1'
print('lm-eval=' + metadata.version('lm_eval'), flush=True)
print('orbax-checkpoint=' + metadata.version('orbax-checkpoint'), flush=True)
print('jax=' + jax.__version__, flush=True)
print('flax=' + flax.__version__, flush=True)
print('optax=' + optax.__version__, flush=True)
print('numpy=' + metadata.version('numpy'), flush=True)
print('datasets=' + datasets.__version__, flush=True)
print('pyarrow=' + pyarrow.__version__, flush=True)
print('transformers=' + transformers.__version__, flush=True)
PYCHK

export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"
export TOKENIZERS_PARALLELISM=false

RUN_CMD=(python3 scripts/zero_shot_eval_jax.py
  --init-from "\$INIT_FROM"
  --output-dir "\$OUTPUT_DIR"
  --batch-size "\$BATCH_SIZE")
if [ -n "\$LIMIT" ]; then RUN_CMD+=(--limit "\$LIMIT"); fi
RUN_CMD_STR=\$(printf '%q ' "\${RUN_CMD[@]}")

if [ "\$FOREGROUND" = "1" ]; then
  "\${RUN_CMD[@]}"
else
  tmux new-session -d -s train \
    "cd '\$WORK_DIR'; \$RUN_CMD_STR 2>&1 | tee -a '\$REMOTE_LOG'"
  echo "train tmux session started on \$(hostname)"
fi
EOFCMD

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="$REMOTE_CMD" \
  2>&1 | tee "launch_zero_shot_${TPU_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "Launched."
echo "  Follow: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Output: $OUTPUT_DIR"
