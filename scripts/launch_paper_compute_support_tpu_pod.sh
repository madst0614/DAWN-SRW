#!/usr/bin/env bash
# Launch the frozen v4172 paper compute-support profile on one existing TPU pod.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
readonly REMOTE_USER="madst0614"
BRANCH="codex/v4167-poc"
REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201"
OUTPUT_PARENT=""
BATCH_SIZE="160"
MAX_VAL_TOKENS="10000000"
MAX_BATCHES="0"
INSTALL_DEPS=1
UPDATE_REPO=1
REPLACE=0
DRY_RUN=0

readonly SESSION="train"
readonly REMOTE_LOG="~/train.log"
readonly EXACT_COMMIT="d229a246215a777e27a545ef6066422134a64b2c"
readonly EXPECTED_STEP="76293"
readonly EXPECTED_CONFIG_HASH="08733ae4fefdfcda2bb8e61e51a6e6fce40c0b0e4d84cb80d715085da645039b"
readonly EXPECTED_CHECKPOINT_IDENTITY="a7ce8afcd0242bc4e9b567c9e5066c36ca223461eaa6ae6f251e6525d1f91c17"

usage() {
    printf '%s\n' \
        "Usage: $0 --tpu NAME --replace [options]" \
        "" \
        "  --tpu NAME             Existing authorized TPU v4-64 pod" \
        "  --branch NAME          Exact remote branch (default: $BRANCH)" \
        "  --output-parent PATH   Optional GCS parent for the independent run" \
        "  --batch-size N         Global support-profile batch (default: $BATCH_SIZE)" \
        "  --max-val-tokens N     Packed C4 token cap (default: $MAX_VAL_TOKENS)" \
        "  --max-batches N        0 uses every complete batch" \
        "  --replace              Replace tmux train and its current processes" \
        "  --no-install           Verify, but do not install, pinned dependencies" \
        "  --skip-repo-update     Use the existing clean remote checkout" \
        "  --dry-run              Print the exact remote command"
}

normalize_gcs() {
    if [[ "$1" == dawn-tpu-data-c4/* ]]; then
        printf 'gs://%s' "$1"
    else
        printf '%s' "$1"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --output-parent) OUTPUT_PARENT="$(normalize_gcs "$2")"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-val-tokens) MAX_VAL_TOKENS="$2"; shift 2 ;;
        --max-batches) MAX_BATCHES="$2"; shift 2 ;;
        --replace) REPLACE=1; shift ;;
        --no-install) INSTALL_DEPS=0; shift ;;
        --skip-repo-update) UPDATE_REPO=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu is required" >&2
    exit 1
fi
if [[ "$REPLACE" != "1" ]]; then
    echo "ERROR: --replace is required for this paper run" >&2
    exit 1
fi
for value in "$BATCH_SIZE" "$MAX_VAL_TOKENS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: batch size and token cap must be positive integers" >&2
        exit 1
    fi
done
if ! [[ "$MAX_BATCHES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --max-batches must be zero or a positive integer" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
    ACTUAL_ACCELERATOR="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format='value(acceleratorType)')"
    if [[ "$ACTUAL_ACCELERATOR" != "v4-64" ]]; then
        echo "ERROR: TPU $TPU_NAME acceleratorType=$ACTUAL_ACCELERATOR, expected v4-64" >&2
        exit 1
    fi
fi

printf -v Q_REPO '%q' "$REPO_URL"
printf -v Q_BRANCH '%q' "$BRANCH"
printf -v Q_CHECKPOINT '%q' "$CHECKPOINT"
printf -v Q_OUTPUT_PARENT '%q' "$OUTPUT_PARENT"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
REPO_URL=$Q_REPO
BRANCH=$Q_BRANCH
CHECKPOINT=$Q_CHECKPOINT
OUTPUT_PARENT=$Q_OUTPUT_PARENT
SESSION=train
WORK_DIR="\$HOME/DAWN-SRW"
REMOTE_LOG="\$HOME/train.log"

mkdir -p "\$(dirname "\$REMOTE_LOG")"
tmux kill-session -t "\$SESSION" 2>/dev/null || true
pkill -TERM -f 'scripts/[t]rain_jax.py' 2>/dev/null || true
pkill -TERM -f 'scripts/[p]rofile_paper_compute_support.py' 2>/dev/null || true
: > "\$REMOTE_LOG"
exec > >(tee -a "\$REMOTE_LOG") 2>&1

echo "[setup] host=\$(hostname) branch=\$BRANCH"
if [[ "$UPDATE_REPO" == "1" ]]; then
    if [[ -d "\$WORK_DIR/.git" ]]; then
        cd "\$WORK_DIR"
        if [[ -n "\$(git status --porcelain)" ]]; then
            echo "ERROR: remote checkout has uncommitted changes" >&2
            exit 1
        fi
    elif [[ -e "\$WORK_DIR" ]]; then
        echo "ERROR: \$WORK_DIR exists but is not a git checkout" >&2
        exit 1
    else
        git clone --no-checkout --single-branch --depth 1 --branch "\$BRANCH" \
            "\$REPO_URL" "\$WORK_DIR"
        cd "\$WORK_DIR"
    fi

    DEPLOY_DEPTH=8
    while true; do
        git fetch origin "\$BRANCH" --depth "\$DEPLOY_DEPTH"
        BRANCH_TIP=\$(git rev-parse FETCH_HEAD)
        if git cat-file -e "$EXACT_COMMIT^{commit}" 2>/dev/null &&
                git merge-base --is-ancestor "$EXACT_COMMIT" "\$BRANCH_TIP"; then
            break
        fi
        if (( DEPLOY_DEPTH >= 4096 )); then
            echo "ERROR: pinned commit $EXACT_COMMIT is not a connected ancestor of \$BRANCH at \$BRANCH_TIP" >&2
            exit 1
        fi
        DEPLOY_DEPTH=\$((DEPLOY_DEPTH * 2))
    done
    echo "[setup] pinned_commit=$EXACT_COMMIT branch_tip=\$BRANCH_TIP fetch_depth=\$DEPLOY_DEPTH"
    git checkout -B "\$BRANCH" "$EXACT_COMMIT"
else
    [[ -d "\$WORK_DIR/.git" ]] || {
        echo "ERROR: --skip-repo-update requires \$WORK_DIR/.git" >&2
        exit 1
    }
    cd "\$WORK_DIR"
    if [[ -n "\$(git status --porcelain)" ]]; then
        echo "ERROR: remote checkout has uncommitted changes" >&2
        exit 1
    fi
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
    python3 -m pip install --upgrade pip -q
    python3 -m pip install "jax[tpu]==0.6.2" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
    python3 -m pip install \
        "flax==0.10.7" "optax==0.2.8" "numpy==2.2.6" \
        pyyaml gcsfs google-cloud-storage matplotlib -q
fi
python3 - <<'PYCHK'
from importlib import metadata
import jax
assert jax.__version__ == "0.6.2", jax.__version__
assert metadata.version("flax") == "0.10.7"
assert metadata.version("optax") == "0.2.8"
assert metadata.version("numpy") == "2.2.6"
print(
    "paper-compute-runtime "
    f"jax={jax.__version__} flax={metadata.version('flax')} "
    f"optax={metadata.version('optax')} numpy={metadata.version('numpy')}",
    flush=True,
)
PYCHK

CMD=(
    python3 -u scripts/profile_paper_compute_support.py
    --init-from "\$CHECKPOINT"
    --expected-step "$EXPECTED_STEP"
    --expected-checkpoint-config-hash "$EXPECTED_CONFIG_HASH"
    --expected-checkpoint-identity "$EXPECTED_CHECKPOINT_IDENTITY"
    --batch-size "$BATCH_SIZE"
    --max-val-tokens "$MAX_VAL_TOKENS"
    --max-batches "$MAX_BATCHES"
)
[[ -z "\$OUTPUT_PARENT" ]] || CMD+=(--output-parent "\$OUTPUT_PARENT")
CMD_TEXT=\$(printf '%q ' "\${CMD[@]}")
echo "PAPER_COMPUTE worker=\$(hostname) branch=\$BRANCH commit=\$(git rev-parse HEAD)"
echo "PAPER_COMPUTE command=\$CMD_TEXT"
tmux new-session -d -x 240 -y 60 -s "\$SESSION" \
    "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; { \$CMD_TEXT; } 2>&1 | tee -a '\$REMOTE_LOG'"
echo "PAPER_COMPUTE launched session=\$SESSION log=\$REMOTE_LOG"
EOF

echo "PAPER COMPUTE SUPPORT LAUNCH"
echo "  tpu=$TPU_NAME zone=$ZONE project=$PROJECT branch=$BRANCH"
echo "  exact_commit=$EXACT_COMMIT"
echo "  checkpoint=$CHECKPOINT expected_step=$EXPECTED_STEP"
echo "  batch_size=$BATCH_SIZE max_val_tokens=$MAX_VAL_TOKENS"
echo "  session=$SESSION log=$REMOTE_LOG"
echo "  watch=bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --summary"

if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$REMOTE_CMD"
    exit 0
fi

gcloud compute tpus tpu-vm ssh "$REMOTE_USER@$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker=all \
    --command="$REMOTE_CMD"

echo "PAPER COMPUTE launch complete"
