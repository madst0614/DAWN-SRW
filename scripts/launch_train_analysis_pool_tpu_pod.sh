#!/usr/bin/env bash
# Launch the canonical train_analysis_pool on every worker of an existing TPU pod.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
CHECKPOINT=""
TARGET=""
OUTPUT=""
BENCHMARK_ROOT="gs://dawn-tpu-data-c4/dataset/operator_interpretability"
PRESET="scientific"
ITEMS=""
RUNTIME="v4-64"
MAX_EXAMPLES="64"
MESH_DATA=""
MESH_MODEL=""
readonly SESSION="train"
readonly REMOTE_LOG="~/train.log"
INSTALL_DEPS=1
UPDATE_REPO=1
DETACH=1
RESUME=1
REPLACE=0
DRY_RUN=0

usage() {
    printf '%s\n' \
        "Usage: $0 --tpu NAME (--target ID | --checkpoint PATH) [options]" \
        "" \
        "  --tpu NAME                 Existing TPU VM/Pod" \
        "  --target ID                Registered target, for example v4171_400m" \
        "  --checkpoint PATH          Ad-hoc Orbax path; mutually exclusive with --target" \
        "  --runtime ID               Physical runtime profile (default: $RUNTIME)" \
        "  --output PATH              Optional artifact root; checkpoint side_analysis is default" \
        "  --benchmark-root PATH      Immutable prepared benchmark root" \
        "  --preset NAME              Item bundle: zero_shot, mechanistic_screen, circuit, causal, scientific, all" \
        "  --items IDS                Concrete comma-separated item ids; overrides preset" \
        "  --max-examples-per-phase N Fixed per-phase cap (default: $MAX_EXAMPLES)" \
        "  --mesh-data N              Ad-hoc assertion; target/runtime value cannot be overridden" \
        "  --mesh-model N             Ad-hoc assertion; registered target owns this value" \
        "  --branch NAME              Git branch (default: $BRANCH)" \
        "  --zone ZONE                Default: $ZONE" \
        "  --project PROJECT          Default: $PROJECT" \
        "  --no-resume                Recompute instead of protocol-bound resume" \
        "  --no-install               Skip remote dependency installation" \
        "  --skip-repo-update         Use the existing remote checkout" \
        "  --foreground               Do not use tmux" \
        "  --replace                  Replace an existing session with the same name" \
        "  --dry-run                  Print the exact launch without executing it"
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
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$(normalize_gcs "$2")"; shift 2 ;;
        --runtime) RUNTIME="$2"; shift 2 ;;
        --output) OUTPUT="$(normalize_gcs "$2")"; shift 2 ;;
        --benchmark-root) BENCHMARK_ROOT="$(normalize_gcs "$2")"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --items) ITEMS="$2"; shift 2 ;;
        --max-examples-per-phase) MAX_EXAMPLES="$2"; shift 2 ;;
        --mesh-data) MESH_DATA="$2"; shift 2 ;;
        --mesh-model) MESH_MODEL="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --no-install) INSTALL_DEPS=0; shift ;;
        --skip-repo-update) UPDATE_REPO=0; shift ;;
        --foreground) DETACH=0; shift ;;
        --replace) REPLACE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu is required" >&2
    usage >&2
    exit 1
fi
if [[ -n "$TARGET" && -n "$CHECKPOINT" ]] || [[ -z "$TARGET" && -z "$CHECKPOINT" ]]; then
    echo "ERROR: exactly one of --target or --checkpoint is required" >&2
    usage >&2
    exit 1
fi
if ! [[ "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-examples-per-phase must be a positive integer" >&2
    exit 1
fi
case "$PRESET" in
    contract|zero_shot|mechanistic_screen|mib_ioi_circuit|mib_mcqa_circuit|mib_arithmetic_circuit|mib_arc_circuit|circuit|ravel_causal|causal|scientific|all) ;;
    *) echo "ERROR: unsupported canonical preset $PRESET" >&2; exit 1 ;;
esac
case "$RUNTIME" in
    v4-32|v4-64|v4-128) ;;
    *) echo "ERROR: unsupported runtime profile $RUNTIME" >&2; exit 1 ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
    ACTUAL_ACCELERATOR="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format='value(acceleratorType)')"
    if [[ "$ACTUAL_ACCELERATOR" != "$RUNTIME" ]]; then
        echo "ERROR: runtime=$RUNTIME but TPU $TPU_NAME acceleratorType=$ACTUAL_ACCELERATOR" >&2
        exit 1
    fi
fi

printf -v Q_REPO '%q' "$REPO_URL"
printf -v Q_BRANCH '%q' "$BRANCH"
printf -v Q_TARGET '%q' "$TARGET"
printf -v Q_CHECKPOINT '%q' "$CHECKPOINT"
printf -v Q_OUTPUT '%q' "$OUTPUT"
printf -v Q_BENCHMARK_ROOT '%q' "$BENCHMARK_ROOT"
printf -v Q_PRESET '%q' "$PRESET"
printf -v Q_ITEMS '%q' "$ITEMS"
printf -v Q_RUNTIME '%q' "$RUNTIME"
printf -v Q_LOG '%q' "$REMOTE_LOG"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
REPO_URL=$Q_REPO
BRANCH=$Q_BRANCH
TARGET=$Q_TARGET
CHECKPOINT=$Q_CHECKPOINT
OUTPUT=$Q_OUTPUT
BENCHMARK_ROOT=$Q_BENCHMARK_ROOT
PRESET=$Q_PRESET
ITEMS=$Q_ITEMS
RUNTIME=$Q_RUNTIME
SESSION=train
REMOTE_LOG=$Q_LOG
WORK_DIR="\$HOME/DAWN-SRW"
REMOTE_LOG="\${REMOTE_LOG/#\~/\$HOME}"

mkdir -p "\$(dirname "\$REMOTE_LOG")"
if tmux has-session -t "\$SESSION" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
        tmux kill-session -t "\$SESSION"
    else
        echo "ERROR: tmux session \$SESSION already exists; use --replace" >&2
        exit 1
    fi
fi
: > "\$REMOTE_LOG"
exec > >(tee -a "\$REMOTE_LOG") 2>&1

if [[ "$UPDATE_REPO" == "1" ]]; then
    if [[ -d "\$WORK_DIR/.git" ]]; then
        cd "\$WORK_DIR"
        git fetch origin "\$BRANCH" --depth 1
        git checkout -B "\$BRANCH" FETCH_HEAD
    elif [[ -e "\$WORK_DIR" ]]; then
        echo "ERROR: \$WORK_DIR exists but is not a git checkout" >&2
        exit 1
    else
        git clone --single-branch --depth 1 --branch "\$BRANCH" \
            "\$REPO_URL" "\$WORK_DIR"
        cd "\$WORK_DIR"
    fi
else
    [[ -d "\$WORK_DIR/.git" ]] || {
        echo "ERROR: --skip-repo-update requires \$WORK_DIR/.git" >&2
        exit 1
    }
    cd "\$WORK_DIR"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
    python3 -m pip install --upgrade pip -q
    python3 -m pip install "jax[tpu]==0.6.2" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
    python3 -m pip install \
        "flax==0.10.7" "optax==0.2.8" "numpy==2.2.6" \
        pyyaml google-cloud-storage requests sentencepiece -q
    python3 -m pip install -r requirements_zero_shot_eval.txt -q
    python3 - <<'PYCHK'
from importlib import metadata
import jax
assert jax.__version__ == '0.6.2'
assert metadata.version('flax') == '0.10.7'
assert metadata.version('optax') == '0.2.8'
assert metadata.version('numpy') == '2.2.6'
print(
    'analysis-runtime '
    f'jax={jax.__version__} flax={metadata.version("flax")} '
    f'optax={metadata.version("optax")} numpy={metadata.version("numpy")}',
    flush=True,
)
PYCHK
fi

CMD=(
    python3 -u scripts/analyze_train_analysis_pool.py
    --benchmark-root "\$BENCHMARK_ROOT"
    --runtime "\$RUNTIME"
    --preset "\$PRESET"
    --max-examples-per-phase "$MAX_EXAMPLES"
    --init-distributed
)
if [[ -n "\$TARGET" ]]; then
    CMD+=(--target "\$TARGET")
else
    CMD+=(--checkpoint "\$CHECKPOINT")
fi
[[ -z "\$OUTPUT" ]] || CMD+=(--output "\$OUTPUT")
[[ -z "\$ITEMS" ]] || CMD+=(--items "\$ITEMS")
[[ -z "$MESH_DATA" ]] || CMD+=(--mesh-data "$MESH_DATA")
[[ -z "$MESH_MODEL" ]] || CMD+=(--mesh-model "$MESH_MODEL")
[[ "$RESUME" == "1" ]] || CMD+=(--no-resume)

CMD_TEXT=\$(printf '%q ' "\${CMD[@]}")
echo "TRAIN_ANALYSIS_POOL worker=\$(hostname) branch=\$BRANCH"
echo "TRAIN_ANALYSIS_POOL command=\$CMD_TEXT"
if [[ "$DETACH" == "1" ]]; then
    tmux new-session -d -x 240 -y 60 -s "\$SESSION" \
        "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; { \$CMD_TEXT; } 2>&1 | tee -a '\$REMOTE_LOG'"
    echo "TRAIN_ANALYSIS_POOL launched session=\$SESSION log=\$REMOTE_LOG"
else
    "\${CMD[@]}"
fi
EOF

WATCH_CMD="bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --summary"
echo "TRAIN_ANALYSIS_POOL LAUNCH"
echo "  tpu=$TPU_NAME zone=$ZONE project=$PROJECT branch=$BRANCH"
echo "  target=${TARGET:-ad-hoc} checkpoint=${CHECKPOINT:-target-registry} runtime=$RUNTIME"
echo "  output=${OUTPUT:-checkpoint-side-analysis-default}"
echo "  preset=$PRESET items=${ITEMS:-preset}"
echo "  session=$SESSION log=$REMOTE_LOG"
if [[ -n "$TARGET" ]]; then
    SOURCE_COPY="--target $TARGET"
else
    SOURCE_COPY="--checkpoint $CHECKPOINT"
fi
echo "Copy-paste: $0 --tpu $TPU_NAME --zone $ZONE --project $PROJECT --branch $BRANCH $SOURCE_COPY --runtime $RUNTIME --benchmark-root $BENCHMARK_ROOT --preset $PRESET"
echo "Watch logs: $WATCH_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN remote command:"
    printf '%s\n' "$REMOTE_CMD"
    exit 0
fi

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker=all \
    --command="$REMOTE_CMD"

echo "TRAIN_ANALYSIS_POOL launch complete"
echo "Watch logs: $WATCH_CMD"
