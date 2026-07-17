#!/usr/bin/env bash
# Launch the canonical train_analysis_pool on every worker of an existing TPU pod.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
CHECKPOINT=""
OUTPUT=""
BENCHMARK_ROOT="gs://dawn-tpu-data-c4/dataset/operator_interpretability"
BENCHMARKS="primary"
PRESET="scientific"
ITEMS=""
MAX_EXAMPLES="64"
MESH_DATA=""
MESH_MODEL=""
SESSION="train_analysis_pool"
REMOTE_LOG="~/train_analysis_pool.log"
INSTALL_DEPS=1
UPDATE_REPO=1
DETACH=1
RESUME=1
REPLACE=0
DRY_RUN=0

usage() {
    printf '%s\n' \
        "Usage: $0 --tpu NAME --checkpoint PATH [options]" \
        "" \
        "  --tpu NAME                 Existing TPU VM/Pod" \
        "  --checkpoint PATH          Orbax step, checkpoints directory, run, or latest" \
        "  --output PATH              Optional artifact root; checkpoint side_analysis is default" \
        "  --benchmark-root PATH      Immutable prepared benchmark root" \
        "  --benchmarks IDS           primary, all, or canonical comma-separated ids" \
        "  --preset NAME              contract, circuit, causal, scientific" \
        "  --items IDS                Canonical comma-separated items; overrides preset" \
        "  --max-examples-per-phase N Fixed per-phase cap (default: $MAX_EXAMPLES)" \
        "  --mesh-data N              Override checkpoint mesh data axis" \
        "  --mesh-model N             Override checkpoint mesh model axis" \
        "  --branch NAME              Git branch (default: $BRANCH)" \
        "  --zone ZONE                Default: $ZONE" \
        "  --project PROJECT          Default: $PROJECT" \
        "  --log PATH                 Default: $REMOTE_LOG" \
        "  --session NAME             Default: $SESSION" \
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
        --checkpoint) CHECKPOINT="$(normalize_gcs "$2")"; shift 2 ;;
        --output) OUTPUT="$(normalize_gcs "$2")"; shift 2 ;;
        --benchmark-root) BENCHMARK_ROOT="$(normalize_gcs "$2")"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --items) ITEMS="$2"; shift 2 ;;
        --max-examples-per-phase) MAX_EXAMPLES="$2"; shift 2 ;;
        --mesh-data) MESH_DATA="$2"; shift 2 ;;
        --mesh-model) MESH_MODEL="$2"; shift 2 ;;
        --log) REMOTE_LOG="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
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

if [[ -z "$TPU_NAME" || -z "$CHECKPOINT" ]]; then
    echo "ERROR: --tpu and --checkpoint are required" >&2
    usage >&2
    exit 1
fi
if ! [[ "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-examples-per-phase must be a positive integer" >&2
    exit 1
fi
case "$PRESET" in
    contract|circuit|causal|scientific) ;;
    *) echo "ERROR: unsupported canonical preset $PRESET" >&2; exit 1 ;;
esac

printf -v Q_REPO '%q' "$REPO_URL"
printf -v Q_BRANCH '%q' "$BRANCH"
printf -v Q_CHECKPOINT '%q' "$CHECKPOINT"
printf -v Q_OUTPUT '%q' "$OUTPUT"
printf -v Q_BENCHMARK_ROOT '%q' "$BENCHMARK_ROOT"
printf -v Q_BENCHMARKS '%q' "$BENCHMARKS"
printf -v Q_PRESET '%q' "$PRESET"
printf -v Q_ITEMS '%q' "$ITEMS"
printf -v Q_SESSION '%q' "$SESSION"
printf -v Q_LOG '%q' "$REMOTE_LOG"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
REPO_URL=$Q_REPO
BRANCH=$Q_BRANCH
CHECKPOINT=$Q_CHECKPOINT
OUTPUT=$Q_OUTPUT
BENCHMARK_ROOT=$Q_BENCHMARK_ROOT
BENCHMARKS=$Q_BENCHMARKS
PRESET=$Q_PRESET
ITEMS=$Q_ITEMS
SESSION=$Q_SESSION
REMOTE_LOG=$Q_LOG
WORK_DIR="\$HOME/DAWN-SRW"
REMOTE_LOG="\${REMOTE_LOG/#\~/\$HOME}"

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
        flax optax orbax-checkpoint==0.11.24 numpy pyyaml gcsfs \
        transformers==4.57.6 datasets==4.8.5 huggingface-hub requests -q
fi

CMD=(
    python3 -u scripts/analyze_train_analysis_pool.py
    --checkpoint "\$CHECKPOINT"
    --benchmark-root "\$BENCHMARK_ROOT"
    --benchmarks "\$BENCHMARKS"
    --preset "\$PRESET"
    --max-examples-per-phase "$MAX_EXAMPLES"
    --init-distributed
)
[[ -z "\$OUTPUT" ]] || CMD+=(--output "\$OUTPUT")
[[ -z "\$ITEMS" ]] || CMD+=(--items "\$ITEMS")
[[ -z "$MESH_DATA" ]] || CMD+=(--mesh-data "$MESH_DATA")
[[ -z "$MESH_MODEL" ]] || CMD+=(--mesh-model "$MESH_MODEL")
[[ "$RESUME" == "1" ]] || CMD+=(--no-resume)

mkdir -p "\$(dirname "\$REMOTE_LOG")"
if tmux has-session -t "\$SESSION" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
        tmux kill-session -t "\$SESSION"
    else
        echo "ERROR: tmux session \$SESSION already exists; use --replace" >&2
        exit 1
    fi
fi
CMD_TEXT=\$(printf '%q ' "\${CMD[@]}")
echo "TRAIN_ANALYSIS_POOL worker=\$(hostname) branch=\$BRANCH"
echo "TRAIN_ANALYSIS_POOL command=\$CMD_TEXT"
if [[ "$DETACH" == "1" ]]; then
    tmux new-session -d -x 240 -y 60 -s "\$SESSION" \
        "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; { \$CMD_TEXT; } 2>&1 | tee '\$REMOTE_LOG'"
    echo "TRAIN_ANALYSIS_POOL launched session=\$SESSION log=\$REMOTE_LOG"
else
    "\${CMD[@]}" 2>&1 | tee "\$REMOTE_LOG"
fi
EOF

WATCH_CMD="bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --log $REMOTE_LOG --target $SESSION --summary"
echo "TRAIN_ANALYSIS_POOL LAUNCH"
echo "  tpu=$TPU_NAME zone=$ZONE project=$PROJECT branch=$BRANCH"
echo "  checkpoint=$CHECKPOINT"
echo "  output=${OUTPUT:-checkpoint-side-analysis-default}"
echo "  benchmarks=$BENCHMARKS preset=$PRESET items=${ITEMS:-preset}"
echo "  session=$SESSION log=$REMOTE_LOG"
echo "Copy-paste: $0 --tpu $TPU_NAME --zone $ZONE --project $PROJECT --branch $BRANCH --checkpoint $CHECKPOINT --benchmark-root $BENCHMARK_ROOT --preset $PRESET"
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
