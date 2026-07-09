#!/bin/bash
# =============================================================================
# TPU VM/Pod launcher for DAWN-SRW v4166 analysis
# =============================================================================
# This script does not create a TPU.  It launches the analysis process on an
# already-created TPU VM/Pod, on every worker, inside tmux session "train" so
# existing tmux pipe-pane/capture-pane log workflows continue to work.
#
# Example:
#   bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh \
#     --tpu dawn-400m-v4-64 \
#     --zone us-central2-b \
#     --project dawn-486218 \
#     --branch main \
#     --output gs://dawn-tpu-data-c4/analysis/v4166_400M_final \
#     --stages eval,prune,geometry,usage,trace,ablation,report
#
# Train-side checkpoint analysis preset:
#   bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh \
#     --preset v4166-1B \
#     --tpu spatial-400m \
#     --project dawn-486218 \
#     --zone us-central2-b \
#     --branch codex/v4167-poc \
#     --mode train
#
# If the TPU does not exist yet, create it separately, for example:
#   gcloud compute tpus tpu-vm create dawn-400m-v4-64 \
#     --zone=us-central2-b \
#     --accelerator-type=v4-64 \
#     --version=tpu-vm-v4-base \
#     --project=dawn-486218
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
GH_TOKEN=""
REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
PRESET=""

CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_400M_c4_40B_v4_64/run_vspatial-r1-v4.1.6.6_20260622_212706_3201/checkpoints/000000076293"
OUTPUT="gs://dawn-tpu-data-c4/analysis/v4166_400M_final"
STAGES="eval,prune,geometry,usage,trace,ablation,report"
ANALYSIS_ARGS=""
MODE="analysis"
WORKERS="auto"
DETACH="1"
DETACH_EXPLICIT="0"
INSTALL_DEPS="1"
TMUX_SESSION="train"
REMOTE_LOG="~/train.log"
DRY_RUN="0"
OUTPUT_EXPLICIT="0"
MODE_EXPLICIT="0"
CHECKPOINT_DIR_EXPLICIT="0"
PRUNE_EPS_EXPLICIT="0"
REMOTE_LOG_EXPLICIT="0"

TRAIN_ANALYSIS_CONFIG="${DAWN_TRAIN_ANALYSIS_CONFIG:-}"
TRAIN_ANALYSIS_CHECKPOINT_DIR="${DAWN_TRAIN_ANALYSIS_CHECKPOINT_DIR:-gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_1p3B_c4_20B_v4_64_new}"
TRAIN_ANALYSIS_MAX_BATCHES="${DAWN_TRAIN_ANALYSIS_MAX_BATCHES:-8}"
TRAIN_ANALYSIS_PRUNE_EPS="${DAWN_TRAIN_ANALYSIS_PRUNE_EPS:-1e-6,1e-5,1e-4,1e-3}"
if [[ -n "${DAWN_TRAIN_ANALYSIS_PRUNE_EPS+x}" ]]; then
    PRUNE_EPS_EXPLICIT="1"
fi

normalize_gcs_arg() {
    local value="$1"
    if [[ "$value" == dawn-tpu-data-c4/* ]]; then
        printf 'gs://%s' "$value"
    else
        printf '%s' "$value"
    fi
}

path_name() {
    local value="${1%/}"
    value="${value//\\//}"
    printf '%s' "${value##*/}"
}

path_parent() {
    local value="${1%/}"
    value="${value//\\//}"
    if [[ "$value" == */* ]]; then
        printf '%s' "${value%/*}"
    else
        printf '.'
    fi
}

apply_preset() {
    local preset_lc="${1,,}"
    case "$preset_lc" in
        v4166-1b|v4166-1p3b|v4166-1p3b-c4-20b|v4166-1p3b-c4-20b-v4-64)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_1p3B_c4_20B_v4_64_new"
            fi
            if [[ "$PRUNE_EPS_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRUNE_EPS="1e-2,1e-1"
            fi
            ;;
        *)
            echo "ERROR: unknown --preset $1" >&2
            echo "Known presets: v4166-1B" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --mode) MODE="$2"; MODE_EXPLICIT="1"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --config) TRAIN_ANALYSIS_CONFIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$(normalize_gcs_arg "$2")"; shift 2 ;;
        --checkpoint-dir) TRAIN_ANALYSIS_CHECKPOINT_DIR="$(normalize_gcs_arg "$2")"; CHECKPOINT_DIR_EXPLICIT="1"; shift 2 ;;
        --output) OUTPUT="$(normalize_gcs_arg "$2")"; OUTPUT_EXPLICIT="1"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --foreground) DETACH="0"; DETACH_EXPLICIT="1"; shift ;;
        --detach) DETACH="1"; DETACH_EXPLICIT="1"; shift ;;
        --log) REMOTE_LOG="$2"; REMOTE_LOG_EXPLICIT="1"; shift 2 ;;
        --no-install) INSTALL_DEPS="0"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --from-scratch) ANALYSIS_ARGS="$ANALYSIS_ARGS --from-scratch"; shift ;;
        --retry-failed) ANALYSIS_ARGS="$ANALYSIS_ARGS --retry-failed"; shift ;;
        --fail-fast) ANALYSIS_ARGS="$ANALYSIS_ARGS --fail-fast"; shift ;;
        --mesh-data) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-data $2"; shift 2 ;;
        --mesh-model) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-model $2"; shift 2 ;;
        --eval-max-tokens) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-max-tokens $2"; shift 2 ;;
        --eval-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-batch-size $2"; shift 2 ;;
        --prune-eps) TRAIN_ANALYSIS_PRUNE_EPS="$2"; PRUNE_EPS_EXPLICIT="1"; shift 2 ;;
        --usage-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-max-sequences $2"; shift 2 ;;
        --usage-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-batch-size $2"; shift 2 ;;
        --usage-seq-len) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-seq-len $2"; shift 2 ;;
        --trace-max-prompts) ANALYSIS_ARGS="$ANALYSIS_ARGS --trace-max-prompts $2"; shift 2 ;;
        --ablation-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-max-sequences $2"; shift 2 ;;
        --ablation-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-batch-size $2"; shift 2 ;;
        --ablation-k-list) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-k-list $2"; shift 2 ;;
        --ablation-strategies) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-strategies $2"; shift 2 ;;
        --max-jobs-per-stage) TRAIN_ANALYSIS_MAX_BATCHES="$2"; ANALYSIS_ARGS="$ANALYSIS_ARGS --max-jobs-per-stage $2"; shift 2 ;;
        --extra-arg) ANALYSIS_ARGS="$ANALYSIS_ARGS $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [options]"
            echo ""
            echo "Required:"
            echo "  --tpu NAME"
            echo ""
            echo "Core:"
            echo "  --preset NAME             Known: v4166-1B"
            echo "  --mode MODE               analysis, train, or train_analysis. Default: $MODE"
            echo "  --config PATH             Optional train_analysis fallback config. Default: checkpoint full_config"
            echo "  --checkpoint PATH_OR_GS   Default: $CHECKPOINT"
            echo "  --checkpoint-dir DIR      train_analysis base checkpoint dir. Default: $TRAIN_ANALYSIS_CHECKPOINT_DIR"
            echo "  --output PATH_OR_GS       Default: $OUTPUT"
            echo "  --stages CSV              Default: $STAGES"
            echo "  --branch BRANCH           Default: $BRANCH"
            echo "  --repo-url URL            Default: $REPO_URL"
            echo "  --log PATH                Remote log path. Default: ~/train.log"
            echo ""
            echo "TPU/GCP:"
            echo "  --zone ZONE               Default: $ZONE"
            echo "  --project PROJECT         Default: $PROJECT"
            echo "  --workers all|0|N         Default: all"
            echo ""
            echo "Execution:"
            echo "  --detach                  Run in tmux session train (default)"
            echo "  --foreground              Run foreground on the SSH command"
            echo "  --dry-run                 Print resolved command without launching"
            echo "  --from-scratch            Disable analysis artifact resume"
            echo "  --retry-failed"
            echo "  --fail-fast"
            echo "  --mesh-data N"
            echo "  --mesh-model N"
            echo "  --max-jobs-per-stage N"
            echo "  --extra-arg '...'"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$PRESET" ]]; then
    apply_preset "$PRESET"
fi

CHECKPOINT="$(normalize_gcs_arg "$CHECKPOINT")"
TRAIN_ANALYSIS_CHECKPOINT_DIR="$(normalize_gcs_arg "$TRAIN_ANALYSIS_CHECKPOINT_DIR")"
OUTPUT="$(normalize_gcs_arg "$OUTPUT")"

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu required" >&2
    exit 1
fi

case "$MODE" in
    analysis|full|full_analysis)
        MODE="analysis"
        ;;
    train|training|train_analysis)
        MODE="train_analysis"
        ;;
    *)
        echo "ERROR: unsupported --mode $MODE (expected analysis, train, or train_analysis)" >&2
        exit 1
        ;;
esac

if [[ "$MODE" == "train_analysis" ]]; then
    STAGES="train_analysis"
    CHECKPOINT="$TRAIN_ANALYSIS_CHECKPOINT_DIR"
    if [[ "$OUTPUT_EXPLICIT" == "0" ]]; then
        if [[ "$(path_name "$TRAIN_ANALYSIS_CHECKPOINT_DIR")" == "checkpoints" ]]; then
            OUTPUT="$(path_parent "$TRAIN_ANALYSIS_CHECKPOINT_DIR")/side_analysis"
        else
            OUTPUT="${TRAIN_ANALYSIS_CHECKPOINT_DIR%/}/side_analysis"
        fi
    fi
fi

if [[ "$WORKERS" = "auto" ]]; then
    WORKERS="all"
fi

if [[ "$REPO_URL" == git@github.com:* ]]; then
    REPO_URL="https://github.com/${REPO_URL#git@github.com:}"
fi
if [[ -n "$GH_TOKEN" && "$REPO_URL" == https://github.com/* ]]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO_URL#https://github.com/}"
fi
REPO_URL_DISPLAY="$REPO_URL"
if [[ -n "$GH_TOKEN" ]]; then
    REPO_URL_DISPLAY="${REPO_URL_DISPLAY/$GH_TOKEN/***}"
fi

COPY_CMD="bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh --tpu $TPU_NAME --project $PROJECT --zone $ZONE --branch $BRANCH --mode $MODE"
if [[ -n "$PRESET" ]]; then
    COPY_CMD="$COPY_CMD --preset $PRESET"
fi
if [[ "$MODE" == "train_analysis" ]]; then
    COPY_CMD="$COPY_CMD --checkpoint-dir $TRAIN_ANALYSIS_CHECKPOINT_DIR"
    if [[ -n "$TRAIN_ANALYSIS_CONFIG" ]]; then
        COPY_CMD="$COPY_CMD --config $TRAIN_ANALYSIS_CONFIG"
    fi
    if [[ "$OUTPUT_EXPLICIT" == "1" ]]; then
        COPY_CMD="$COPY_CMD --output $OUTPUT"
    fi
    if [[ "$REMOTE_LOG_EXPLICIT" == "1" ]]; then
        COPY_CMD="$COPY_CMD --log $REMOTE_LOG"
    fi
fi
WATCH_LOG_CMD="bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --log $REMOTE_LOG --target $TMUX_SESSION --summary"

echo "============================================================"
echo "DAWN-SRW v4166 analysis launcher"
echo "============================================================"
echo "Run:"
echo "  mode            : $MODE"
echo "  tpu             : $TPU_NAME"
echo "  project         : $PROJECT"
echo "  zone            : $ZONE"
echo "  branch          : $BRANCH"
echo "  repo            : $REPO_URL_DISPLAY"
if [[ -n "$PRESET" ]]; then
    echo "  preset          : $PRESET"
fi
echo "  workers         : $WORKERS"
echo "  detached        : $DETACH"
echo "  tmux_session    : $TMUX_SESSION"
echo "  remote_log      : $REMOTE_LOG"
if [[ "$MODE" == "train_analysis" ]]; then
    echo "  config          : ${TRAIN_ANALYSIS_CONFIG:-checkpoint full_config}"
    echo "  checkpoint_dir  : $TRAIN_ANALYSIS_CHECKPOINT_DIR"
    echo "  output          : $OUTPUT"
    echo "  analysis_batches: $TRAIN_ANALYSIS_MAX_BATCHES"
    echo "  prune_eps       : $TRAIN_ANALYSIS_PRUNE_EPS"
else
    echo "  checkpoint      : $CHECKPOINT"
    echo "  output          : $OUTPUT"
    echo "  stages          : $STAGES"
    echo "  args            : ${ANALYSIS_ARGS:-<none>}"
fi
echo ""
echo "Copy-paste:"
echo "  $COPY_CMD"
echo ""
echo "Watch logs:"
echo "  $WATCH_LOG_CMD"
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run: no TPU command will be sent."
    if [[ "$MODE" == "train_analysis" ]]; then
        echo "Remote Python:"
        if [[ -n "$TRAIN_ANALYSIS_CONFIG" ]]; then
            echo "  python3 -u scripts/analyze_dawn_srw_v4166.py --train-analysis --config $TRAIN_ANALYSIS_CONFIG --checkpoint-dir $TRAIN_ANALYSIS_CHECKPOINT_DIR --output $OUTPUT --train-analysis-max-batches $TRAIN_ANALYSIS_MAX_BATCHES --prune-eps $TRAIN_ANALYSIS_PRUNE_EPS --init-distributed"
        else
            echo "  python3 -u scripts/analyze_dawn_srw_v4166.py --train-analysis --checkpoint-dir $TRAIN_ANALYSIS_CHECKPOINT_DIR --output $OUTPUT --train-analysis-max-batches $TRAIN_ANALYSIS_MAX_BATCHES --prune-eps $TRAIN_ANALYSIS_PRUNE_EPS --init-distributed"
        fi
    else
        echo "Remote Python:"
        echo "  python3 -u scripts/analyze_dawn_srw_v4166.py --checkpoint $CHECKPOINT --output $OUTPUT --stages $STAGES --init-distributed ${ANALYSIS_ARGS:-}"
    fi
    exit 0
fi

echo "Checking TPU status..."
TPU_STATE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)")"
echo "$TPU_STATE"

ACCELERATOR_TYPE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(acceleratorType)" || true)"
NETWORK_ENDPOINTS="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(networkEndpoints[].ipAddress)" || true)"
WORKER_COUNT="$(printf '%s\n' "$NETWORK_ENDPOINTS" | awk 'NF {count += NF} END {print count + 0}')"
ACCELERATOR_WORKER_COUNT=0
ACCELERATOR_SIZE="${ACCELERATOR_TYPE##*-}"
if [[ "$ACCELERATOR_SIZE" =~ ^[0-9]+$ ]]; then
    ACCELERATOR_WORKER_COUNT=$(( (ACCELERATOR_SIZE + 7) / 8 ))
fi
if [ "$ACCELERATOR_WORKER_COUNT" -gt "$WORKER_COUNT" ]; then
    WORKER_COUNT="$ACCELERATOR_WORKER_COUNT"
fi
if [ "$WORKER_COUNT" -le 0 ]; then
    echo "ERROR: Could not determine TPU worker count." >&2
    exit 1
fi
echo "  Accelerator: ${ACCELERATOR_TYPE:-unknown}"
echo "  Detected workers: $WORKER_COUNT"

declare -a TARGET_WORKERS=()
if [[ "$WORKERS" = "all" ]]; then
    for worker in $(seq 0 $((WORKER_COUNT - 1))); do
        TARGET_WORKERS+=("$worker")
    done
else
    TARGET_WORKERS+=("$WORKERS")
fi

run_worker_command() {
    local worker="$1"
    local command="$2"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --worker="$worker" \
        --command="$command"
}

echo "Preflighting SSH on target worker(s): ${TARGET_WORKERS[*]}"
for worker in "${TARGET_WORKERS[@]}"; do
    echo "  Worker $worker SSH preflight..."
    if ! run_worker_command "$worker" 'hostname; date -Is'; then
        echo "ERROR: worker $worker SSH failed. Aborting launch." >&2
        exit 1
    fi
done

if [[ "$MODE" == "train_analysis" ]]; then
read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
ANALYSIS_PATTERN="[a]nalyze_dawn_srw_v4166"
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PYTHON_PATTERN="[p]ython3"
PGREP_PATTERN="${ANALYSIS_PATTERN}|${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}|${PYTHON_PATTERN} scripts"
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
pkill -9 -f "${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
sleep 3
pgrep -af "$PGREP_PATTERN" || true
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: DAWN train/analysis process remains after cleanup:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
EOFCLEANUP
else
read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
ANALYSIS_PATTERN="[a]nalyze_dawn_srw_v4166"
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PGREP_PATTERN="${ANALYSIS_PATTERN}|${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}"
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
sleep 3
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: DAWN process remains after cleanup:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
EOFCLEANUP
fi

cleanup_target_workers() {
    local failed=0
    for worker in "${TARGET_WORKERS[@]}"; do
        echo "  Cleaning worker $worker..."
        if ! run_worker_command "$worker" "$CLEANUP_CMD"; then
            echo "ERROR: worker $worker cleanup failed." >&2
            failed=1
        fi
    done
    return "$failed"
}

if [[ "$MODE" == "train_analysis" && "$DETACH" == "0" ]]; then
    echo "Cleaning old train_analysis processes on target worker(s)..."
else
    echo "Cleaning old train/analysis processes on target worker(s)..."
fi
if ! cleanup_target_workers; then
    echo "ERROR: cleanup verification failed. Aborting launch." >&2
    exit 1
fi

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CHECKPOINT='${CHECKPOINT}'
OUTPUT='${OUTPUT}'
STAGES='${STAGES}'
ANALYSIS_ARGS='${ANALYSIS_ARGS}'
MODE='${MODE}'
TRAIN_ANALYSIS_CONFIG='${TRAIN_ANALYSIS_CONFIG}'
TRAIN_ANALYSIS_CHECKPOINT_DIR='${TRAIN_ANALYSIS_CHECKPOINT_DIR}'
TRAIN_ANALYSIS_MAX_BATCHES='${TRAIN_ANALYSIS_MAX_BATCHES}'
TRAIN_ANALYSIS_PRUNE_EPS='${TRAIN_ANALYSIS_PRUNE_EPS}'
DETACH='${DETACH}'
INSTALL_DEPS='${INSTALL_DEPS}'
TMUX_SESSION='${TMUX_SESSION}'
REMOTE_LOG='${REMOTE_LOG}'
REMOTE_LOG_PATH="\${REMOTE_LOG/#\\~/\$HOME}"
WORK_DIR="\$HOME/DAWN-SRW"

echo "=== DAWN v4166 analysis worker startup ==="
echo "HOSTNAME=\$(hostname)"
echo "DATE=\$(date -Is)"
echo "BRANCH=\$BRANCH"
echo "MODE=\$MODE"
echo "CHECKPOINT=\$CHECKPOINT"
echo "OUTPUT=\$OUTPUT"
echo "STAGES=\$STAGES"
echo "ANALYSIS_ARGS=\$ANALYSIS_ARGS"
if [ "\$MODE" = "train_analysis" ]; then
    echo "TRAIN_ANALYSIS_CONFIG=\${TRAIN_ANALYSIS_CONFIG:-checkpoint full_config}"
    echo "TRAIN_ANALYSIS_CHECKPOINT_DIR=\$TRAIN_ANALYSIS_CHECKPOINT_DIR"
    echo "TRAIN_ANALYSIS_MAX_BATCHES=\$TRAIN_ANALYSIS_MAX_BATCHES"
    echo "TRAIN_ANALYSIS_PRUNE_EPS=\$TRAIN_ANALYSIS_PRUNE_EPS"
fi

if [ -d "\$WORK_DIR/.git" ]; then
    cd "\$WORK_DIR"
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
    echo "Repo updated to \$BRANCH"
else
    rm -rf "\$WORK_DIR"
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
    cd "\$WORK_DIR"
    echo "Repo cloned (branch: \$BRANCH)"
fi

if [ "\$INSTALL_DEPS" = "1" ]; then
    echo "[setup] installing TPU analysis dependencies"
    python3 -m pip install --upgrade pip -q
    python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
    python3 -m pip install flax optax orbax-checkpoint==0.11.24 numpy pyyaml gcsfs transformers matplotlib -q
fi

export PYTHONUNBUFFERED=1
export DAWN_ANALYSIS_INIT_DISTRIBUTED=1
export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"

if [ "\$MODE" = "train_analysis" ]; then
    ANALYSIS_CMD=(
        python3 -u scripts/analyze_dawn_srw_v4166.py
        --train-analysis
        --checkpoint-dir "\$TRAIN_ANALYSIS_CHECKPOINT_DIR"
        --output "\$OUTPUT"
        --train-analysis-max-batches "\$TRAIN_ANALYSIS_MAX_BATCHES"
        --prune-eps "\$TRAIN_ANALYSIS_PRUNE_EPS"
        --init-distributed
    )
    if [ -n "\$TRAIN_ANALYSIS_CONFIG" ]; then
        ANALYSIS_CMD+=(--config "\$TRAIN_ANALYSIS_CONFIG")
    fi
else
    ANALYSIS_CMD=(
        python3 -u scripts/analyze_dawn_srw_v4166.py
        --checkpoint "\$CHECKPOINT"
        --output "\$OUTPUT"
        --stages "\$STAGES"
        --init-distributed
    )
fi
if [ -n "\$ANALYSIS_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(\$ANALYSIS_ARGS)
    ANALYSIS_CMD+=("\${EXTRA_ARGS[@]}")
fi
ANALYSIS_CMD_STR=\$(printf "%q " "\${ANALYSIS_CMD[@]}")

cd "\$WORK_DIR"
mkdir -p "\$(dirname "\$REMOTE_LOG_PATH")"
: > "\$REMOTE_LOG_PATH"
if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session \$TMUX_SESSION"
    tmux kill-session -t "\$TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -x 240 -y 60 -s "\$TMUX_SESSION" \
        "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; export DAWN_ANALYSIS_INIT_DISTRIBUTED=1; export JAX_TRACEBACK_FILTERING='\$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='\$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='\$TF_CPP_MIN_LOG_LEVEL'; { echo '=== TPU analysis process startup ==='; echo \"HOSTNAME=\$(hostname)\"; echo \"DATE=\$(date -Is)\"; echo \"CMD: \$ANALYSIS_CMD_STR\"; \$ANALYSIS_CMD_STR; status=\$?; echo \"Analysis exited with status \$status\"; exit \$status; } 2>&1 | tee '\$REMOTE_LOG_PATH'"
    echo "[run] detached in tmux session \$TMUX_SESSION, log=\$REMOTE_LOG_PATH"
else
    echo "[run] foreground analysis"
    "\${ANALYSIS_CMD[@]}" 2>&1 | tee "\$REMOTE_LOG_PATH"
fi
EOFCMD

echo "Sending analysis command to target worker(s): ${TARGET_WORKERS[*]}"
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
declare -a LAUNCH_PIDS=()
declare -a LAUNCH_LOGS=()
for worker in "${TARGET_WORKERS[@]}"; do
    log_file="launch_dawn_v4166_analysis_${TPU_NAME}_${LAUNCH_TS}_worker_${worker}.log"
    LAUNCH_LOGS[$worker]="$log_file"
    echo "  Worker $worker launch starting (log: $log_file)"
    (
        run_worker_command "$worker" "$REMOTE_CMD"
    ) >"$log_file" 2>&1 &
    LAUNCH_PIDS[$worker]=$!
done

declare -a FAILED_WORKERS=()
for worker in "${TARGET_WORKERS[@]}"; do
    if ! wait "${LAUNCH_PIDS[$worker]}"; then
        FAILED_WORKERS+=("$worker")
        echo "ERROR: worker $worker setup/start failed. See ${LAUNCH_LOGS[$worker]}" >&2
    fi
done

if [ "${#FAILED_WORKERS[@]}" -gt 0 ]; then
    echo "ERROR: launch failed on worker(s): ${FAILED_WORKERS[*]}. Cleaning up target workers." >&2
    cleanup_target_workers || true
    exit 1
fi

echo ""
if [[ "$MODE" == "train_analysis" && "$DETACH" == "0" ]]; then
    SUMMARY_LOG=""
    SUMMARY_WORKER=""
    for worker in "${TARGET_WORKERS[@]}"; do
        log_file="${LAUNCH_LOGS[$worker]}"
        if grep -q "DAWN-SRW v4166 TRAIN ANALYSIS" "$log_file"; then
            SUMMARY_LOG="$log_file"
            SUMMARY_WORKER="$worker"
            break
        fi
    done
    echo "Train analysis complete."
    if [[ -n "$SUMMARY_LOG" ]]; then
        echo "  Summary worker : $SUMMARY_WORKER"
        echo "  Summary log    : $SUMMARY_LOG"
        echo ""
        awk 'BEGIN{show=0} /^============================================================$/ {show=1} show {print}' "$SUMMARY_LOG"
    else
        echo "WARNING: train_analysis summary block was not found in local launch logs." >&2
        echo "Local launch logs:"
        for worker in "${TARGET_WORKERS[@]}"; do
            echo "  worker $worker: ${LAUNCH_LOGS[$worker]}"
        done
    fi
    echo ""
    echo "Copy-paste:"
    echo "  $COPY_CMD"
    echo ""
    echo "Watch logs:"
    echo "  $WATCH_LOG_CMD"
    exit 0
fi

echo "Launch complete. Analysis is running in tmux session '$TMUX_SESSION' on target workers."
echo "  Primary log:     bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT"
echo "  Primary pane:    bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --pane"
echo "  Attach primary:  bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --attach"
echo "  Primary summary: bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --summary"
echo "  All summary:     bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --all --summary"
echo "  Literal worker 0 log: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach literal worker 0: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"
echo "  Your capture flow works on whichever worker you attach: tmux pipe-pane -t train 'cat >> ~/rebuttal_log.txt'"
