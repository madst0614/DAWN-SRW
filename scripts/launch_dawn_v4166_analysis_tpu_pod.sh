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

CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_400M_c4_40B_v4_64/run_vspatial-r1-v4.1.6.6_20260622_212706_3201/checkpoints/000000076293"
OUTPUT="gs://dawn-tpu-data-c4/analysis/v4166_400M_final"
STAGES="eval,prune,geometry,usage,trace,ablation,report"
ANALYSIS_ARGS=""
WORKERS="auto"
DETACH="1"
INSTALL_DEPS="1"
TMUX_SESSION="train"
REMOTE_LOG="~/train.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --foreground) DETACH="0"; shift ;;
        --detach) DETACH="1"; shift ;;
        --no-install) INSTALL_DEPS="0"; shift ;;
        --from-scratch) ANALYSIS_ARGS="$ANALYSIS_ARGS --from-scratch"; shift ;;
        --retry-failed) ANALYSIS_ARGS="$ANALYSIS_ARGS --retry-failed"; shift ;;
        --fail-fast) ANALYSIS_ARGS="$ANALYSIS_ARGS --fail-fast"; shift ;;
        --mesh-data) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-data $2"; shift 2 ;;
        --mesh-model) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-model $2"; shift 2 ;;
        --eval-max-tokens) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-max-tokens $2"; shift 2 ;;
        --eval-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-batch-size $2"; shift 2 ;;
        --prune-eps) ANALYSIS_ARGS="$ANALYSIS_ARGS --prune-eps $2"; shift 2 ;;
        --usage-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-max-sequences $2"; shift 2 ;;
        --usage-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-batch-size $2"; shift 2 ;;
        --usage-seq-len) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-seq-len $2"; shift 2 ;;
        --trace-max-prompts) ANALYSIS_ARGS="$ANALYSIS_ARGS --trace-max-prompts $2"; shift 2 ;;
        --ablation-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-max-sequences $2"; shift 2 ;;
        --ablation-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-batch-size $2"; shift 2 ;;
        --ablation-k-list) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-k-list $2"; shift 2 ;;
        --ablation-strategies) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-strategies $2"; shift 2 ;;
        --max-jobs-per-stage) ANALYSIS_ARGS="$ANALYSIS_ARGS --max-jobs-per-stage $2"; shift 2 ;;
        --extra-arg) ANALYSIS_ARGS="$ANALYSIS_ARGS $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [options]"
            echo ""
            echo "Required:"
            echo "  --tpu NAME"
            echo ""
            echo "Core:"
            echo "  --checkpoint PATH_OR_GS   Default: $CHECKPOINT"
            echo "  --output PATH_OR_GS       Default: $OUTPUT"
            echo "  --stages CSV              Default: $STAGES"
            echo "  --branch BRANCH           Default: $BRANCH"
            echo "  --repo-url URL            Default: $REPO_URL"
            echo ""
            echo "TPU/GCP:"
            echo "  --zone ZONE               Default: $ZONE"
            echo "  --project PROJECT         Default: $PROJECT"
            echo "  --workers all|0|N         Default: all"
            echo ""
            echo "Execution:"
            echo "  --detach                  Run in tmux session train (default)"
            echo "  --foreground              Run foreground on the SSH command"
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

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu required" >&2
    exit 1
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

echo "============================================"
echo "Launching DAWN-SRW v4166 analysis on TPU"
echo "  TPU:        $TPU_NAME"
echo "  Zone:       $ZONE"
echo "  Project:    $PROJECT"
echo "  Branch:     $BRANCH"
echo "  Repo:       $REPO_URL_DISPLAY"
echo "  Workers:    $WORKERS"
echo "  Detached:   $DETACH"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output:     $OUTPUT"
echo "  Stages:     $STAGES"
echo "  Args:       ${ANALYSIS_ARGS:-<none>}"
echo "============================================"

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

echo "Cleaning old train/analysis processes on target worker(s)..."
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
DETACH='${DETACH}'
INSTALL_DEPS='${INSTALL_DEPS}'
TMUX_SESSION='${TMUX_SESSION}'
REMOTE_LOG='${REMOTE_LOG}'
WORK_DIR="\$HOME/dawn-spatial"

echo "=== DAWN v4166 analysis worker startup ==="
echo "HOSTNAME=\$(hostname)"
echo "DATE=\$(date -Is)"
echo "BRANCH=\$BRANCH"
echo "CHECKPOINT=\$CHECKPOINT"
echo "OUTPUT=\$OUTPUT"
echo "STAGES=\$STAGES"
echo "ANALYSIS_ARGS=\$ANALYSIS_ARGS"

if [ -d "\$WORK_DIR/.git" ]; then
    cd "\$WORK_DIR"
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
else
    rm -rf "\$WORK_DIR"
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
    cd "\$WORK_DIR"
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

ANALYSIS_CMD=(
    python3 -u scripts/analyze_dawn_srw_v4166.py
    --checkpoint "\$CHECKPOINT"
    --output "\$OUTPUT"
    --stages "\$STAGES"
    --init-distributed
)
if [ -n "\$ANALYSIS_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(\$ANALYSIS_ARGS)
    ANALYSIS_CMD+=("\${EXTRA_ARGS[@]}")
fi
ANALYSIS_CMD_STR=\$(printf "%q " "\${ANALYSIS_CMD[@]}")

cd "\$WORK_DIR"
if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session \$TMUX_SESSION"
    tmux kill-session -t "\$TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -s "\$TMUX_SESSION" \
        "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; export DAWN_ANALYSIS_INIT_DISTRIBUTED=1; export JAX_TRACEBACK_FILTERING='\$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='\$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='\$TF_CPP_MIN_LOG_LEVEL'; { echo '=== TPU analysis process startup ==='; echo \"HOSTNAME=\$(hostname)\"; echo \"DATE=\$(date -Is)\"; echo \"CMD: \$ANALYSIS_CMD_STR\"; \$ANALYSIS_CMD_STR; } 2>&1 | tee \$REMOTE_LOG; echo 'Analysis finished. Press enter to close.'; read"
    echo "[run] detached in tmux session \$TMUX_SESSION, log=\$REMOTE_LOG"
else
    echo "[run] foreground analysis"
    "\${ANALYSIS_CMD[@]}" 2>&1 | tee "\${REMOTE_LOG/#\\~/$HOME}"
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
echo "Launch request sent."
echo "  tmux session: $TMUX_SESSION"
echo "  Worker 0 log: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Watch: bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT"
echo "  Summary: bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --summary"
echo "  All hosts: bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --all --summary"
echo "  Worker 0 attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Your capture flow works on worker 0: tmux pipe-pane -t train 'cat >> ~/rebuttal_log.txt'"
