#!/bin/bash
# =============================================================================
# TPU VM/Pod Launcher -- run from local machine or Cloud Shell
# =============================================================================
# Sends setup_and_run_tpu_pod.sh to all workers with the specified branch/config.
# Supports single-host TPU VMs such as v4-8 and multi-host TPU pods such as
# v4-64 or v4-128. This script assumes the TPU VM/queued resource already exists.
#
# Usage:
#   bash scripts/launch_tpu_pod.sh --tpu spatial-analysis1 --config configs/v4_8.yaml
#   bash scripts/launch_tpu_pod.sh --tpu dawn-400m-v4-64 --branch main --config configs/v4_64.yaml
#   bash scripts/launch_tpu_pod.sh  # uses defaults (v4-64 settings)
#
# Prerequisites:
#   1. TPU VM or queued resource created separately:
#      gcloud compute tpus tpu-vm create dawn-400m-v4-64 \
#        --zone=us-central2-b --accelerator-type=v4-64 \
#        --version=tpu-vm-v4-base --spot
# =============================================================================

set -euo pipefail

# Defaults
TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG="configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml"
GH_TOKEN=""
TRAIN_ARGS=""
TRAIN_SCRIPT="scripts/train_jax.py"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu)      TPU_NAME="$2"; shift 2 ;;
        --zone)     ZONE="$2";     shift 2 ;;
        --project)  PROJECT="$2";  shift 2 ;;
        --branch)   BRANCH="$2";   shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        --script)   TRAIN_SCRIPT="$2"; shift 2 ;;
        --token)    GH_TOKEN="$2"; shift 2 ;;
        --from-scratch) TRAIN_ARGS="$TRAIN_ARGS --from-scratch"; shift ;;
        --resume-from) TRAIN_ARGS="$TRAIN_ARGS --resume-from $2"; shift 2 ;;
        --debug)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                TRAIN_ARGS="$TRAIN_ARGS --debug $2"
                shift 2
            else
                TRAIN_ARGS="$TRAIN_ARGS --debug"
                shift
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [--tpu NAME] [--zone ZONE] [--project PROJECT] [--branch BRANCH] [--config CONFIG] [--script TRAIN_SCRIPT] [--token GH_TOKEN] [--from-scratch] [--resume-from RUN_FOLDER_OR_ORBAX_STEP] [--debug [N]]"
            echo ""
            echo "Supports single-host TPU VMs such as v4-8 and multi-host TPU pods such as v4-64/v4-128."
            echo "The TPU VM or queued resource must already exist; this script only launches setup/training."
            echo ""
            echo "  --tpu      TPU VM name         (default: $TPU_NAME)"
            echo "  --zone     GCP zone            (default: $ZONE)"
            echo "  --project  GCP project          (default: $PROJECT)"
            echo "  --branch   Git branch to clone  (default: $BRANCH)"
            echo "  --config   Training config YAML (default: $CONFIG)"
            echo "  --script   Training script      (default: $TRAIN_SCRIPT)"
            echo "  --from-scratch  Start training from scratch (ignore checkpoints)"
            echo "  --resume-from RUN_FOLDER_OR_ORBAX_STEP  Resume from an Orbax run folder or step directory"
            echo "  --debug [N]  Enable train_jax.py debug diagnostics every N steps (default: 1)"
            echo "  --token    GitHub access token   (for private repos)"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Launching TPU VM/Pod training"
echo "  TPU:     $TPU_NAME"
echo "  Zone:    $ZONE"
echo "  Project: $PROJECT"
echo "  Branch:  $BRANCH"
echo "  Config:  $CONFIG"
echo "  Script:  $TRAIN_SCRIPT"
if [ -n "$TRAIN_ARGS" ]; then
echo "  Args:    $TRAIN_ARGS"
fi
echo "============================================"

# Check TPU status and discover worker count
echo "Checking TPU status..."
TPU_STATE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)")"
echo "$TPU_STATE"
ACCELERATOR_TYPE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(acceleratorType)")"
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
echo "  Accelerator: $ACCELERATOR_TYPE"
echo "  Workers:     $WORKER_COUNT"

if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/DAWN-SRW.git"
else
    REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
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

echo "Preflighting SSH on every worker..."
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    echo "=== Launching TPU worker index $worker ==="
    echo "  Worker $worker SSH preflight..."
    if ! run_worker_command "$worker" 'hostname; date -Is'; then
        echo "ERROR: worker $worker SSH failed. Aborting launch. No training started." >&2
        exit 1
    fi
done

read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PYTHON_PATTERN="[p]ython3"
PGREP_PATTERN="${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}|${PYTHON_PATTERN} scripts"
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
sleep 3
pgrep -af "$PGREP_PATTERN" || true
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: DAWN training process remains after cleanup:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
EOFCLEANUP

cleanup_all_workers() {
    local failed=0
    for worker in $(seq 0 $((WORKER_COUNT - 1))); do
        echo "  Cleaning worker $worker..."
        if ! run_worker_command "$worker" "$CLEANUP_CMD"; then
            echo "ERROR: worker $worker cleanup failed." >&2
            failed=1
        fi
    done
    return "$failed"
}

echo "Cleaning old training processes on every worker..."
if ! cleanup_all_workers; then
    echo "ERROR: cleanup verification failed. Aborting launch." >&2
    exit 1
fi

# Build inline bootstrap: clone/update repo first, then run setup script
read -r -d '' REMOTE_CMD_TEMPLATE <<EOFCMD || true
set -e
TPU_WORKER_INDEX='__TPU_WORKER_INDEX__'
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
TRAIN_SCRIPT='${TRAIN_SCRIPT}'
GH_TOKEN='${GH_TOKEN}'
TRAIN_ARGS='${TRAIN_ARGS}'
export TPU_WORKER_INDEX BRANCH CONFIG TRAIN_SCRIPT GH_TOKEN TRAIN_ARGS

echo "=== Launching TPU worker index \$TPU_WORKER_INDEX ==="
echo "TPU_WORKER_INDEX=\$TPU_WORKER_INDEX"
echo "HOSTNAME=\$(hostname)"
echo "DATE=\$(date -Is)"

# Bootstrap: ensure ~/DAWN-SRW exists with the right branch
if [ -d ~/DAWN-SRW/.git ]; then
    cd ~/DAWN-SRW
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
    echo "Repo updated to \$BRANCH"
else
    rm -rf ~/DAWN-SRW
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" ~/DAWN-SRW
    echo "Repo cloned (branch: \$BRANCH)"
fi

# Run the setup+training script (nohup inside will detach training)
cd ~/DAWN-SRW
bash scripts/setup_and_run_tpu_pod.sh
EOFCMD

# Send command to all workers
echo "Sending bootstrap+training command to all workers..."
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
declare -a LAUNCH_PIDS=()
declare -a LAUNCH_LOGS=()
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    log_file="launch_${TPU_NAME}_${LAUNCH_TS}_worker_${worker}.log"
    LAUNCH_LOGS[$worker]="$log_file"
    REMOTE_CMD="${REMOTE_CMD_TEMPLATE//__TPU_WORKER_INDEX__/$worker}"
    echo "=== Launching TPU worker index $worker ==="
    (
        run_worker_command "$worker" "$REMOTE_CMD"
    ) >"$log_file" 2>&1 &
    LAUNCH_PIDS[$worker]=$!
    echo "  Worker $worker launch started (log: $log_file)"
done

declare -a FAILED_WORKERS=()
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    if ! wait "${LAUNCH_PIDS[$worker]}"; then
        FAILED_WORKERS+=("$worker")
        echo "ERROR: worker $worker setup/start failed. See ${LAUNCH_LOGS[$worker]}" >&2
    fi
done

if [ "${#FAILED_WORKERS[@]}" -gt 0 ]; then
    echo "ERROR: launch failed on worker(s): ${FAILED_WORKERS[*]}. Cleaning up all workers." >&2
    echo "First-failure log grep: bash scripts/grep_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --workers $WORKER_COUNT" >&2
    cleanup_all_workers || true
    exit 1
fi

echo ""
echo "Launch complete. Training is running in tmux session 'train' on all workers."
echo "  Log:     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tail -f ~/train.log'"
echo "  Attach:  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tmux attach -t train'"
echo "  Kill:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command='tmux kill-session -t train'"
echo "  First failure grep: bash scripts/grep_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --workers $WORKER_COUNT"
