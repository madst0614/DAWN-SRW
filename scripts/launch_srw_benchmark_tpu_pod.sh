#!/bin/bash
# =============================================================================
# DAWN-SRW TPU Pod Benchmark Launcher
# =============================================================================
# Launches scripts/benchmark_srw_tpu.py on every TPU host. Pass --config more
# than once to run configs sequentially and print the comparison in ~/train.log.
# Results are console-first by default: no JSONL/MD/CSV files are produced.
# =============================================================================

set -euo pipefail

TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIGS=()
MODEL_VERSION=""
STEPS="20"
WARMUP_STEPS="5"
OUTPUT_DIR=""
XLA_DUMP_ENABLED="1"
XLA_DUMP_BASE=""
GH_TOKEN=""
DUMMY_DATA="0"
ALLOW_MODEL_VERSION_OVERRIDE="0"

usage() {
    cat <<EOF
Usage: $0 --tpu NAME --branch BRANCH --config CONFIG [--config CONFIG2 ...] [options]

Options:
  --zone ZONE
  --project PROJECT
  --steps N
  --warmup-steps N
  --model-version VERSION        Optional expected version check
  --allow-model-version-override Allow --model-version to override config
  --output-dir DIR               Remote artifact root for XLA dumps only
  --xla-dump [DIR]               Enable XLA dumps (default: enabled)
  --no-xla-dump                  Disable XLA dumps
  --dummy-data                   Explicit synthetic-data smoke test
  --token TOKEN                  GitHub token for private repos
  -h, --help

Example:
  bash scripts/launch_srw_benchmark_tpu_pod.sh \\
    --tpu spatial-400m \\
    --branch codex/v4167-poc \\
    --config configs/train_config_v4168_400M_c4_40B_v4_64_block_sparse.yaml \\
    --config configs/train_config_v4168_400M_c4_40B_v4_64_block_sparse_2.yaml
EOF
}

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

shell_quote() {
    printf '%q' "$1"
}

join_shell_quoted_args() {
    local out=""
    local arg=""
    local quoted=""
    for arg in "$@"; do
        printf -v quoted '%q' "$arg"
        if [[ -n "$out" ]]; then
            out+=" "
        fi
        out+="$quoted"
    done
    printf '%s' "$out"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --config) CONFIGS+=("$2"); shift 2 ;;
        --model-version) MODEL_VERSION="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --xla-dump)
            XLA_DUMP_ENABLED="1"
            if [[ $# -ge 2 && "$2" != --* ]]; then
                XLA_DUMP_BASE="$2"
                shift 2
            else
                shift
            fi
            ;;
        --no-xla-dump) XLA_DUMP_ENABLED="0"; shift ;;
        --dummy-data) DUMMY_DATA="1"; shift ;;
        --allow-model-version-override) ALLOW_MODEL_VERSION_OVERRIDE="1"; shift ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "ERROR: at least one --config is required." >&2
    exit 1
fi
if [[ -n "$MODEL_VERSION" &&
      "$MODEL_VERSION" != "spatial-r1-v4.1.6.6" &&
      "$MODEL_VERSION" != "spatial-r1-v4.1.6.8" ]]; then
    echo "ERROR: --model-version must be spatial-r1-v4.1.6.6 or spatial-r1-v4.1.6.8." >&2
    exit 1
fi
if ! is_nonnegative_int "$STEPS" || [[ "$STEPS" -le 0 ]]; then
    echo "ERROR: --steps must be an integer > 0." >&2
    exit 1
fi
if ! is_nonnegative_int "$WARMUP_STEPS"; then
    echo "ERROR: --warmup-steps must be an integer >= 0." >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="benchmark_runs/srw_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$XLA_DUMP_BASE" ]]; then
    XLA_DUMP_BASE="${OUTPUT_DIR}/xla"
fi

CONFIG_FIRST="${CONFIGS[0]}"
CONFIG_REST_ARGS=""
if [[ "${#CONFIGS[@]}" -gt 1 ]]; then
    CONFIG_REST_ARGS="$(join_shell_quoted_args "${CONFIGS[@]:1}")"
fi
CONFIG_COUNT="${#CONFIGS[@]}"

MODEL_VERSION_LABEL="${MODEL_VERSION:-from config}"

echo "============================================"
echo "Launching SRW TPU benchmark"
echo "  TPU:            $TPU_NAME"
echo "  Zone:           $ZONE"
echo "  Project:        $PROJECT"
echo "  Branch:         $BRANCH"
echo "  Config count:   $CONFIG_COUNT"
for config in "${CONFIGS[@]}"; do
    echo "    - $config"
done
echo "  Model version:  $MODEL_VERSION_LABEL"
echo "  Steps:          $STEPS"
echo "  Warmup steps:   $WARMUP_STEPS"
echo "  XLA dumps:      $XLA_DUMP_ENABLED ($XLA_DUMP_BASE)"
echo "  Dummy data:     $DUMMY_DATA"
echo "  Result files:   none by default; benchmark prints to stdout/train.log"
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
if [[ "$ACCELERATOR_WORKER_COUNT" -gt "$WORKER_COUNT" ]]; then
    WORKER_COUNT="$ACCELERATOR_WORKER_COUNT"
fi
if [[ "$WORKER_COUNT" -le 0 ]]; then
    echo "ERROR: Could not determine TPU worker count." >&2
    exit 1
fi
echo "  Accelerator: $ACCELERATOR_TYPE"
echo "  Workers:     $WORKER_COUNT"

if [[ -n "$GH_TOKEN" ]]; then
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
    echo "  Worker $worker SSH preflight..."
    if ! run_worker_command "$worker" 'hostname; date -Is'; then
        echo "ERROR: worker $worker SSH failed. Aborting launch." >&2
        exit 1
    fi
done

read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "[p]ython3 .*scripts/benchmark_srw_tpu.py" || true
pkill -9 -f "[p]ython .*scripts/benchmark_srw_tpu.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
EOFCLEANUP

echo "Cleaning old benchmark processes on every worker..."
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    echo "  Cleaning worker $worker..."
    run_worker_command "$worker" "$CLEANUP_CMD" || true
done

REPO_URL_Q="$(shell_quote "$REPO_URL")"
BRANCH_Q="$(shell_quote "$BRANCH")"
CONFIG_FIRST_Q="$(shell_quote "$CONFIG_FIRST")"
CONFIG_REST_Q="$(shell_quote "$CONFIG_REST_ARGS")"
MODEL_VERSION_Q="$(shell_quote "$MODEL_VERSION")"
STEPS_Q="$(shell_quote "$STEPS")"
WARMUP_STEPS_Q="$(shell_quote "$WARMUP_STEPS")"
XLA_DUMP_ENABLED_Q="$(shell_quote "$XLA_DUMP_ENABLED")"
XLA_DUMP_BASE_Q="$(shell_quote "$XLA_DUMP_BASE")"
DUMMY_DATA_Q="$(shell_quote "$DUMMY_DATA")"
ALLOW_MODEL_VERSION_OVERRIDE_Q="$(shell_quote "$ALLOW_MODEL_VERSION_OVERRIDE")"
GH_TOKEN_Q="$(shell_quote "$GH_TOKEN")"

read -r -d '' REMOTE_CMD_TEMPLATE <<EOFCMD || true
set -e
TPU_WORKER_INDEX='__TPU_WORKER_INDEX__'
REPO_URL=${REPO_URL_Q}
BRANCH=${BRANCH_Q}
CONFIG=${CONFIG_FIRST_Q}
CONFIG_REST=${CONFIG_REST_Q}
MODEL_VERSION=${MODEL_VERSION_Q}
STEPS=${STEPS_Q}
WARMUP_STEPS=${WARMUP_STEPS_Q}
XLA_DUMP_ENABLED=${XLA_DUMP_ENABLED_Q}
XLA_DUMP_BASE=${XLA_DUMP_BASE_Q}
DUMMY_DATA=${DUMMY_DATA_Q}
ALLOW_MODEL_VERSION_OVERRIDE=${ALLOW_MODEL_VERSION_OVERRIDE_Q}
GH_TOKEN=${GH_TOKEN_Q}
export TPU_WORKER_INDEX BRANCH CONFIG GH_TOKEN

echo "=== Launching SRW benchmark worker \$TPU_WORKER_INDEX ==="
echo "HOSTNAME=\$(hostname)"
echo "DATE=\$(date -Is)"

if [ -d ~/DAWN-SRW/.git ]; then
    cd ~/DAWN-SRW
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
else
    rm -rf ~/DAWN-SRW
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" ~/DAWN-SRW
    cd ~/DAWN-SRW
fi

BENCH_ARGS=""
if [ -n "\$CONFIG_REST" ]; then
    BENCH_ARGS="\$CONFIG_REST"
fi
BENCH_ARGS="\$BENCH_ARGS --steps \$STEPS --warmup-steps \$WARMUP_STEPS"
if [ -n "\$MODEL_VERSION" ]; then
    BENCH_ARGS="\$BENCH_ARGS --model-version \$MODEL_VERSION"
fi
if [ "\$ALLOW_MODEL_VERSION_OVERRIDE" = "1" ]; then
    BENCH_ARGS="\$BENCH_ARGS --allow-model-version-override"
fi
if [ "\$XLA_DUMP_ENABLED" = "1" ]; then
    BENCH_XLA="\${XLA_DUMP_BASE}/host_\${TPU_WORKER_INDEX}"
    BENCH_ARGS="\$BENCH_ARGS --xla-dump-dir \$BENCH_XLA"
fi
if [ "\$DUMMY_DATA" = "1" ]; then
    BENCH_ARGS="\$BENCH_ARGS --dummy-data"
fi

export RUN_KIND='benchmark'
export TRAIN_SCRIPT='scripts/benchmark_srw_tpu.py'
export TRAIN_ARGS="\$BENCH_ARGS"
export ENABLE_XLA_DUMP=0
unset XLA_DUMP_DIR

echo "BENCHMARK_SCRIPT=\$TRAIN_SCRIPT"
echo "CONFIG(first)=\$CONFIG"
echo "CONFIG(rest)=\$CONFIG_REST"
echo "BENCHMARK_ARGS=\$TRAIN_ARGS"
if [ "\$XLA_DUMP_ENABLED" = "1" ]; then
    echo "Benchmark XLA dump root: \$XLA_DUMP_BASE"
else
    echo "Benchmark XLA dumps: disabled"
fi

bash scripts/setup_and_run_tpu_pod.sh
EOFCMD

echo "Sending benchmark command to all workers..."
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
declare -a LAUNCH_PIDS=()
declare -a LAUNCH_LOGS=()
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    log_file="launch_srw_benchmark_${TPU_NAME}_${LAUNCH_TS}_worker_${worker}.log"
    LAUNCH_LOGS[$worker]="$log_file"
    REMOTE_CMD="${REMOTE_CMD_TEMPLATE//__TPU_WORKER_INDEX__/$worker}"
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

if [[ "${#FAILED_WORKERS[@]}" -gt 0 ]]; then
    echo "ERROR: benchmark launch failed on worker(s): ${FAILED_WORKERS[*]}." >&2
    exit 1
fi

echo ""
echo "Benchmark launched in tmux session 'train' on every worker."
echo "  Watch:       bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT"
echo "  Main output: ~/train.log on each TPU worker"
echo "  Result mode: console-only benchmark records and comparison table"
if [[ "$XLA_DUMP_ENABLED" = "1" ]]; then
    echo "  XLA dumps:   $XLA_DUMP_BASE"
fi
