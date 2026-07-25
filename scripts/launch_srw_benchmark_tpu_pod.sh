#!/bin/bash
# =============================================================================
# DAWN-SRW TPU Pod Benchmark Launcher
# =============================================================================
# Launches scripts/benchmark_srw_tpu.py on every TPU host. Pass --config more
# than once to run configs sequentially and print the comparison in ~/train.log.
# Config selection is explicit by default; use --auto-compare to add the
# matching standard v4166/v4168 400M config automatically.
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
REMOTE_USER="madst0614"
BRANCH="main"
CONFIGS=()
MODEL_VERSION=""
STEPS="20"
WARMUP_STEPS="5"
FORWARD_PROFILE_STEPS="1"
MODULE_PROFILE_STEPS="1"
STEPS_SET="0"
WARMUP_STEPS_SET="0"
FORWARD_PROFILE_STEPS_SET="0"
MODULE_PROFILE_STEPS_SET="0"
FAST_ONLY="0"
BACKWARD_PROFILE="0"
DETAILED_PROFILE_LAYERS="0"
OUTPUT_DIR=""
XLA_DUMP_ENABLED="1"
XLA_DUMP_BASE=""
GH_TOKEN=""
DUMMY_DATA="0"
ALLOW_MODEL_VERSION_OVERRIDE="0"
AUTO_COMPARE="0"

usage() {
    cat <<EOF
Usage: $0 --tpu NAME --branch BRANCH --config CONFIG [--config CONFIG2 ...] [options]

Options:
  --zone ZONE
  --project PROJECT
  --steps N
  --warmup-steps N
  --forward-profile-steps N     Forward-only profile steps per config (default: 1)
  --module-profile-steps N      Split-module profile steps per config (default: 1)
  --fast, --fast-only           Run quick real-data detailed forward diagnosis
  --backward-profile            With --fast, include v4174 QKV/RST backward profiles
  --detailed-profile-layers N   Limit v4174 detailed profiling to first N layers
  --model-version VERSION        Optional expected version check
  --allow-model-version-override Allow --model-version to override config
  --output-dir DIR               Remote artifact root for XLA dumps only
  --xla-dump [DIR]               Enable XLA dumps (default: enabled)
  --no-xla-dump                  Disable XLA dumps
  --dummy-data                   Explicit synthetic-data smoke test
  --auto-compare                 Auto-add the matching standard v4166/v4168 config
  --no-auto-compare              Keep only explicitly supplied configs (default)
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
        --steps) STEPS="$2"; STEPS_SET="1"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; WARMUP_STEPS_SET="1"; shift 2 ;;
        --forward-profile-steps) FORWARD_PROFILE_STEPS="$2"; FORWARD_PROFILE_STEPS_SET="1"; shift 2 ;;
        --module-profile-steps) MODULE_PROFILE_STEPS="$2"; MODULE_PROFILE_STEPS_SET="1"; shift 2 ;;
        --fast|--fast-only) FAST_ONLY="1"; shift ;;
        --backward-profile) BACKWARD_PROFILE="1"; shift ;;
        --detailed-profile-layers) DETAILED_PROFILE_LAYERS="$2"; shift 2 ;;
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
        --auto-compare) AUTO_COMPARE="1"; shift ;;
        --no-auto-compare) AUTO_COMPARE="0"; shift ;;
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

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu must explicitly name an existing TPU resource." >&2
    exit 1
fi
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "ERROR: at least one --config is required." >&2
    exit 1
fi
if [[ -n "$MODEL_VERSION" &&
      "$MODEL_VERSION" != "spatial-r1-v4.1.6.6" &&
      "$MODEL_VERSION" != "spatial-r1-v4.1.6.8" &&
      "$MODEL_VERSION" != "spatial-r1-v4.1.7.4" ]]; then
    echo "ERROR: unsupported --model-version: $MODEL_VERSION." >&2
    exit 1
fi
if ! is_nonnegative_int "$STEPS"; then
    echo "ERROR: --steps must be an integer >= 0." >&2
    exit 1
fi
if [[ "$FAST_ONLY" != "1" && "$STEPS" -le 0 ]]; then
    echo "ERROR: --steps must be an integer > 0 unless --fast is set." >&2
    exit 1
fi
if ! is_nonnegative_int "$WARMUP_STEPS"; then
    echo "ERROR: --warmup-steps must be an integer >= 0." >&2
    exit 1
fi
if ! is_nonnegative_int "$FORWARD_PROFILE_STEPS"; then
    echo "ERROR: --forward-profile-steps must be an integer >= 0." >&2
    exit 1
fi
if ! is_nonnegative_int "$MODULE_PROFILE_STEPS"; then
    echo "ERROR: --module-profile-steps must be an integer >= 0." >&2
    exit 1
fi
if ! is_nonnegative_int "$DETAILED_PROFILE_LAYERS"; then
    echo "ERROR: --detailed-profile-layers must be an integer >= 0." >&2
    exit 1
fi
if [[ "$BACKWARD_PROFILE" = "1" && "$FAST_ONLY" != "1" ]]; then
    echo "ERROR: --backward-profile requires --fast." >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="benchmark_runs/srw_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$XLA_DUMP_BASE" ]]; then
    XLA_DUMP_BASE="${OUTPUT_DIR}/xla"
fi

if [[ "$AUTO_COMPARE" = "1" && "${#CONFIGS[@]}" -eq 1 ]]; then
    ONLY_CONFIG="${CONFIGS[0]}"
    AUTO_CONFIG=""
    case "$ONLY_CONFIG" in
        configs/train_config_v4166_400M_c4_40B_v4_64.yaml)
            AUTO_CONFIG="configs/train_config_v4168_400M_c4_40B_v4_64_block_sparse.yaml"
            CONFIGS+=("$AUTO_CONFIG")
            ;;
        configs/train_config_v4168_400M_c4_40B_v4_64_block_sparse.yaml)
            AUTO_CONFIG="configs/train_config_v4166_400M_c4_40B_v4_64.yaml"
            CONFIGS=("$AUTO_CONFIG" "$ONLY_CONFIG")
            ;;
    esac
    if [[ -n "$AUTO_CONFIG" ]]; then
        echo "Auto-compare enabled: added $AUTO_CONFIG"
    fi
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
if [[ "$FAST_ONLY" = "1" ]]; then
    echo "  Mode:           quick detailed v4174 diagnosis"
    echo "  Train steps:    skipped"
    echo "  Forward profile:1"
    echo "  Backward profile:$BACKWARD_PROFILE"
    echo "  Detailed layers:$DETAILED_PROFILE_LAYERS (0=all)"
    echo "  Module profile: 0"
else
    echo "  Mode:           train benchmark + profiles"
    echo "  Steps:          $STEPS"
    echo "  Warmup steps:   $WARMUP_STEPS"
    echo "  Forward profile:$FORWARD_PROFILE_STEPS"
    echo "  Module profile: $MODULE_PROFILE_STEPS"
fi
echo "  Fast only:      $FAST_ONLY"
echo "  XLA dumps:      $XLA_DUMP_ENABLED ($XLA_DUMP_BASE)"
echo "  Dummy data:     $DUMMY_DATA"
echo "  Auto compare:   $AUTO_COMPARE"
echo "  Result files:   ${OUTPUT_DIR}/benchmark_metrics_host_<worker>.jsonl"
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
    gcloud compute tpus tpu-vm ssh "$REMOTE_USER@$TPU_NAME" \
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
sleep 3
ACCEL_HOLDERS="$(sudo lsof -t /dev/accel* 2>/dev/null | sort -u || true)"
if [ -n "$ACCEL_HOLDERS" ]; then
    echo "ERROR: TPU accelerator holder remains after cleanup:" >&2
    echo "$ACCEL_HOLDERS" >&2
    exit 1
fi
sudo rm -f /tmp/libtpu_lockfile
EOFCLEANUP

echo "Cleaning old benchmark processes on every worker..."
cleanup_failed=0
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    echo "  Cleaning worker $worker..."
    if ! run_worker_command "$worker" "$CLEANUP_CMD"; then
        echo "ERROR: worker $worker cleanup failed." >&2
        cleanup_failed=1
    fi
done
if [ "$cleanup_failed" -ne 0 ]; then
    echo "ERROR: cleanup verification failed. Aborting benchmark launch." >&2
    exit 1
fi

REPO_URL_Q="$(shell_quote "$REPO_URL")"
BRANCH_Q="$(shell_quote "$BRANCH")"
CONFIG_FIRST_Q="$(shell_quote "$CONFIG_FIRST")"
CONFIG_REST_Q="$(shell_quote "$CONFIG_REST_ARGS")"
MODEL_VERSION_Q="$(shell_quote "$MODEL_VERSION")"
STEPS_Q="$(shell_quote "$STEPS")"
WARMUP_STEPS_Q="$(shell_quote "$WARMUP_STEPS")"
FORWARD_PROFILE_STEPS_Q="$(shell_quote "$FORWARD_PROFILE_STEPS")"
MODULE_PROFILE_STEPS_Q="$(shell_quote "$MODULE_PROFILE_STEPS")"
STEPS_SET_Q="$(shell_quote "$STEPS_SET")"
WARMUP_STEPS_SET_Q="$(shell_quote "$WARMUP_STEPS_SET")"
FORWARD_PROFILE_STEPS_SET_Q="$(shell_quote "$FORWARD_PROFILE_STEPS_SET")"
MODULE_PROFILE_STEPS_SET_Q="$(shell_quote "$MODULE_PROFILE_STEPS_SET")"
FAST_ONLY_Q="$(shell_quote "$FAST_ONLY")"
BACKWARD_PROFILE_Q="$(shell_quote "$BACKWARD_PROFILE")"
DETAILED_PROFILE_LAYERS_Q="$(shell_quote "$DETAILED_PROFILE_LAYERS")"
XLA_DUMP_ENABLED_Q="$(shell_quote "$XLA_DUMP_ENABLED")"
XLA_DUMP_BASE_Q="$(shell_quote "$XLA_DUMP_BASE")"
OUTPUT_DIR_Q="$(shell_quote "$OUTPUT_DIR")"
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
FORWARD_PROFILE_STEPS=${FORWARD_PROFILE_STEPS_Q}
MODULE_PROFILE_STEPS=${MODULE_PROFILE_STEPS_Q}
STEPS_SET=${STEPS_SET_Q}
WARMUP_STEPS_SET=${WARMUP_STEPS_SET_Q}
FORWARD_PROFILE_STEPS_SET=${FORWARD_PROFILE_STEPS_SET_Q}
MODULE_PROFILE_STEPS_SET=${MODULE_PROFILE_STEPS_SET_Q}
FAST_ONLY=${FAST_ONLY_Q}
BACKWARD_PROFILE=${BACKWARD_PROFILE_Q}
DETAILED_PROFILE_LAYERS=${DETAILED_PROFILE_LAYERS_Q}
XLA_DUMP_ENABLED=${XLA_DUMP_ENABLED_Q}
XLA_DUMP_BASE=${XLA_DUMP_BASE_Q}
OUTPUT_DIR=${OUTPUT_DIR_Q}
DUMMY_DATA=${DUMMY_DATA_Q}
ALLOW_MODEL_VERSION_OVERRIDE=${ALLOW_MODEL_VERSION_OVERRIDE_Q}
GH_TOKEN=${GH_TOKEN_Q}
export TPU_WORKER_INDEX BRANCH CONFIG GH_TOKEN

tmux kill-session -t train 2>/dev/null || true
: > "\$HOME/train.log"
export TRAIN_LOG_INITIALIZED=1
exec > >(tee -a "\$HOME/train.log") 2>&1

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
if [ "\$FAST_ONLY" = "1" ]; then
    BENCH_ARGS="\$BENCH_ARGS --fast"
    if [ "\$BACKWARD_PROFILE" = "1" ]; then
        BENCH_ARGS="\$BENCH_ARGS --backward-profile"
    fi
    BENCH_ARGS="\$BENCH_ARGS --detailed-profile-layers \$DETAILED_PROFILE_LAYERS"
    if [ "\$STEPS_SET" = "1" ]; then
        BENCH_ARGS="\$BENCH_ARGS --steps \$STEPS"
    fi
    if [ "\$WARMUP_STEPS_SET" = "1" ]; then
        BENCH_ARGS="\$BENCH_ARGS --warmup-steps \$WARMUP_STEPS"
    fi
    if [ "\$FORWARD_PROFILE_STEPS_SET" = "1" ]; then
        BENCH_ARGS="\$BENCH_ARGS --forward-profile-steps \$FORWARD_PROFILE_STEPS"
    fi
    if [ "\$MODULE_PROFILE_STEPS_SET" = "1" ]; then
        BENCH_ARGS="\$BENCH_ARGS --module-profile-steps \$MODULE_PROFILE_STEPS"
    fi
else
    BENCH_ARGS="\$BENCH_ARGS --steps \$STEPS --warmup-steps \$WARMUP_STEPS"
    BENCH_ARGS="\$BENCH_ARGS --forward-profile-steps \$FORWARD_PROFILE_STEPS"
    BENCH_ARGS="\$BENCH_ARGS --module-profile-steps \$MODULE_PROFILE_STEPS"
fi
BENCH_METRICS_JSONL="\${OUTPUT_DIR}/benchmark_metrics_host_\${TPU_WORKER_INDEX}.jsonl"
BENCH_ARGS="\$BENCH_ARGS --metrics-jsonl \$BENCH_METRICS_JSONL"
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
echo "Benchmark metrics JSONL: \$BENCH_METRICS_JSONL"
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
echo "  Metrics:     ${OUTPUT_DIR}/benchmark_metrics_host_<worker>.jsonl"
if [[ "$XLA_DUMP_ENABLED" = "1" ]]; then
    echo "  XLA dumps:   $XLA_DUMP_BASE"
fi
