#!/usr/bin/env bash
# Follow TPU VM/Pod logs with automatic worker detection.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
REMOTE_LOG="~/train.log"
WORKERS="primary"
TAIL_LINES=160
FOLLOW=1
FILTER_PATTERN=""
MODE="tail"
PRIMARY_DETECT_ATTEMPTS=6
PRIMARY_DETECT_SLEEP=5

SUMMARY_PATTERN='EVAL |PRUNE .*SUMMARY|PRUNE eps=.*(SKIP|SUMMARY)|USAGE (REDUCE|SUMMARY|HOST DONE)|TRACE (START|prompt|SUMMARY)|ABLATION (START|BASE|job|SUMMARY)|REPORT|Epoch .*complete|Training complete|Val loss|Pruned eval|Best val|Final step|ERROR|FAILED|Traceback|RuntimeError|AssertionError|RESOURCE_EXHAUSTED'
ERROR_PATTERN='Traceback|RuntimeError|AssertionError|FAILED|ERROR|RESOURCE_EXHAUSTED|OutOfMemory|SIGABRT|Aborted|Terminating process|unhealthy|CONSUMER_INVALID|PERMISSION_DENIED'
PRIMARY_DETECT_PATTERN='primary=True|host=0/[0-9]+|process_index=0|Host ID: 0|USAGE (REDUCE|SUMMARY)|PRUNE SUMMARY|TRACE SUMMARY|ABLATION SUMMARY|REPORT'

usage() {
    printf '%s\n' \
        "Usage: $0 --tpu NAME [options]" \
        "" \
        "Core:" \
        "  --tpu NAME             TPU VM/Pod name" \
        "  --zone ZONE            Default: $ZONE" \
        "  --project PROJECT      Default: $PROJECT" \
        "  --log PATH             Remote log path. Default: $REMOTE_LOG" \
        "" \
        "Workers:" \
        "  --primary              Auto-detect and follow the JAX primary host (default)" \
        "  --all                  Follow all detected workers, prefixing lines with [wNN]" \
        "  --worker N             Follow one literal gcloud worker index" \
        "  --workers LIST|all     Follow comma-separated workers or all" \
        "" \
        "Output:" \
        "  --summary              Show result/progress/error lines only" \
        "  --errors               Show error lines only" \
        "  --grep PATTERN         Custom remote grep -E pattern" \
        "  --tail N               Initial tail lines. Default: $TAIL_LINES" \
        "  --no-follow            Print tail and exit" \
        "  --status               Snapshot tmux/process/last-log line on all workers and exit" \
        "" \
        "Examples:" \
        "  $0 --tpu spatial-400m --project dawn-486218 --zone us-central2-b" \
        "  $0 --tpu spatial-400m --project dawn-486218 --zone us-central2-b --summary" \
        "  $0 --tpu spatial-400m --project dawn-486218 --zone us-central2-b --all --summary" \
        "  $0 --tpu spatial-400m --project dawn-486218 --zone us-central2-b --status"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --log) REMOTE_LOG="$2"; shift 2 ;;
        --primary) WORKERS="primary"; shift ;;
        --all) WORKERS="all"; shift ;;
        --worker) WORKERS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --summary) FILTER_PATTERN="$SUMMARY_PATTERN"; shift ;;
        --errors) FILTER_PATTERN="$ERROR_PATTERN"; shift ;;
        --grep) FILTER_PATTERN="$2"; shift 2 ;;
        --tail) TAIL_LINES="$2"; shift 2 ;;
        --follow) FOLLOW=1; shift ;;
        --no-follow) FOLLOW=0; shift ;;
        --status) MODE="status"; WORKERS="all"; FOLLOW=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1 (use --help)" >&2; exit 1 ;;
    esac
done

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu NAME is required." >&2
    exit 1
fi
if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --tail must be a non-negative integer." >&2
    exit 1
fi

detect_worker_count() {
    local accelerator_type network_endpoints worker_count accelerator_size accelerator_workers
    accelerator_type="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format="value(acceleratorType)" 2>/dev/null || true)"
    network_endpoints="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --format="value(networkEndpoints[].ipAddress)" 2>/dev/null || true)"
    worker_count="$(printf '%s\n' "$network_endpoints" | awk 'NF {count += NF} END {print count + 0}')"
    accelerator_workers=0
    accelerator_size="${accelerator_type##*-}"
    if [[ "$accelerator_size" =~ ^[0-9]+$ ]]; then
        accelerator_workers=$(( (accelerator_size + 7) / 8 ))
    fi
    if [[ "$accelerator_workers" -gt "$worker_count" ]]; then
        worker_count="$accelerator_workers"
    fi
    if [[ "$worker_count" -le 0 ]]; then
        worker_count=1
    fi
    echo "$worker_count"
}

worker_count="$(detect_worker_count)"

quote_remote() {
    printf '%q' "$1"
}

remote_log_q="$(quote_remote "$REMOTE_LOG")"
filter_q=""
if [[ -n "$FILTER_PATTERN" ]]; then
    filter_q="$(quote_remote "$FILTER_PATTERN")"
fi

build_tail_cmd() {
    local cmd
    if [[ "$FOLLOW" -eq 1 ]]; then
        cmd="tail -n $TAIL_LINES -F $remote_log_q"
    else
        cmd="tail -n $TAIL_LINES $remote_log_q"
    fi
    if [[ -n "$FILTER_PATTERN" ]]; then
        if [[ "$FOLLOW" -eq 1 ]]; then
            cmd="$cmd | grep --line-buffered -E $filter_q"
        else
            cmd="$cmd | grep -E $filter_q || true"
        fi
    fi
    printf '%s' "$cmd"
}

build_status_cmd() {
    printf '%s\n' \
        'echo "HOST=$(hostname) DATE=$(date -Is)"' \
        'echo "TMUX:"' \
        'tmux list-sessions 2>/dev/null || true' \
        'echo "PROCS:"' \
        'pgrep -af "train_jax|train_jax_minimal|analyze_dawn_srw_v4166" | head -n 8 || true' \
        'echo "LAST_LOG:"' \
        "tail -n 5 $remote_log_q 2>/dev/null || echo \"no log at $REMOTE_LOG\""
}

run_worker_command() {
    local worker="$1"
    local command="$2"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --worker="$worker" \
        --command="$command"
}

detect_primary_worker() {
    local attempt worker out cmd pattern_q
    pattern_q="$(quote_remote "$PRIMARY_DETECT_PATTERN")"
    cmd="tail -n 5000 $remote_log_q 2>/dev/null | grep -E $pattern_q | tail -n 1 || true"
    for attempt in $(seq 1 "$PRIMARY_DETECT_ATTEMPTS"); do
        for worker in $(seq 0 $((worker_count - 1))); do
            out="$(run_worker_command "$worker" "$cmd" 2>/dev/null || true)"
            if [[ -n "${out//[[:space:]]/}" ]]; then
                echo "$worker"
                return 0
            fi
        done
        if [[ "$attempt" -lt "$PRIMARY_DETECT_ATTEMPTS" ]]; then
            sleep "$PRIMARY_DETECT_SLEEP"
        fi
    done
    return 1
}

declare -a TARGET_WORKERS=()
PRIMARY_WORKER=""
if [[ "$WORKERS" == "all" || "$WORKERS" == "auto" ]]; then
    for worker in $(seq 0 $((worker_count - 1))); do
        TARGET_WORKERS+=("$worker")
    done
elif [[ "$WORKERS" == "primary" ]]; then
    echo "Detecting JAX primary host from remote logs..."
    PRIMARY_WORKER="$(detect_primary_worker || true)"
    if [[ -n "$PRIMARY_WORKER" ]]; then
        TARGET_WORKERS+=("$PRIMARY_WORKER")
    else
        echo "WARNING: Could not detect primary host yet; falling back to all workers with prefixes." >&2
        for worker in $(seq 0 $((worker_count - 1))); do
            TARGET_WORKERS+=("$worker")
        done
    fi
else
    IFS=',' read -r -a TARGET_WORKERS <<< "$WORKERS"
fi

echo "TPU log watcher"
echo "  TPU:     $TPU_NAME"
echo "  Project: $PROJECT"
echo "  Zone:    $ZONE"
echo "  Log:     $REMOTE_LOG"
echo "  Workers: ${TARGET_WORKERS[*]} (detected=$worker_count)"
if [[ -n "$PRIMARY_WORKER" ]]; then
    echo "  Primary: gcloud worker $PRIMARY_WORKER"
fi
if [[ -n "$FILTER_PATTERN" ]]; then
    echo "  Filter:  enabled"
fi
echo ""

if [[ "$MODE" == "status" ]]; then
    status_cmd="$(build_status_cmd)"
    for worker in "${TARGET_WORKERS[@]}"; do
        echo "===== worker $worker ====="
        run_worker_command "$worker" "$status_cmd" || echo "FAILED worker $worker"
    done
    exit 0
fi

tail_cmd="$(build_tail_cmd)"
declare -a PIDS=()
cleanup() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        kill "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

if [[ "${#TARGET_WORKERS[@]}" -eq 1 ]]; then
    run_worker_command "${TARGET_WORKERS[0]}" "$tail_cmd"
else
    for worker in "${TARGET_WORKERS[@]}"; do
        (
            run_worker_command "$worker" "$tail_cmd" 2>&1 \
                | sed -u "s/^/[w$(printf '%02d' "$worker")] /"
        ) &
        PIDS+=("$!")
    done
    wait
fi
