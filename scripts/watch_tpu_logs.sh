#!/usr/bin/env bash
# Follow TPU VM/Pod logs with automatic worker detection.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
readonly REMOTE_USER="madst0614"
readonly REMOTE_LOG="~/train.log"
readonly PANE_TARGET="train"
SOURCE="auto"
WORKERS="primary"
TAIL_LINES=160
FOLLOW=1
FILTER_PATTERN=""
MODE="tail"
PRIMARY_DETECT_ATTEMPTS=6
PRIMARY_DETECT_SLEEP=5
PANE_COLS="${WATCH_TPU_LOG_COLS:-240}"
PANE_ROWS="${WATCH_TPU_LOG_ROWS:-60}"

SUMMARY_PATTERN='TRAIN_ANALYSIS_POOL|EVAL |Epoch .*complete|Training complete|Val loss|Best val|Final step|ERROR|FAILED|Traceback|RuntimeError|AssertionError|RESOURCE_EXHAUSTED'
ERROR_PATTERN='Traceback|RuntimeError|AssertionError|FAILED|ERROR|RESOURCE_EXHAUSTED|OutOfMemory|SIGABRT|Aborted|Terminating process|unhealthy|CONSUMER_INVALID|PERMISSION_DENIED'
PRIMARY_DETECT_PATTERN='primary=True|host=0/[0-9]+|process_index=0|Host ID: 0|TRAIN_ANALYSIS_POOL (load|item=.*status=|COMPLETE)'

usage() {
    printf '%s\n' \
        "Usage: $0 --tpu NAME [options]" \
        "" \
        "Core:" \
        "  --tpu NAME             TPU VM/Pod name" \
        "  --zone ZONE            Default: $ZONE" \
        "  --project PROJECT      Default: $PROJECT" \
        "" \
        "Workers:" \
        "  --primary              Auto-detect and follow the JAX primary host (default)" \
        "  --all                  Follow all detected workers, prefixing lines with [wNN]" \
        "  --worker N             Follow one literal gcloud worker index" \
        "  --workers LIST|all     Follow comma-separated workers or all" \
        "" \
        "Output:" \
        "  --file                 Follow the remote log file" \
        "  --pane                 Follow tmux pane output without requiring ~/train.log" \
        "  --attach               Attach to the tmux session on the detected primary worker" \
        "  --summary              Show result/progress/error lines only" \
        "  --errors               Show error lines only" \
        "  --grep PATTERN         Custom remote grep -E pattern" \
        "  --tail N               Initial tail lines. Default: $TAIL_LINES" \
        "  --cols N               Width for tmux pane capture. Default: $PANE_COLS" \
        "  --rows N               Height for tmux pane capture. Default: $PANE_ROWS" \
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
        --file) SOURCE="file"; shift ;;
        --pane|--screen) SOURCE="pane"; shift ;;
        --attach) MODE="attach"; SOURCE="pane"; FOLLOW=0; shift ;;
        --primary) WORKERS="primary"; shift ;;
        --all) WORKERS="all"; shift ;;
        --worker) WORKERS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --summary) FILTER_PATTERN="$SUMMARY_PATTERN"; shift ;;
        --errors) FILTER_PATTERN="$ERROR_PATTERN"; shift ;;
        --grep) FILTER_PATTERN="$2"; shift 2 ;;
        --tail) TAIL_LINES="$2"; shift 2 ;;
        --cols) PANE_COLS="$2"; shift 2 ;;
        --rows) PANE_ROWS="$2"; shift 2 ;;
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
if ! [[ "$PANE_COLS" =~ ^[0-9]+$ && "$PANE_COLS" -gt 0 ]]; then
    echo "ERROR: --cols must be a positive integer." >&2
    exit 1
fi
if ! [[ "$PANE_ROWS" =~ ^[0-9]+$ && "$PANE_ROWS" -gt 0 ]]; then
    echo "ERROR: --rows must be a positive integer." >&2
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

quote_remote_path() {
    local path="$1"
    if [[ "$path" == "~" ]]; then
        printf '$HOME'
    elif [[ "$path" == "~/"* ]]; then
        printf '$HOME/%s' "$(quote_remote "${path:2}")"
    elif [[ "$path" == "\$HOME" ]]; then
        printf '$HOME'
    elif [[ "$path" == "\$HOME/"* ]]; then
        printf '$HOME/%s' "$(quote_remote "${path:6}")"
    else
        quote_remote "$path"
    fi
}

remote_log_q="$(quote_remote_path "$REMOTE_LOG")"
pane_target_q="$(quote_remote "$PANE_TARGET")"
filter_q=""
if [[ -n "$FILTER_PATTERN" ]]; then
    filter_q="$(quote_remote "$FILTER_PATTERN")"
fi

filter_range_cmd() {
    if [[ -n "$FILTER_PATTERN" ]]; then
        printf "tr '\\r' '\\n' | sed 's/[[:blank:]]*$//' | grep -E %s || true" "$filter_q"
    else
        printf "tr '\\r' '\\n' | sed 's/[[:blank:]]*$//'"
    fi
}

resize_pane_cmd() {
    printf 'tmux resize-window -t %s -x %s -y %s 2>/dev/null || true' \
        "$pane_target_q" "$PANE_COLS" "$PANE_ROWS"
}

capture_pane_cmd() {
    local resize_cmd
    resize_cmd="$(resize_pane_cmd)"
    printf '%s; tmux capture-pane -J -t %s -pS - 2>/dev/null || tmux capture-pane -t %s -pS - 2>/dev/null' \
        "$resize_cmd" \
        "$pane_target_q" "$pane_target_q"
}

build_file_tail_cmd() {
    local cmd
    if [[ "$FOLLOW" -eq 1 ]]; then
        cmd="tail -n $TAIL_LINES -F $remote_log_q"
    else
        cmd="tail -n $TAIL_LINES $remote_log_q"
    fi
    cmd="$cmd | tr '\\r' '\\n' | sed 's/[[:blank:]]*$//'"
    if [[ -n "$FILTER_PATTERN" ]]; then
        if [[ "$FOLLOW" -eq 1 ]]; then
            cmd="$cmd | grep --line-buffered -E $filter_q"
        else
            cmd="$cmd | grep -E $filter_q || true"
        fi
    fi
    printf '%s' "$cmd"
}

build_pane_tail_cmd() {
    local emit_filter pane_capture
    emit_filter="$(filter_range_cmd)"
    pane_capture="$(capture_pane_cmd)"
    if [[ "$FOLLOW" -eq 0 ]]; then
        printf '{ %s; } | tail -n %s | %s' \
            "$pane_capture" "$TAIL_LINES" "$emit_filter"
        return
    fi
    printf '%s' "\
tmp=\"/tmp/watch_tpu_pane_${PANE_TARGET//[^A-Za-z0-9_]/_}.\${USER:-user}.log\"
last=0
echo \"[watch] source=tmux-pane target=$PANE_TARGET tail=$TAIL_LINES\"
while true; do
  if ! tmux has-session -t $pane_target_q 2>/dev/null; then
    echo \"[watch] waiting for tmux target $PANE_TARGET\"
    sleep 2
    continue
  fi
  { $pane_capture; } > \"\$tmp\" || { sleep 2; continue; }
  total=\$(wc -l < \"\$tmp\" | tr -d ' ')
  if [ \"\${total:-0}\" -le 0 ]; then
    sleep 2
    continue
  fi
  if [ \"\$last\" -eq 0 ]; then
    start=\$((total - $TAIL_LINES + 1))
    [ \"\$start\" -lt 1 ] && start=1
    sed -n \"\${start},\${total}p\" \"\$tmp\" | $emit_filter
  elif [ \"\$total\" -gt \"\$last\" ]; then
    start=\$((last + 1))
    sed -n \"\${start},\${total}p\" \"\$tmp\" | $emit_filter
  elif [ \"\$total\" -lt \"\$last\" ]; then
    echo \"[watch] pane history reset; showing last $TAIL_LINES lines\"
    start=\$((total - $TAIL_LINES + 1))
    [ \"\$start\" -lt 1 ] && start=1
    sed -n \"\${start},\${total}p\" \"\$tmp\" | $emit_filter
  fi
  last=\"\$total\"
  sleep 2
done"
}

build_auto_tail_cmd() {
    local file_cmd
    file_cmd="$(build_file_tail_cmd)"
    printf 'echo "[watch] source=file log=%s"; echo "[watch] use --pane only for tmux screen capture"; %s' \
        "$REMOTE_LOG" "$file_cmd"
}

build_tail_cmd() {
    case "$SOURCE" in
        file) build_file_tail_cmd ;;
        pane) build_pane_tail_cmd ;;
        auto) build_auto_tail_cmd ;;
        *) echo "ERROR: unknown source $SOURCE" >&2; exit 1 ;;
    esac
}

build_attach_cmd() {
    local resize_cmd
    resize_cmd="$(resize_pane_cmd)"
    printf '%s; tmux attach -t %s' "$resize_cmd" "$pane_target_q"
}

build_status_cmd() {
    local pane_capture
    pane_capture="$(capture_pane_cmd)"
    printf '%s\n' \
        'echo "HOST=$(hostname) DATE=$(date -Is)"' \
        'echo "TMUX:"' \
        'tmux list-sessions 2>/dev/null || true' \
        'echo "PROCS:"' \
        'pgrep -af "train_jax|train_jax_minimal|analyze_train_analysis_pool" | head -n 8 || true' \
        'echo "LAST_LOG:"' \
        "tail -n 5 $remote_log_q 2>/dev/null || echo \"no log at $REMOTE_LOG\"" \
        'echo "PANE:"' \
        "{ $pane_capture; } | tail -n 20 || echo \"no tmux target $PANE_TARGET\""
}

run_worker_command() {
    local worker="$1"
    local command="$2"
    gcloud compute tpus tpu-vm ssh "$REMOTE_USER@$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --worker="$worker" \
        --command="$command"
}

detect_primary_worker() {
    local attempt worker out cmd pattern_q pane_capture
    pattern_q="$(quote_remote "$PRIMARY_DETECT_PATTERN")"
    pane_capture="$(capture_pane_cmd)"
    cmd="{ if [ -r $remote_log_q ]; then tail -n 5000 $remote_log_q | tr '\\r' '\\n'; fi; { $pane_capture; } | tr '\\r' '\\n' || true; } | grep -E $pattern_q | tail -n 1 || true"
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
echo "  Source:  $SOURCE"
echo "  Tmux:    $PANE_TARGET"
echo "  Pane:    ${PANE_COLS}x${PANE_ROWS}"
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

if [[ "$MODE" == "attach" ]]; then
    if [[ "${#TARGET_WORKERS[@]}" -ne 1 ]]; then
        echo "ERROR: --attach needs a single detected worker. Use --worker N --attach or retry after primary is detectable." >&2
        exit 1
    fi
    run_worker_command "${TARGET_WORKERS[0]}" "$(build_attach_cmd)"
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
