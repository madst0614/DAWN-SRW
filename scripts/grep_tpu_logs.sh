#!/bin/bash
# Collect first-failure diagnostics from TPU worker train logs.

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT=""
WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [--zone ZONE] [--project PROJECT] [--workers N]"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

if [ -z "$TPU_NAME" ]; then
    echo "ERROR: --tpu NAME is required." >&2
    exit 1
fi

PROJECT_ARG=()
if [ -n "$PROJECT" ]; then
    PROJECT_ARG=(--project="$PROJECT")
fi

REMOTE_GREP='grep -nE "minimal-stage|entering barrier|passed barrier|FAILED barrier|task:7|unhealthy|Traceback|RuntimeError|AssertionError|RESOURCE_EXHAUSTED|Terminating process|SIGABRT|Aborted" ~/train.log | tail -n 120'

for worker in $(seq 0 $((WORKERS - 1))); do
    echo "===== worker $worker ====="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        "${PROJECT_ARG[@]}" \
        --worker="$worker" \
        --command="$REMOTE_GREP" \
        || echo "FAILED worker $worker"
done
