#!/bin/bash
# =============================================================================
# TPU Pod Launcher — run from local machine or Cloud Shell
# =============================================================================
# Sends setup_and_run_tpu_pod.sh to all workers with the specified branch/config.
#
# Usage:
#   bash scripts/launch_tpu_pod.sh --tpu dawn-400m-v4-64 --branch main --config configs/v4_64.yaml
#   bash scripts/launch_tpu_pod.sh  # uses defaults (v4-64 settings)
#
# Prerequisites:
#   1. TPU VM created:
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

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu)      TPU_NAME="$2"; shift 2 ;;
        --zone)     ZONE="$2";     shift 2 ;;
        --project)  PROJECT="$2";  shift 2 ;;
        --branch)   BRANCH="$2";   shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--tpu NAME] [--zone ZONE] [--project PROJECT] [--branch BRANCH] [--config CONFIG]"
            echo ""
            echo "  --tpu      TPU VM name         (default: $TPU_NAME)"
            echo "  --zone     GCP zone            (default: $ZONE)"
            echo "  --project  GCP project          (default: $PROJECT)"
            echo "  --branch   Git branch to clone  (default: $BRANCH)"
            echo "  --config   Training config YAML (default: $CONFIG)"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Launching TPU Pod training"
echo "  TPU:     $TPU_NAME"
echo "  Zone:    $ZONE"
echo "  Project: $PROJECT"
echo "  Branch:  $BRANCH"
echo "  Config:  $CONFIG"
echo "============================================"

# Check TPU status
echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)"

# Send command to all workers, passing BRANCH and CONFIG as env vars
echo "Sending setup+training command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="BRANCH='${BRANCH}' CONFIG='${CONFIG}' bash ~/dawn/scripts/setup_and_run_tpu_pod.sh" \
    2>&1 | tee "launch_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Launch complete."
