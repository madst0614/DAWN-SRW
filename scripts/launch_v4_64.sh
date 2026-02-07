#!/bin/bash
# =============================================================================
# v4-64 TPU Pod Launcher — run from local machine or Cloud Shell
# =============================================================================
# Sends setup_and_run_v4_64.sh to all 8 workers simultaneously.
#
# Prerequisites:
#   1. TPU VM created:
#      gcloud compute tpus tpu-vm create dawn-400m-v4-64 \
#        --zone=us-central2-b \
#        --accelerator-type=v4-64 \
#        --version=tpu-vm-v4-base \
#        --spot
#
#   2. Code tarball uploaded to GCS:
#      tar czf dawn.tar.gz dawn/
#      gsutil cp dawn.tar.gz gs://dawn-tpu-data-c4/code/dawn.tar.gz
#
# Usage:
#   bash scripts/launch_v4_64.sh
# =============================================================================

set -euo pipefail

TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"

echo "============================================"
echo "Launching v4-64 TPU Pod training"
echo "  TPU: $TPU_NAME"
echo "  Zone: $ZONE"
echo "  Project: $PROJECT"
echo "============================================"

# Check TPU status
echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)"

# Send command to all workers
echo "Sending setup+training command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="bash ~/dawn/scripts/setup_and_run_v4_64.sh" \
    2>&1 | tee "launch_v4_64_$(date +%Y%m%d_%H%M%S).log"

echo "Launch complete."
