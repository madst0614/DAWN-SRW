#!/bin/bash
# =============================================================================
# v4-64 TPU Pod (8 hosts, 64 chips) Setup + Training Script
# =============================================================================
# This script runs on EACH worker (host) of the TPU Pod.
# Usage:
#   gcloud compute tpus tpu-vm ssh dawn-400m-v4-64 \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="bash ~/dawn/scripts/setup_and_run_v4_64.sh"
# =============================================================================

set -euo pipefail

TPU_NAME="dawn-400m-v4-64"
ZONE="us-central2-b"
PROJECT="dawn-486218"
REPO_GCS="gs://dawn-tpu-data-c4/code/dawn.tar.gz"
CONFIG="configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml"
WORK_DIR="$HOME/dawn"

echo "============================================"
echo "Host $(hostname) — Setting up v4-64 training"
echo "============================================"

# 1. Install dependencies (all workers)
echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

# 2. Deploy code from GCS (all workers need identical code)
echo "[2/4] Deploying code from GCS..."
cd "$HOME"
if [ -f dawn.tar.gz ]; then
    rm dawn.tar.gz
fi
gsutil cp "$REPO_GCS" .
if [ -d dawn ]; then
    rm -rf dawn
fi
tar xzf dawn.tar.gz
cd dawn

# 3. Verify JAX sees TPU devices
echo "[3/4] Verifying JAX TPU setup..."
python -c "
import jax
print(f'Host: {jax.process_index()}/{jax.process_count()}')
print(f'Local devices: {jax.local_device_count()}')
print(f'Total devices: {jax.device_count()}')
print(f'Backend: {jax.default_backend()}')
assert jax.default_backend() == 'tpu', 'Not running on TPU!'
assert jax.local_device_count() == 8, f'Expected 8 local devices, got {jax.local_device_count()}'
print('TPU setup verified OK')
"

# 4. Launch training
echo "[4/4] Starting training..."
echo "  Config: $CONFIG"
echo "  Host: $(hostname)"
echo "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

python scripts/train_jax.py --config "$CONFIG"

echo "Training complete on host $(hostname)"
