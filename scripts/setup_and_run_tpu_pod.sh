#!/bin/bash
# =============================================================================
# TPU Pod Setup + Training Script (runs on each worker)
# =============================================================================
# Expects BRANCH and CONFIG passed as environment variables from the launcher.
#
# Usage (direct):
#   BRANCH=main CONFIG=configs/train_config_v17_1_tpu_400M_c4_5B_v4_64.yaml \
#     bash scripts/setup_and_run_tpu_pod.sh
#
# Usually invoked via launch_tpu_pod.sh which sets env vars automatically.
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/madst0614/DAWN.git"
BRANCH="${BRANCH:?ERROR: BRANCH env var not set}"
CONFIG="${CONFIG:?ERROR: CONFIG env var not set}"
WORK_DIR="$HOME/dawn"

echo "============================================"
echo "Host $(hostname) — Setting up TPU Pod training"
echo "  Branch: $BRANCH"
echo "  Config: $CONFIG"
echo "============================================"

# 1. Install dependencies (all workers)
echo "[1/4] Installing dependencies..."
pip install --upgrade pip -q
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install flax optax numpy pyyaml gcsfs -q

# 2. Deploy code via git clone
echo "[2/4] Cloning repo (branch: $BRANCH)..."
cd "$HOME"
if [ -d dawn ]; then
    rm -rf dawn
fi
git clone -b "$BRANCH" --single-branch --depth 1 "$REPO_URL" dawn
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
print('TPU setup verified OK')
"

# 4. Launch training
echo "[4/4] Starting training..."
echo "  Config: $CONFIG"
echo "  Host: $(hostname)"
echo "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

python scripts/train_jax.py --config "$CONFIG"

echo "Training complete on host $(hostname)"
