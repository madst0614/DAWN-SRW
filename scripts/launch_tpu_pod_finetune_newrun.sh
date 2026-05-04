#!/bin/bash
# =============================================================================
# TPU Pod Launcher for fine-tune/new-run
# Force-pulls the requested branch on every worker before launching.
# =============================================================================
set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG=""
GH_TOKEN="${GH_TOKEN:-}"
INIT_FROM=""
RESUME_FROM=""
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --token) GH_TOKEN="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tpu NAME --branch BRANCH --config CONFIG [--init-from GCS_OR_FILE | --resume-from RUN_OR_CKPT]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TPU_NAME" ]]; then echo "ERROR: --tpu is required" >&2; exit 1; fi
if [[ -z "$CONFIG" ]]; then echo "ERROR: --config is required" >&2; exit 1; fi
if [[ -n "$INIT_FROM" && -n "$RESUME_FROM" ]]; then
  echo "ERROR: use only one of --init-from or --resume-from" >&2
  exit 1
fi

if [[ -n "$GH_TOKEN" ]]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
  REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

echo "============================================"
echo "Launching TPU Pod fine-tune/new-run"
echo "  TPU:       $TPU_NAME"
echo "  Zone:      $ZONE"
echo "  Project:   $PROJECT"
echo "  Branch:    $BRANCH"
echo "  Config:    $CONFIG"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Resume:    ${RESUME_FROM:-<none>}"
echo "============================================"

gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
GH_TOKEN='${GH_TOKEN}'
INIT_FROM='${INIT_FROM}'
RESUME_FROM='${RESUME_FROM}'
TRAIN_ARGS='${TRAIN_ARGS}'
export BRANCH CONFIG GH_TOKEN INIT_FROM RESUME_FROM TRAIN_ARGS
WORK_DIR="\$HOME/dawn-spatial"

echo "[remote] force-sync repo branch: \$BRANCH"
if [ -d "\$WORK_DIR/.git" ]; then
  cd "\$WORK_DIR"
  git remote set-url origin "\$REPO_URL" || true
  git fetch origin "\$BRANCH" --depth 1
  git checkout -B "\$BRANCH" FETCH_HEAD
  git reset --hard FETCH_HEAD
  git clean -fd
else
  rm -rf "\$WORK_DIR"
  git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
  cd "\$WORK_DIR"
fi

echo "[remote] HEAD: \$(git rev-parse --short HEAD) \$(git log -1 --oneline)"

# Hard guard: this launcher must never call the broken init-newrun path.
if grep -R "train_jax_init_newrun" scripts/launch_tpu_pod_finetune_newrun.sh scripts/setup_and_run_tpu_pod_finetune_newrun.sh 2>/dev/null; then
  echo "ERROR: finetune launcher/setup still references train_jax_init_newrun.py" >&2
  exit 2
fi

test -f scripts/train_jax_finetune_newrun.py

echo "[remote] verified finetune script exists; launching setup"
bash scripts/setup_and_run_tpu_pod_finetune_newrun.sh
EOFCMD

echo "Sending force-sync + launch command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --worker=all \
  --command="$REMOTE_CMD" \
  2>&1 | tee "launch_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch command sent. Check worker 0:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='pgrep -af train_jax; tail -f ~/train.log'"
