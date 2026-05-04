#!/bin/bash
set -euo pipefail

TPU_NAME="dawn-400M"
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
CONFIG=""
INIT_FROM=""
RESUME_FROM=""
GH_TOKEN=""
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --init-from) INIT_FROM="$2"; shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --from-scratch) TRAIN_ARGS="$TRAIN_ARGS --from-scratch"; shift ;;
        --batch-size) TRAIN_ARGS="$TRAIN_ARGS --batch-size $2"; shift 2 ;;
        --lr) TRAIN_ARGS="$TRAIN_ARGS --lr $2"; shift 2 ;;
        --epochs) TRAIN_ARGS="$TRAIN_ARGS --epochs $2"; shift 2 ;;
        --debug) TRAIN_ARGS="$TRAIN_ARGS --debug"; shift ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME --branch BRANCH --config CONFIG (--init-from CKPT_OR_RUN | --resume-from RUN)"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required" >&2
    exit 1
fi
if [ -n "$INIT_FROM" ] && [ -n "$RESUME_FROM" ]; then
    echo "ERROR: --init-from and --resume-from are mutually exclusive" >&2
    exit 1
fi
if [ -z "$INIT_FROM" ] && [ -z "$RESUME_FROM" ]; then
    echo "ERROR: provide --init-from for a new fine-tune run, or --resume-from to resume a fine-tune run" >&2
    exit 1
fi

if [ -n "$GH_TOKEN" ]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
    REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi

echo "============================================"
echo "Launching TPU Pod finetune/new-run"
echo "  TPU:     $TPU_NAME"
echo "  Zone:    $ZONE"
echo "  Project: $PROJECT"
echo "  Branch:  $BRANCH"
echo "  Config:  $CONFIG"
if [ -n "$INIT_FROM" ]; then echo "  Init:    $INIT_FROM"; fi
if [ -n "$RESUME_FROM" ]; then echo "  Resume:  $RESUME_FROM"; fi
if [ -n "$TRAIN_ARGS" ]; then echo "  Args:    $TRAIN_ARGS"; fi
echo "============================================"

gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -e
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CONFIG='${CONFIG}'
GH_TOKEN='${GH_TOKEN}'
INIT_FROM='${INIT_FROM}'
RESUME_FROM='${RESUME_FROM}'
TRAIN_ARGS='${TRAIN_ARGS}'
export BRANCH CONFIG GH_TOKEN INIT_FROM RESUME_FROM TRAIN_ARGS

if [ -d ~/dawn-spatial/.git ]; then
    cd ~/dawn-spatial
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
else
    rm -rf ~/dawn-spatial
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" ~/dawn-spatial
fi

cd ~/dawn-spatial
bash scripts/setup_and_run_tpu_pod_finetune_newrun.sh
EOFCMD

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="$REMOTE_CMD" \
    2>&1 | tee "launch_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Launch complete."
echo "Log: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
