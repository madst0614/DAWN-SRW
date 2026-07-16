#!/bin/bash
set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
INIT_FROM=""
DOWNSTREAM_RUN_ID=""
GH_TOKEN=""
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tpu) TPU_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --run-id) DOWNSTREAM_RUN_ID="$2"; shift 2 ;;
    --config) CONFIGS+=("$2"); shift 2 ;;
    --token) GH_TOKEN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tpu NAME [--branch BRANCH] [--init-from PRETRAIN_RUN_OR_CKPT] [--run-id RUN_ID] --config cfg_or_suite.yaml [--config cfg2.yaml ...]"
      echo ""
      echo "Each --config may be a normal per-task downstream YAML or a downstream_suite YAML."
      echo "If --init-from is omitted, a suite/task config may provide init_from."
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TPU_NAME" ]]; then echo "ERROR: --tpu required" >&2; exit 1; fi
if [[ ${#CONFIGS[@]} -eq 0 ]]; then echo "ERROR: at least one --config required" >&2; exit 1; fi
if [[ -z "$DOWNSTREAM_RUN_ID" ]]; then
  DOWNSTREAM_RUN_ID="run_$(date -u '+%Y%m%dT%H%M%SZ')_$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
fi
if [[ ! "$DOWNSTREAM_RUN_ID" =~ ^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8,32}$ ]]; then
  echo "ERROR: invalid downstream run id: $DOWNSTREAM_RUN_ID" >&2
  exit 1
fi

resolve_local_config_path() {
  local p="$1"
  if [ -f "$p" ]; then
    printf '%s\n' "$p"
  elif [ -f "${p}.yaml" ]; then
    printf '%s\n' "${p}.yaml"
  elif [ -f "${p}.yml" ]; then
    printf '%s\n' "${p}.yml"
  else
    printf '%s\n' "$p"
  fi
}

require_remote_visible_file() {
  local path="$1"
  if ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "ERROR: $path is not tracked by git, so TPU workers will not see it after checkout." >&2
    echo "       Commit and push it to --branch '$BRANCH', or pass a branch that already contains it." >&2
    exit 1
  fi
  if ! git diff --quiet -- "$path" || ! git diff --cached --quiet -- "$path"; then
    echo "ERROR: $path has uncommitted changes, so TPU workers will see different contents." >&2
    echo "       Commit and push the change before launching." >&2
    exit 1
  fi
}

RESOLVED_CONFIGS=()
for CFG in "${CONFIGS[@]}"; do
  RESOLVED="$(resolve_local_config_path "$CFG")"
  if [ ! -f "$RESOLVED" ]; then
    echo "ERROR: config not found locally: $CFG" >&2
    exit 1
  fi
  require_remote_visible_file "$RESOLVED"
  RESOLVED_CONFIGS+=("$RESOLVED")
done
CONFIGS=("${RESOLVED_CONFIGS[@]}")

# Every launch uses the setup/sequence/expansion path, including normal
# per-task configs (the expander passes those through unchanged).
require_remote_visible_file scripts/setup_and_run_downstream_tpu_pod.sh
require_remote_visible_file scripts/run_downstream_sequence.sh
require_remote_visible_file scripts/expand_downstream_suite.py
require_remote_visible_file scripts/downstream_protocol.py

if [[ -n "$GH_TOKEN" ]]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/madst0614/dawn-spatial.git"
else
  REPO_URL="https://github.com/madst0614/dawn-spatial.git"
fi
CONFIGS_JOINED=$(IFS='|'; echo "${CONFIGS[*]}")

echo "============================================"
echo "Launching TPU Pod downstream sequence"
echo "  TPU:       $TPU_NAME"
echo "  Zone:      $ZONE"
echo "  Project:   $PROJECT"
echo "  Branch:    $BRANCH"
echo "  Init from: ${INIT_FROM:-<none>}"
echo "  Run ID:    $DOWNSTREAM_RUN_ID"
echo "  Configs:   ${CONFIGS[*]}"
echo "============================================"

echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --format="value(state)"

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
INIT_FROM='${INIT_FROM}'
DOWNSTREAM_RUN_ID='${DOWNSTREAM_RUN_ID}'
CONFIGS='${CONFIGS_JOINED}'
export BRANCH INIT_FROM DOWNSTREAM_RUN_ID CONFIGS

if [ -d "\$HOME/dawn-spatial/.git" ]; then
  cd "\$HOME/dawn-spatial"
  git fetch origin "\$BRANCH" --depth 1
  git checkout -B "\$BRANCH" FETCH_HEAD
  git reset --hard FETCH_HEAD
  git clean -fd
else
  rm -rf "\$HOME/dawn-spatial"
  git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$HOME/dawn-spatial"
  cd "\$HOME/dawn-spatial"
fi

bash scripts/setup_and_run_downstream_tpu_pod.sh
EOFCMD

echo "Sending downstream command to all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --worker=all \
  --command="$REMOTE_CMD" \
  2>&1 | tee "launch_downstream_${TPU_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Launch complete."
echo "  Log:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"
