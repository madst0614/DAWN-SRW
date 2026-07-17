#!/bin/bash
# =============================================================================
# TPU VM/Pod launcher for DAWN-SRW v4166/v417x analysis
# =============================================================================
# This script does not create a TPU.  It launches the analysis process on an
# already-created TPU VM/Pod, on every worker, with a dedicated analysis tmux
# session and log for train_analysis mode.
#
# Example:
#   bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh \
#     --tpu dawn-400m-v4-64 \
#     --zone us-central2-b \
#     --project dawn-486218 \
#     --branch main \
#     --output gs://dawn-tpu-data-c4/analysis/v4166_400M_final \
#     --stages eval,prune,geometry,usage,trace,ablation,report
#
# Train-side checkpoint analysis preset:
#   bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh \
#     --preset v4166-1B \
#     --tpu spatial-400m \
#     --project dawn-486218 \
#     --zone us-central2-b \
#     --branch codex/v4167-poc \
#     --mode train_analysis
#
# If the TPU does not exist yet, create it separately, for example:
#   gcloud compute tpus tpu-vm create dawn-400m-v4-64 \
#     --zone=us-central2-b \
#     --accelerator-type=v4-64 \
#     --version=tpu-vm-v4-base \
#     --project=dawn-486218
# =============================================================================

set -euo pipefail

TPU_NAME=""
ZONE="us-central2-b"
PROJECT="dawn-486218"
BRANCH="main"
GH_TOKEN=""
REPO_URL="https://github.com/madst0614/DAWN-SRW.git"
PRESET=""

CHECKPOINT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_400M_c4_40B_v4_64/run_vspatial-r1-v4.1.6.6_20260622_212706_3201/checkpoints/000000076293"
OUTPUT="gs://dawn-tpu-data-c4/analysis/v4166_400M_final"
STAGES="eval,prune,geometry,usage,trace,ablation,report"
ANALYSIS_ARGS=""
MODE="analysis"
WORKERS="auto"
DETACH="1"
DETACH_EXPLICIT="0"
INSTALL_DEPS="1"
TMUX_SESSION="train"
REMOTE_LOG="~/train.log"
DRY_RUN="0"
OUTPUT_EXPLICIT="0"
MODE_EXPLICIT="0"
CHECKPOINT_DIR_EXPLICIT="0"
PRUNE_EPS_EXPLICIT="0"
REMOTE_LOG_EXPLICIT="0"
ANALYSIS_PRESET_EXPLICIT="0"
APPEND_REMOTE_LOG="0"
FAIL_ON_CONFLICT="0"
SKIP_REPO_UPDATE="0"
SYNC_LOCAL_ANALYSIS="0"
TRANSITION_TOPK_QK="${DAWN_TRANSITION_TOPK_QK:-512}"
TRANSITION_TOPK_V="${DAWN_TRANSITION_TOPK_V:-2048}"
TRANSITION_TOPK_RST="${DAWN_TRANSITION_TOPK_RST:-4096}"
CAUSAL_MAX_PROMPTS="${DAWN_CAUSAL_MAX_PROMPTS:-6}"
OPERATOR_DATASET_ROOT="${DAWN_OPERATOR_DATASET_ROOT:-gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2}"
OPERATOR_ANALYSIS_PROFILE="${DAWN_OPERATOR_ANALYSIS_PROFILE:-monitor}"
OPERATOR_DATASETS="${DAWN_OPERATOR_DATASETS:-all}"
OPERATOR_CACHE_DIR="${DAWN_OPERATOR_CACHE_DIR:-/tmp/dawn_operator_analysis_cache}"
OPERATOR_BEHAVIOR_MAX_EXAMPLES="${DAWN_OPERATOR_BEHAVIOR_MAX_EXAMPLES:-}"
OPERATOR_TRACE_MAX_EXAMPLES="${DAWN_OPERATOR_TRACE_MAX_EXAMPLES:-}"
OPERATOR_CAUSAL_MAX_EXAMPLES="${DAWN_OPERATOR_CAUSAL_MAX_EXAMPLES:-}"
OPERATOR_TRACE_PER_GROUP="${DAWN_OPERATOR_TRACE_PER_GROUP:-}"
OPERATOR_CAUSAL_PER_GROUP="${DAWN_OPERATOR_CAUSAL_PER_GROUP:-}"
OPERATOR_ANALYSIS_RESUME="${DAWN_OPERATOR_ANALYSIS_RESUME:-1}"
OPERATOR_ANALYSIS_SEED="${DAWN_OPERATOR_ANALYSIS_SEED:-4171}"

TRAIN_ANALYSIS_CONFIG="${DAWN_TRAIN_ANALYSIS_CONFIG:-}"
TRAIN_ANALYSIS_CHECKPOINT_DIR="${DAWN_TRAIN_ANALYSIS_CHECKPOINT_DIR:-gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_1p3B_c4_20B_v4_64_new}"
TRAIN_ANALYSIS_MAX_BATCHES="${DAWN_TRAIN_ANALYSIS_MAX_BATCHES:-8}"
TRAIN_ANALYSIS_PRUNE_EPS="${DAWN_TRAIN_ANALYSIS_PRUNE_EPS:-1e-2,1e-1}"
TRAIN_ANALYSIS_PRESET="${DAWN_TRAIN_ANALYSIS_PRESET:-qk_closed}"
TRAIN_ANALYSIS_ITEMS="${DAWN_TRAIN_ANALYSIS_ITEMS:-}"
TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS="${DAWN_TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS:-3}"
TRAIN_ANALYSIS_GENERATION_MAX_TOKENS="${DAWN_TRAIN_ANALYSIS_GENERATION_MAX_TOKENS:-64}"
TRAIN_ANALYSIS_GENERATION_TEMPERATURE="${DAWN_TRAIN_ANALYSIS_GENERATION_TEMPERATURE:-0.8}"
TRAIN_ANALYSIS_GENERATION_TOP_K="${DAWN_TRAIN_ANALYSIS_GENERATION_TOP_K:-50}"
if [[ -n "${DAWN_TRAIN_ANALYSIS_PRUNE_EPS+x}" ]]; then
    PRUNE_EPS_EXPLICIT="1"
fi
if [[ -n "${DAWN_TRAIN_ANALYSIS_PRESET+x}" ]]; then
    ANALYSIS_PRESET_EXPLICIT="1"
fi

normalize_gcs_arg() {
    local value="$1"
    if [[ "$value" == dawn-tpu-data-c4/* ]]; then
        printf 'gs://%s' "$value"
    else
        printf '%s' "$value"
    fi
}

path_name() {
    local value="${1%/}"
    value="${value//\\//}"
    printf '%s' "${value##*/}"
}

path_parent() {
    local value="${1%/}"
    value="${value//\\//}"
    if [[ "$value" == */* ]]; then
        printf '%s' "${value%/*}"
    else
        printf '.'
    fi
}

apply_preset() {
    local preset_lc="${1,,}"
    case "$preset_lc" in
        v4166-1b|v4166-1p3b|v4166-1p3b-c4-20b|v4166-1p3b-c4-20b-v4-64)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4166_1p3B_c4_20B_v4_64_new"
            fi
            if [[ "$PRUNE_EPS_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRUNE_EPS="1e-2,1e-1"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4166_1b"
            fi
            ;;
        v4171-400m|v4171-400m-c4-40b|v4171-400m-c4-40b-v4-64)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4171_400M_c4_40B_v4_64_emb_tau/run_vspatial-r1-v4.1.7.1_20260712_172338_3201/checkpoints"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4171_self_organization"
            fi
            if [[ "$OUTPUT_EXPLICIT" == "0" ]]; then
                OUTPUT="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4171_400M_c4_40B_v4_64_emb_tau/run_vspatial-r1-v4.1.7.1_20260712_172338_3201/side_analysis/v4171_self_organization_v2"
                OUTPUT_EXPLICIT="1"
            fi
            if [[ "$REMOTE_LOG_EXPLICIT" == "0" ]]; then
                REMOTE_LOG="~/train.log"
                REMOTE_LOG_EXPLICIT="1"
            fi
            APPEND_REMOTE_LOG="1"
            FAIL_ON_CONFLICT="1"
            ;;
        v4171-1p3b|v4171-1p3b-c4-20b|v4171-1p3b-c4-20b-v4-64)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4171_1p3B_c4_20B_v4_64_emb_tau"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4171"
            fi
            ;;
        v4172-400m|v4172-400m-c4-40b|v4172-400m-c4-40b-v4-64)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4172_self_organization"
            fi
            ;;
        v4172-400m-ver1|v4172-400m-c4-40b-v4-64-ver1)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4172_self_organization"
            fi
            ;;
        v4172-400m-ver1-den-qk0p5-v1p0-rst1p2|v4172-400m-c4-40b-v4-64-ver1-den-qk0p5-v1p0-rst1p2)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4172_self_organization"
            fi
            ;;
        v4172-1p3b|v4172-1p3b-c4-20b|v4172-1p3b-c4-20b-v4-64|v4172-1p3b-c4-20b-v4-64-ver1-den-qk0p5-v1p0-rst1p2)
            if [[ "$MODE_EXPLICIT" == "0" ]]; then
                MODE="train_analysis"
            fi
            if [[ "$CHECKPOINT_DIR_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_CHECKPOINT_DIR="gs://dawn-tpu-data-c4/checkpoints/train_config_v4172_1p3B_c4_20B_v4_64_ver1_den_qk0p5_v1p0_rst1p2"
            fi
            if [[ "$ANALYSIS_PRESET_EXPLICIT" == "0" ]]; then
                TRAIN_ANALYSIS_PRESET="v4172"
            fi
            ;;
        *)
            echo "ERROR: unknown --preset $1" >&2
            echo "Known presets: v4166-1B, v4171-400M, v4171-1p3B, v4172-400M variants, v4172-1p3B" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu) TPU_NAME="$2"; shift 2 ;;
        --zone) ZONE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --mode) MODE="$2"; MODE_EXPLICIT="1"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --token) GH_TOKEN="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --config) TRAIN_ANALYSIS_CONFIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$(normalize_gcs_arg "$2")"; shift 2 ;;
        --checkpoint-dir) TRAIN_ANALYSIS_CHECKPOINT_DIR="$(normalize_gcs_arg "$2")"; CHECKPOINT_DIR_EXPLICIT="1"; shift 2 ;;
        --output) OUTPUT="$(normalize_gcs_arg "$2")"; OUTPUT_EXPLICIT="1"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --foreground) DETACH="0"; DETACH_EXPLICIT="1"; shift ;;
        --detach) DETACH="1"; DETACH_EXPLICIT="1"; shift ;;
        --log) REMOTE_LOG="$2"; REMOTE_LOG_EXPLICIT="1"; shift 2 ;;
        --no-install) INSTALL_DEPS="0"; shift ;;
        --skip-repo-update) SKIP_REPO_UPDATE="1"; shift ;;
        --sync-local-analysis) SYNC_LOCAL_ANALYSIS="1"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --from-scratch) ANALYSIS_ARGS="$ANALYSIS_ARGS --from-scratch"; shift ;;
        --retry-failed) ANALYSIS_ARGS="$ANALYSIS_ARGS --retry-failed"; shift ;;
        --fail-fast) ANALYSIS_ARGS="$ANALYSIS_ARGS --fail-fast"; shift ;;
        --parity-only) ANALYSIS_ARGS="$ANALYSIS_ARGS --v417x-parity-only"; shift ;;
        --mesh-data) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-data $2"; shift 2 ;;
        --mesh-model) ANALYSIS_ARGS="$ANALYSIS_ARGS --mesh-model $2"; shift 2 ;;
        --eval-max-tokens) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-max-tokens $2"; shift 2 ;;
        --eval-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --eval-batch-size $2"; shift 2 ;;
        --prune-eps) TRAIN_ANALYSIS_PRUNE_EPS="$2"; PRUNE_EPS_EXPLICIT="1"; shift 2 ;;
        --analysis-preset) TRAIN_ANALYSIS_PRESET="$2"; ANALYSIS_PRESET_EXPLICIT="1"; shift 2 ;;
        --analysis-items) TRAIN_ANALYSIS_ITEMS="$2"; shift 2 ;;
        --analysis-generation-max-prompts) TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS="$2"; shift 2 ;;
        --analysis-generation-max-tokens) TRAIN_ANALYSIS_GENERATION_MAX_TOKENS="$2"; shift 2 ;;
        --analysis-generation-temperature) TRAIN_ANALYSIS_GENERATION_TEMPERATURE="$2"; shift 2 ;;
        --analysis-generation-top-k) TRAIN_ANALYSIS_GENERATION_TOP_K="$2"; shift 2 ;;
        --transition-topk-qk) TRANSITION_TOPK_QK="$2"; shift 2 ;;
        --transition-topk-v) TRANSITION_TOPK_V="$2"; shift 2 ;;
        --transition-topk-rst) TRANSITION_TOPK_RST="$2"; shift 2 ;;
        --causal-max-prompts) CAUSAL_MAX_PROMPTS="$2"; shift 2 ;;
        --operator-dataset-root) OPERATOR_DATASET_ROOT="$(normalize_gcs_arg "$2")"; shift 2 ;;
        --operator-analysis-profile) OPERATOR_ANALYSIS_PROFILE="$2"; shift 2 ;;
        --operator-datasets) OPERATOR_DATASETS="$2"; shift 2 ;;
        --operator-cache-dir) OPERATOR_CACHE_DIR="$2"; shift 2 ;;
        --operator-behavior-max-examples) OPERATOR_BEHAVIOR_MAX_EXAMPLES="$2"; shift 2 ;;
        --operator-trace-max-examples) OPERATOR_TRACE_MAX_EXAMPLES="$2"; shift 2 ;;
        --operator-causal-max-examples) OPERATOR_CAUSAL_MAX_EXAMPLES="$2"; shift 2 ;;
        --operator-trace-per-group) OPERATOR_TRACE_PER_GROUP="$2"; shift 2 ;;
        --operator-causal-per-group) OPERATOR_CAUSAL_PER_GROUP="$2"; shift 2 ;;
        --operator-analysis-resume) OPERATOR_ANALYSIS_RESUME="1"; shift ;;
        --no-operator-analysis-resume) OPERATOR_ANALYSIS_RESUME="0"; shift ;;
        --operator-analysis-seed) OPERATOR_ANALYSIS_SEED="$2"; shift 2 ;;
        --list-analysis-items) ANALYSIS_ARGS="$ANALYSIS_ARGS --list-train-analysis-items"; shift ;;
        --usage-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-max-sequences $2"; shift 2 ;;
        --usage-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-batch-size $2"; shift 2 ;;
        --usage-seq-len) ANALYSIS_ARGS="$ANALYSIS_ARGS --usage-seq-len $2"; shift 2 ;;
        --trace-max-prompts) ANALYSIS_ARGS="$ANALYSIS_ARGS --trace-max-prompts $2"; shift 2 ;;
        --ablation-max-sequences) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-max-sequences $2"; shift 2 ;;
        --ablation-batch-size) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-batch-size $2"; shift 2 ;;
        --ablation-k-list) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-k-list $2"; shift 2 ;;
        --ablation-strategies) ANALYSIS_ARGS="$ANALYSIS_ARGS --ablation-strategies $2"; shift 2 ;;
        --max-jobs-per-stage) TRAIN_ANALYSIS_MAX_BATCHES="$2"; ANALYSIS_ARGS="$ANALYSIS_ARGS --max-jobs-per-stage $2"; shift 2 ;;
        --extra-arg) ANALYSIS_ARGS="$ANALYSIS_ARGS $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --tpu NAME [options]"
            echo ""
            echo "Required:"
            echo "  --tpu NAME"
            echo ""
            echo "Core:"
            echo "  --preset NAME             Known: v4166-1B, v4171-400M, v4171-1p3B, v4172-400M variants, v4172-1p3B"
            echo "  --mode MODE               analysis, train, or train_analysis. Default: $MODE"
            echo "  --config PATH             Optional train_analysis fallback config. Default: checkpoint full_config"
            echo "  --checkpoint PATH_OR_GS   Default: $CHECKPOINT"
            echo "  --checkpoint-dir DIR      train_analysis base checkpoint dir. Default: $TRAIN_ANALYSIS_CHECKPOINT_DIR"
            echo "  --output PATH_OR_GS       Default: $OUTPUT"
            echo "  --stages CSV              Default: $STAGES"
            echo "  --analysis-preset NAME    Train-analysis item preset. Default: $TRAIN_ANALYSIS_PRESET"
            echo "  --analysis-items CSV      Override train-analysis items"
            echo "  --analysis-generation-max-prompts N"
            echo "  --analysis-generation-max-tokens N"
            echo "  --analysis-generation-temperature F"
            echo "  --analysis-generation-top-k N"
            echo "  --transition-topk-qk N    Default: $TRANSITION_TOPK_QK"
            echo "  --transition-topk-v N     Default: $TRANSITION_TOPK_V"
            echo "  --transition-topk-rst N   Default: $TRANSITION_TOPK_RST"
            echo "  --causal-max-prompts N    Default: $CAUSAL_MAX_PROMPTS"
            echo "  --operator-dataset-root PATH"
            echo "  --operator-analysis-profile smoke|monitor|full"
            echo "  --operator-datasets CSV   Default: $OPERATOR_DATASETS"
            echo "  --operator-cache-dir PATH Default: $OPERATOR_CACHE_DIR"
            echo "  --operator-behavior-max-examples N"
            echo "  --operator-trace-max-examples N"
            echo "  --operator-causal-max-examples N"
            echo "  --operator-trace-per-group N"
            echo "  --operator-causal-per-group N"
            echo "  --operator-analysis-resume | --no-operator-analysis-resume"
            echo "  --operator-analysis-seed N Default: $OPERATOR_ANALYSIS_SEED"
            echo "  --list-analysis-items     Print train-analysis item catalog"
            echo "  --branch BRANCH           Default: $BRANCH"
            echo "  --repo-url URL            Default: $REPO_URL"
            echo "  --log PATH                Remote log path. Default: ~/train.log"
            echo ""
            echo "TPU/GCP:"
            echo "  --zone ZONE               Default: $ZONE"
            echo "  --project PROJECT         Default: $PROJECT"
            echo "  --workers all|0|N         Default: all"
            echo ""
            echo "Execution:"
            echo "  --detach                  Run in the mode-specific tmux session (default)"
            echo "  --foreground              Run foreground on the SSH command"
            echo "  --dry-run                 Print resolved command without launching"
            echo "  --skip-repo-update        Use an already-synced remote worktree"
            echo "  --sync-local-analysis     Overlay the current uncommitted analysis files after repo update"
            echo "  --from-scratch            Disable analysis artifact resume"
            echo "  --retry-failed"
            echo "  --fail-fast"
            echo "  --parity-only             Run v417x exact parity smoke and exit before traces"
            echo "  --mesh-data N"
            echo "  --mesh-model N"
            echo "  --max-jobs-per-stage N"
            echo "  --extra-arg '...'"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$PRESET" ]]; then
    apply_preset "$PRESET"
fi

CHECKPOINT="$(normalize_gcs_arg "$CHECKPOINT")"
TRAIN_ANALYSIS_CHECKPOINT_DIR="$(normalize_gcs_arg "$TRAIN_ANALYSIS_CHECKPOINT_DIR")"
OUTPUT="$(normalize_gcs_arg "$OUTPUT")"
OPERATOR_DATASET_ROOT="$(normalize_gcs_arg "$OPERATOR_DATASET_ROOT")"

case "$OPERATOR_ANALYSIS_PROFILE" in
    smoke|monitor|full) ;;
    *)
        echo "ERROR: unsupported --operator-analysis-profile $OPERATOR_ANALYSIS_PROFILE" >&2
        exit 1
        ;;
esac
case "$OPERATOR_ANALYSIS_RESUME" in
    0|1) ;;
    *)
        echo "ERROR: DAWN_OPERATOR_ANALYSIS_RESUME must be 0 or 1" >&2
        exit 1
        ;;
esac

if [[ -z "$TPU_NAME" ]]; then
    echo "ERROR: --tpu required" >&2
    exit 1
fi

case "$MODE" in
    analysis|full|full_analysis)
        MODE="analysis"
        ;;
    train|training|train_analysis)
        MODE="train_analysis"
        ;;
    *)
        echo "ERROR: unsupported --mode $MODE (expected analysis, train, or train_analysis)" >&2
        exit 1
        ;;
esac

if [[ "$MODE" == "train_analysis" ]]; then
    STAGES="train_analysis"
    CHECKPOINT="$TRAIN_ANALYSIS_CHECKPOINT_DIR"
    TMUX_SESSION="train_analysis"
    if [[ "$REMOTE_LOG_EXPLICIT" == "0" ]]; then
        REMOTE_LOG="~/train_analysis.log"
    fi
    if [[ "$OUTPUT_EXPLICIT" == "0" ]]; then
        if [[ "$(path_name "$TRAIN_ANALYSIS_CHECKPOINT_DIR")" == "checkpoints" ]]; then
            OUTPUT="$(path_parent "$TRAIN_ANALYSIS_CHECKPOINT_DIR")/side_analysis"
        else
            OUTPUT="${TRAIN_ANALYSIS_CHECKPOINT_DIR%/}/side_analysis"
        fi
    fi
fi

if [[ "$WORKERS" = "auto" ]]; then
    WORKERS="all"
fi

if [[ "$REPO_URL" == git@github.com:* ]]; then
    REPO_URL="https://github.com/${REPO_URL#git@github.com:}"
fi
if [[ -n "$GH_TOKEN" && "$REPO_URL" == https://github.com/* ]]; then
    REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO_URL#https://github.com/}"
fi
REPO_URL_DISPLAY="$REPO_URL"
if [[ -n "$GH_TOKEN" ]]; then
    REPO_URL_DISPLAY="${REPO_URL_DISPLAY/$GH_TOKEN/***}"
fi

COPY_CMD="bash scripts/launch_dawn_v4166_analysis_tpu_pod.sh --tpu $TPU_NAME --project $PROJECT --zone $ZONE --branch $BRANCH --mode $MODE"
if [[ -n "$PRESET" ]]; then
    COPY_CMD="$COPY_CMD --preset $PRESET"
fi
if [[ "$MODE" == "train_analysis" ]]; then
    COPY_CMD="$COPY_CMD --checkpoint-dir $TRAIN_ANALYSIS_CHECKPOINT_DIR"
    if [[ -n "$TRAIN_ANALYSIS_CONFIG" ]]; then
        COPY_CMD="$COPY_CMD --config $TRAIN_ANALYSIS_CONFIG"
    fi
    if [[ "$OUTPUT_EXPLICIT" == "1" ]]; then
        COPY_CMD="$COPY_CMD --output $OUTPUT"
    fi
    if [[ "$REMOTE_LOG_EXPLICIT" == "1" ]]; then
        COPY_CMD="$COPY_CMD --log $REMOTE_LOG"
    fi
    if [[ -n "$TRAIN_ANALYSIS_PRESET" ]]; then
        COPY_CMD="$COPY_CMD --analysis-preset $TRAIN_ANALYSIS_PRESET"
    fi
    if [[ -n "$TRAIN_ANALYSIS_ITEMS" ]]; then
        COPY_CMD="$COPY_CMD --analysis-items $TRAIN_ANALYSIS_ITEMS"
    fi
    COPY_CMD="$COPY_CMD --analysis-generation-max-prompts $TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS"
    COPY_CMD="$COPY_CMD --analysis-generation-max-tokens $TRAIN_ANALYSIS_GENERATION_MAX_TOKENS"
    COPY_CMD="$COPY_CMD --analysis-generation-temperature $TRAIN_ANALYSIS_GENERATION_TEMPERATURE"
    COPY_CMD="$COPY_CMD --analysis-generation-top-k $TRAIN_ANALYSIS_GENERATION_TOP_K"
    COPY_CMD="$COPY_CMD --transition-topk-qk $TRANSITION_TOPK_QK --transition-topk-v $TRANSITION_TOPK_V --transition-topk-rst $TRANSITION_TOPK_RST --causal-max-prompts $CAUSAL_MAX_PROMPTS"
    COPY_CMD="$COPY_CMD --operator-dataset-root $OPERATOR_DATASET_ROOT --operator-analysis-profile $OPERATOR_ANALYSIS_PROFILE --operator-datasets $OPERATOR_DATASETS --operator-cache-dir $OPERATOR_CACHE_DIR --operator-analysis-seed $OPERATOR_ANALYSIS_SEED"
    if [[ "$OPERATOR_ANALYSIS_RESUME" == "1" ]]; then
        COPY_CMD="$COPY_CMD --operator-analysis-resume"
    else
        COPY_CMD="$COPY_CMD --no-operator-analysis-resume"
    fi
    for _operator_pair in \
        "operator-behavior-max-examples:$OPERATOR_BEHAVIOR_MAX_EXAMPLES" \
        "operator-trace-max-examples:$OPERATOR_TRACE_MAX_EXAMPLES" \
        "operator-causal-max-examples:$OPERATOR_CAUSAL_MAX_EXAMPLES" \
        "operator-trace-per-group:$OPERATOR_TRACE_PER_GROUP" \
        "operator-causal-per-group:$OPERATOR_CAUSAL_PER_GROUP"; do
        _operator_name="${_operator_pair%%:*}"
        _operator_value="${_operator_pair#*:}"
        if [[ -n "$_operator_value" ]]; then
            COPY_CMD="$COPY_CMD --$_operator_name $_operator_value"
        fi
    done
fi
if [[ "$SKIP_REPO_UPDATE" == "1" ]]; then
    COPY_CMD="$COPY_CMD --skip-repo-update"
fi
if [[ "$SYNC_LOCAL_ANALYSIS" == "1" ]]; then
    COPY_CMD="$COPY_CMD --sync-local-analysis"
fi
WATCH_LOG_CMD="bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --log $REMOTE_LOG --target $TMUX_SESSION --summary"

echo "============================================================"
echo "DAWN-SRW v4166/v417x analysis launcher"
echo "============================================================"
echo "Run:"
echo "  mode            : $MODE"
echo "  tpu             : $TPU_NAME"
echo "  project         : $PROJECT"
echo "  zone            : $ZONE"
echo "  branch          : $BRANCH"
echo "  repo            : $REPO_URL_DISPLAY"
if [[ -n "$PRESET" ]]; then
    echo "  preset          : $PRESET"
fi
echo "  workers         : $WORKERS"
echo "  detached        : $DETACH"
echo "  tmux_session    : $TMUX_SESSION"
echo "  remote_log      : $REMOTE_LOG"
echo "  append_log      : $APPEND_REMOTE_LOG"
if [[ "$MODE" == "train_analysis" ]]; then
    echo "  config          : ${TRAIN_ANALYSIS_CONFIG:-checkpoint full_config}"
    echo "  checkpoint_dir  : $TRAIN_ANALYSIS_CHECKPOINT_DIR"
    echo "  output          : $OUTPUT"
    echo "  analysis_batches: $TRAIN_ANALYSIS_MAX_BATCHES"
    echo "  prune_eps       : $TRAIN_ANALYSIS_PRUNE_EPS"
    echo "  analysis_preset : $TRAIN_ANALYSIS_PRESET"
    echo "  analysis_items  : ${TRAIN_ANALYSIS_ITEMS:-preset default}"
    echo "  gen_max_prompts : $TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS"
    echo "  gen_max_tokens  : $TRAIN_ANALYSIS_GENERATION_MAX_TOKENS"
    echo "  gen_temperature : $TRAIN_ANALYSIS_GENERATION_TEMPERATURE"
    echo "  gen_top_k       : $TRAIN_ANALYSIS_GENERATION_TOP_K"
    echo "  transition_topk : qk=$TRANSITION_TOPK_QK v=$TRANSITION_TOPK_V rst=$TRANSITION_TOPK_RST"
    echo "  causal_prompts  : $CAUSAL_MAX_PROMPTS"
    echo "  operator_data   : $OPERATOR_DATASET_ROOT"
    echo "  operator_profile: $OPERATOR_ANALYSIS_PROFILE"
    echo "  operator_sets   : $OPERATOR_DATASETS"
    echo "  operator_cache  : $OPERATOR_CACHE_DIR"
    echo "  operator_resume : $OPERATOR_ANALYSIS_RESUME seed=$OPERATOR_ANALYSIS_SEED"
    echo "  operator_limits : behavior=${OPERATOR_BEHAVIOR_MAX_EXAMPLES:-profile} trace=${OPERATOR_TRACE_MAX_EXAMPLES:-profile} causal=${OPERATOR_CAUSAL_MAX_EXAMPLES:-profile}"
    echo "  repo_update     : $([[ "$SKIP_REPO_UPDATE" == "1" ]] && echo skip || echo fetch)"
    echo "  local_overlay   : $SYNC_LOCAL_ANALYSIS"
else
    echo "  checkpoint      : $CHECKPOINT"
    echo "  output          : $OUTPUT"
    echo "  stages          : $STAGES"
    echo "  args            : ${ANALYSIS_ARGS:-<none>}"
fi
echo ""
echo "Copy-paste:"
echo "  $COPY_CMD"
echo ""
echo "Watch logs:"
echo "  $WATCH_LOG_CMD"
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run: no TPU command will be sent."
    if [[ "$MODE" == "train_analysis" ]]; then
        echo "Remote Python:"
        DRY_RUN_ANALYSIS_CMD="python3 -u scripts/analyze_dawn_srw_v4166.py --train-analysis --checkpoint-dir $TRAIN_ANALYSIS_CHECKPOINT_DIR --output $OUTPUT --train-analysis-max-batches $TRAIN_ANALYSIS_MAX_BATCHES --prune-eps $TRAIN_ANALYSIS_PRUNE_EPS --train-analysis-preset $TRAIN_ANALYSIS_PRESET --train-analysis-generation-max-prompts $TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS --train-analysis-generation-max-tokens $TRAIN_ANALYSIS_GENERATION_MAX_TOKENS --train-analysis-generation-temperature $TRAIN_ANALYSIS_GENERATION_TEMPERATURE --train-analysis-generation-top-k $TRAIN_ANALYSIS_GENERATION_TOP_K --transition-topk-qk $TRANSITION_TOPK_QK --transition-topk-v $TRANSITION_TOPK_V --transition-topk-rst $TRANSITION_TOPK_RST --causal-max-prompts $CAUSAL_MAX_PROMPTS --operator-dataset-root $OPERATOR_DATASET_ROOT --operator-analysis-profile $OPERATOR_ANALYSIS_PROFILE --operator-datasets $OPERATOR_DATASETS --operator-cache-dir $OPERATOR_CACHE_DIR --operator-analysis-seed $OPERATOR_ANALYSIS_SEED --init-distributed"
        if [[ "$OPERATOR_ANALYSIS_RESUME" == "1" ]]; then
            DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-analysis-resume"
        else
            DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --no-operator-analysis-resume"
        fi
        [[ -z "$OPERATOR_BEHAVIOR_MAX_EXAMPLES" ]] || DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-behavior-max-examples $OPERATOR_BEHAVIOR_MAX_EXAMPLES"
        [[ -z "$OPERATOR_TRACE_MAX_EXAMPLES" ]] || DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-trace-max-examples $OPERATOR_TRACE_MAX_EXAMPLES"
        [[ -z "$OPERATOR_CAUSAL_MAX_EXAMPLES" ]] || DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-causal-max-examples $OPERATOR_CAUSAL_MAX_EXAMPLES"
        [[ -z "$OPERATOR_TRACE_PER_GROUP" ]] || DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-trace-per-group $OPERATOR_TRACE_PER_GROUP"
        [[ -z "$OPERATOR_CAUSAL_PER_GROUP" ]] || DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --operator-causal-per-group $OPERATOR_CAUSAL_PER_GROUP"
        if [[ -n "$TRAIN_ANALYSIS_CONFIG" ]]; then
            DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --config $TRAIN_ANALYSIS_CONFIG"
        fi
        if [[ -n "$TRAIN_ANALYSIS_ITEMS" ]]; then
            DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD --train-analysis-items $TRAIN_ANALYSIS_ITEMS"
        fi
        if [[ -n "$ANALYSIS_ARGS" ]]; then
            DRY_RUN_ANALYSIS_CMD="$DRY_RUN_ANALYSIS_CMD $ANALYSIS_ARGS"
        fi
        echo "  $DRY_RUN_ANALYSIS_CMD"
    else
        echo "Remote Python:"
        echo "  python3 -u scripts/analyze_dawn_srw_v4166.py --checkpoint $CHECKPOINT --output $OUTPUT --stages $STAGES --init-distributed ${ANALYSIS_ARGS:-}"
    fi
    exit 0
fi

echo "Checking TPU status..."
TPU_STATE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(state)")"
echo "$TPU_STATE"

ACCELERATOR_TYPE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(acceleratorType)" || true)"
NETWORK_ENDPOINTS="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --format="value(networkEndpoints[].ipAddress)" || true)"
WORKER_COUNT="$(printf '%s\n' "$NETWORK_ENDPOINTS" | awk 'NF {count += NF} END {print count + 0}')"
ACCELERATOR_WORKER_COUNT=0
ACCELERATOR_SIZE="${ACCELERATOR_TYPE##*-}"
if [[ "$ACCELERATOR_SIZE" =~ ^[0-9]+$ ]]; then
    ACCELERATOR_WORKER_COUNT=$(( (ACCELERATOR_SIZE + 7) / 8 ))
fi
if [ "$ACCELERATOR_WORKER_COUNT" -gt "$WORKER_COUNT" ]; then
    WORKER_COUNT="$ACCELERATOR_WORKER_COUNT"
fi
if [ "$WORKER_COUNT" -le 0 ]; then
    echo "ERROR: Could not determine TPU worker count." >&2
    exit 1
fi
echo "  Accelerator: ${ACCELERATOR_TYPE:-unknown}"
echo "  Detected workers: $WORKER_COUNT"

declare -a TARGET_WORKERS=()
if [[ "$WORKERS" = "all" ]]; then
    for worker in $(seq 0 $((WORKER_COUNT - 1))); do
        TARGET_WORKERS+=("$worker")
    done
else
    TARGET_WORKERS+=("$WORKERS")
fi

run_worker_command() {
    local worker="$1"
    local command="$2"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --worker="$worker" \
        --command="$command"
}

echo "Preflighting SSH on target worker(s): ${TARGET_WORKERS[*]}"
for worker in "${TARGET_WORKERS[@]}"; do
    echo "  Worker $worker SSH preflight..."
    if ! run_worker_command "$worker" 'hostname; date -Is'; then
        echo "ERROR: worker $worker SSH failed. Aborting launch." >&2
        exit 1
    fi
done

OVERLAY_ARCHIVE=""
if [[ "$SYNC_LOCAL_ANALYSIS" == "1" ]]; then
    OVERLAY_ARCHIVE="$(mktemp -t dawn_v417x_analysis_overlay_XXXXXX.tar.gz)"
    trap '[[ -z "${OVERLAY_ARCHIVE:-}" ]] || rm -f "$OVERLAY_ARCHIVE"' EXIT
    tar -czf "$OVERLAY_ARCHIVE" \
        analysis/dawn_analysis_common.py \
        analysis/dawn_analysis_trace.py \
        analysis/dawn_analysis_usage.py \
        analysis/dawn_operator_analysis.py \
        analysis/dawn_operator_datasets.py \
        analysis/dawn_train_analysis_items.py \
        analysis/dawn_train_analysis_prompt.py \
        analysis/dawn_v4171_transition.py \
        analysis/prompts/v4171_transition_pairs.jsonl \
        models/dawn_srw_v4171.py \
        models/dawn_srw_v4172.py \
        scripts/analyze_dawn_srw_v4166.py \
        scripts/launch_dawn_v4166_analysis_tpu_pod.sh
    echo "Syncing local analysis overlay to target worker(s)..."
    for worker in "${TARGET_WORKERS[@]}"; do
        gcloud compute tpus tpu-vm scp "$OVERLAY_ARCHIVE" \
            "$TPU_NAME:~/dawn_v417x_analysis_overlay.tar.gz" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --worker="$worker"
    done
fi

if [[ "$FAIL_ON_CONFLICT" == "1" ]]; then
read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
ANALYSIS_PATTERN="[a]nalyze_dawn_srw_v4166"
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PGREP_PATTERN="${ANALYSIS_PATTERN}|${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}"
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: conflicting DAWN train/analysis process already exists:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
ACCEL_USERS="$(sudo lsof /dev/accel* 2>/dev/null | grep -v PID || true)"
if [ -n "$ACCEL_USERS" ]; then
    echo "ERROR: TPU accelerator is already in use; refusing to kill the owner:" >&2
    echo "$ACCEL_USERS" >&2
    exit 1
fi
EOFCLEANUP
elif [[ "$MODE" == "train_analysis" ]]; then
read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
ANALYSIS_PATTERN="[a]nalyze_dawn_srw_v4166"
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PYTHON_PATTERN="[p]ython3"
PGREP_PATTERN="${ANALYSIS_PATTERN}|${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}|${PYTHON_PATTERN} scripts"
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${PYTHON_PATTERN} scripts/${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
pkill -9 -f "${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
sleep 3
pgrep -af "$PGREP_PATTERN" || true
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: DAWN train/analysis process remains after cleanup:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
EOFCLEANUP
else
read -r -d '' CLEANUP_CMD <<'EOFCLEANUP' || true
set -e
ANALYSIS_PATTERN="[a]nalyze_dawn_srw_v4166"
TRAIN_JAX_PATTERN="[t]rain_jax"
TRAIN_JAX_MINIMAL_PATTERN="[t]rain_jax_minimal"
PGREP_PATTERN="${ANALYSIS_PATTERN}|${TRAIN_JAX_PATTERN}|${TRAIN_JAX_MINIMAL_PATTERN}"
tmux kill-session -t train 2>/dev/null || true
pkill -9 -f "${ANALYSIS_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_PATTERN}\\.py" || true
pkill -9 -f "${TRAIN_JAX_MINIMAL_PATTERN}\\.py" || true
sudo lsof /dev/accel* 2>/dev/null | grep -v PID | awk '{print $2}' | sort -u | xargs -r sudo kill -9 || true
sleep 3
REMAINING="$(pgrep -af "$PGREP_PATTERN" || true)"
if [ -n "$REMAINING" ]; then
    echo "ERROR: DAWN process remains after cleanup:" >&2
    echo "$REMAINING" >&2
    exit 1
fi
EOFCLEANUP
fi

cleanup_target_workers() {
    local failed=0
    for worker in "${TARGET_WORKERS[@]}"; do
        if [[ "$FAIL_ON_CONFLICT" == "1" ]]; then
            echo "  Conflict preflight worker $worker..."
        else
            echo "  Cleaning worker $worker..."
        fi
        if ! run_worker_command "$worker" "$CLEANUP_CMD"; then
            echo "ERROR: worker $worker cleanup failed." >&2
            failed=1
        fi
    done
    return "$failed"
}

if [[ "$FAIL_ON_CONFLICT" == "1" ]]; then
    echo "Checking for conflicting train/analysis processes on target worker(s)..."
elif [[ "$MODE" == "train_analysis" && "$DETACH" == "0" ]]; then
    echo "Cleaning old train_analysis processes on target worker(s)..."
else
    echo "Cleaning old train/analysis processes on target worker(s)..."
fi
if ! cleanup_target_workers; then
    echo "ERROR: process conflict/cleanup verification failed. Aborting launch." >&2
    exit 1
fi

read -r -d '' REMOTE_CMD <<EOFCMD || true
set -euo pipefail
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
CHECKPOINT='${CHECKPOINT}'
OUTPUT='${OUTPUT}'
STAGES='${STAGES}'
ANALYSIS_ARGS='${ANALYSIS_ARGS}'
MODE='${MODE}'
TRAIN_ANALYSIS_CONFIG='${TRAIN_ANALYSIS_CONFIG}'
TRAIN_ANALYSIS_CHECKPOINT_DIR='${TRAIN_ANALYSIS_CHECKPOINT_DIR}'
TRAIN_ANALYSIS_MAX_BATCHES='${TRAIN_ANALYSIS_MAX_BATCHES}'
TRAIN_ANALYSIS_PRUNE_EPS='${TRAIN_ANALYSIS_PRUNE_EPS}'
TRAIN_ANALYSIS_PRESET='${TRAIN_ANALYSIS_PRESET}'
TRAIN_ANALYSIS_ITEMS='${TRAIN_ANALYSIS_ITEMS}'
TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS='${TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS}'
TRAIN_ANALYSIS_GENERATION_MAX_TOKENS='${TRAIN_ANALYSIS_GENERATION_MAX_TOKENS}'
TRAIN_ANALYSIS_GENERATION_TEMPERATURE='${TRAIN_ANALYSIS_GENERATION_TEMPERATURE}'
TRAIN_ANALYSIS_GENERATION_TOP_K='${TRAIN_ANALYSIS_GENERATION_TOP_K}'
TRANSITION_TOPK_QK='${TRANSITION_TOPK_QK}'
TRANSITION_TOPK_V='${TRANSITION_TOPK_V}'
TRANSITION_TOPK_RST='${TRANSITION_TOPK_RST}'
CAUSAL_MAX_PROMPTS='${CAUSAL_MAX_PROMPTS}'
OPERATOR_DATASET_ROOT='${OPERATOR_DATASET_ROOT}'
OPERATOR_ANALYSIS_PROFILE='${OPERATOR_ANALYSIS_PROFILE}'
OPERATOR_DATASETS='${OPERATOR_DATASETS}'
OPERATOR_CACHE_DIR='${OPERATOR_CACHE_DIR}'
OPERATOR_BEHAVIOR_MAX_EXAMPLES='${OPERATOR_BEHAVIOR_MAX_EXAMPLES}'
OPERATOR_TRACE_MAX_EXAMPLES='${OPERATOR_TRACE_MAX_EXAMPLES}'
OPERATOR_CAUSAL_MAX_EXAMPLES='${OPERATOR_CAUSAL_MAX_EXAMPLES}'
OPERATOR_TRACE_PER_GROUP='${OPERATOR_TRACE_PER_GROUP}'
OPERATOR_CAUSAL_PER_GROUP='${OPERATOR_CAUSAL_PER_GROUP}'
OPERATOR_ANALYSIS_RESUME='${OPERATOR_ANALYSIS_RESUME}'
OPERATOR_ANALYSIS_SEED='${OPERATOR_ANALYSIS_SEED}'
SKIP_REPO_UPDATE='${SKIP_REPO_UPDATE}'
SYNC_LOCAL_ANALYSIS='${SYNC_LOCAL_ANALYSIS}'
DETACH='${DETACH}'
INSTALL_DEPS='${INSTALL_DEPS}'
TMUX_SESSION='${TMUX_SESSION}'
REMOTE_LOG='${REMOTE_LOG}'
APPEND_REMOTE_LOG='${APPEND_REMOTE_LOG}'
REMOTE_LOG_PATH="\${REMOTE_LOG/#\\~/\$HOME}"
WORK_DIR="\$HOME/DAWN-SRW"

echo "=== DAWN-SRW analysis worker startup ==="
echo "HOSTNAME=\$(hostname)"
echo "DATE=\$(date -Is)"
echo "BRANCH=\$BRANCH"
echo "MODE=\$MODE"
echo "CHECKPOINT=\$CHECKPOINT"
echo "OUTPUT=\$OUTPUT"
echo "STAGES=\$STAGES"
echo "ANALYSIS_ARGS=\$ANALYSIS_ARGS"
if [ "\$MODE" = "train_analysis" ]; then
    echo "TRAIN_ANALYSIS_CONFIG=\${TRAIN_ANALYSIS_CONFIG:-checkpoint full_config}"
    echo "TRAIN_ANALYSIS_CHECKPOINT_DIR=\$TRAIN_ANALYSIS_CHECKPOINT_DIR"
    echo "TRAIN_ANALYSIS_MAX_BATCHES=\$TRAIN_ANALYSIS_MAX_BATCHES"
    echo "TRAIN_ANALYSIS_PRUNE_EPS=\$TRAIN_ANALYSIS_PRUNE_EPS"
    echo "TRAIN_ANALYSIS_PRESET=\$TRAIN_ANALYSIS_PRESET"
    echo "TRAIN_ANALYSIS_ITEMS=\${TRAIN_ANALYSIS_ITEMS:-preset default}"
    echo "TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS=\$TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS"
    echo "TRAIN_ANALYSIS_GENERATION_MAX_TOKENS=\$TRAIN_ANALYSIS_GENERATION_MAX_TOKENS"
    echo "TRAIN_ANALYSIS_GENERATION_TEMPERATURE=\$TRAIN_ANALYSIS_GENERATION_TEMPERATURE"
    echo "TRAIN_ANALYSIS_GENERATION_TOP_K=\$TRAIN_ANALYSIS_GENERATION_TOP_K"
    echo "OPERATOR_DATASET_ROOT=\$OPERATOR_DATASET_ROOT"
    echo "OPERATOR_ANALYSIS_PROFILE=\$OPERATOR_ANALYSIS_PROFILE"
    echo "OPERATOR_DATASETS=\$OPERATOR_DATASETS"
    echo "OPERATOR_ANALYSIS_RESUME=\$OPERATOR_ANALYSIS_RESUME"
    echo "APPEND_REMOTE_LOG=\$APPEND_REMOTE_LOG"
fi

if [ "\$SKIP_REPO_UPDATE" = "1" ]; then
    if [ ! -d "\$WORK_DIR/.git" ]; then
        echo "ERROR: --skip-repo-update requires \$WORK_DIR/.git" >&2
        exit 1
    fi
    cd "\$WORK_DIR"
    echo "Repo update skipped; using synced worktree"
elif [ -d "\$WORK_DIR/.git" ]; then
    cd "\$WORK_DIR"
    git fetch origin "\$BRANCH" --depth 1
    git checkout -B "\$BRANCH" FETCH_HEAD
    echo "Repo updated to \$BRANCH"
else
    rm -rf "\$WORK_DIR"
    git clone -b "\$BRANCH" --single-branch --depth 1 "\$REPO_URL" "\$WORK_DIR"
    cd "\$WORK_DIR"
    echo "Repo cloned (branch: \$BRANCH)"
fi

if [ "\$SYNC_LOCAL_ANALYSIS" = "1" ]; then
    OVERLAY_PATH="\$HOME/dawn_v417x_analysis_overlay.tar.gz"
    if [ ! -f "\$OVERLAY_PATH" ]; then
        echo "ERROR: local analysis overlay missing: \$OVERLAY_PATH" >&2
        exit 1
    fi
    tar -xzf "\$OVERLAY_PATH" -C "\$WORK_DIR"
    rm -f "\$OVERLAY_PATH"
    echo "Local analysis overlay applied"
fi

if [ "\$INSTALL_DEPS" = "1" ]; then
    echo "[setup] installing TPU analysis dependencies"
    python3 -m pip install --upgrade pip -q
    python3 -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
    python3 -m pip install flax optax orbax-checkpoint==0.11.24 numpy pyyaml gcsfs transformers matplotlib -q
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')" >/dev/null 2>&1 || true
fi

export PYTHONUNBUFFERED=1
export DAWN_ANALYSIS_INIT_DISTRIBUTED=1
export JAX_TRACEBACK_FILTERING="\${JAX_TRACEBACK_FILTERING:-auto}"
export JAX_LOG_COMPILES="\${JAX_LOG_COMPILES:-0}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"

if [ "\$MODE" = "train_analysis" ]; then
    ANALYSIS_CMD=(
        python3 -u scripts/analyze_dawn_srw_v4166.py
        --train-analysis
        --checkpoint-dir "\$TRAIN_ANALYSIS_CHECKPOINT_DIR"
        --output "\$OUTPUT"
        --train-analysis-max-batches "\$TRAIN_ANALYSIS_MAX_BATCHES"
        --prune-eps "\$TRAIN_ANALYSIS_PRUNE_EPS"
        --train-analysis-preset "\$TRAIN_ANALYSIS_PRESET"
        --train-analysis-generation-max-prompts "\$TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS"
        --train-analysis-generation-max-tokens "\$TRAIN_ANALYSIS_GENERATION_MAX_TOKENS"
        --train-analysis-generation-temperature "\$TRAIN_ANALYSIS_GENERATION_TEMPERATURE"
        --train-analysis-generation-top-k "\$TRAIN_ANALYSIS_GENERATION_TOP_K"
        --transition-topk-qk "\$TRANSITION_TOPK_QK"
        --transition-topk-v "\$TRANSITION_TOPK_V"
        --transition-topk-rst "\$TRANSITION_TOPK_RST"
        --causal-max-prompts "\$CAUSAL_MAX_PROMPTS"
        --operator-dataset-root "\$OPERATOR_DATASET_ROOT"
        --operator-analysis-profile "\$OPERATOR_ANALYSIS_PROFILE"
        --operator-datasets "\$OPERATOR_DATASETS"
        --operator-cache-dir "\$OPERATOR_CACHE_DIR"
        --operator-analysis-seed "\$OPERATOR_ANALYSIS_SEED"
        --init-distributed
    )
    if [ -n "\$TRAIN_ANALYSIS_CONFIG" ]; then
        ANALYSIS_CMD+=(--config "\$TRAIN_ANALYSIS_CONFIG")
    fi
    if [ -n "\$TRAIN_ANALYSIS_ITEMS" ]; then
        ANALYSIS_CMD+=(--train-analysis-items "\$TRAIN_ANALYSIS_ITEMS")
    fi
    if [ "\$OPERATOR_ANALYSIS_RESUME" = "1" ]; then
        ANALYSIS_CMD+=(--operator-analysis-resume)
    else
        ANALYSIS_CMD+=(--no-operator-analysis-resume)
    fi
    [ -z "\$OPERATOR_BEHAVIOR_MAX_EXAMPLES" ] || ANALYSIS_CMD+=(--operator-behavior-max-examples "\$OPERATOR_BEHAVIOR_MAX_EXAMPLES")
    [ -z "\$OPERATOR_TRACE_MAX_EXAMPLES" ] || ANALYSIS_CMD+=(--operator-trace-max-examples "\$OPERATOR_TRACE_MAX_EXAMPLES")
    [ -z "\$OPERATOR_CAUSAL_MAX_EXAMPLES" ] || ANALYSIS_CMD+=(--operator-causal-max-examples "\$OPERATOR_CAUSAL_MAX_EXAMPLES")
    [ -z "\$OPERATOR_TRACE_PER_GROUP" ] || ANALYSIS_CMD+=(--operator-trace-per-group "\$OPERATOR_TRACE_PER_GROUP")
    [ -z "\$OPERATOR_CAUSAL_PER_GROUP" ] || ANALYSIS_CMD+=(--operator-causal-per-group "\$OPERATOR_CAUSAL_PER_GROUP")
else
    ANALYSIS_CMD=(
        python3 -u scripts/analyze_dawn_srw_v4166.py
        --checkpoint "\$CHECKPOINT"
        --output "\$OUTPUT"
        --stages "\$STAGES"
        --init-distributed
    )
fi
if [ -n "\$ANALYSIS_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(\$ANALYSIS_ARGS)
    ANALYSIS_CMD+=("\${EXTRA_ARGS[@]}")
fi
ANALYSIS_CMD_STR=\$(printf "%q " "\${ANALYSIS_CMD[@]}")

cd "\$WORK_DIR"
mkdir -p "\$(dirname "\$REMOTE_LOG_PATH")"
TEE_MODE=""
if [ "\$APPEND_REMOTE_LOG" = "1" ]; then
    touch "\$REMOTE_LOG_PATH"
    TEE_MODE="-a"
else
    : > "\$REMOTE_LOG_PATH"
fi
if [ "\$DETACH" = "1" ]; then
    echo "[run] starting tmux session \$TMUX_SESSION"
    tmux kill-session -t "\$TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -x 240 -y 60 -s "\$TMUX_SESSION" \
        "cd '\$WORK_DIR'; export PYTHONUNBUFFERED=1; export DAWN_ANALYSIS_INIT_DISTRIBUTED=1; export JAX_TRACEBACK_FILTERING='\$JAX_TRACEBACK_FILTERING'; export JAX_LOG_COMPILES='\$JAX_LOG_COMPILES'; export TF_CPP_MIN_LOG_LEVEL='\$TF_CPP_MIN_LOG_LEVEL'; { echo '=== TPU analysis process startup ==='; echo \"HOSTNAME=\$(hostname)\"; echo \"DATE=\$(date -Is)\"; echo \"CMD: \$ANALYSIS_CMD_STR\"; \$ANALYSIS_CMD_STR; } 2>&1 | tee \$TEE_MODE '\$REMOTE_LOG_PATH'"
    echo "[run] detached in tmux session \$TMUX_SESSION, log=\$REMOTE_LOG_PATH"
else
    echo "[run] foreground analysis"
    "\${ANALYSIS_CMD[@]}" 2>&1 | tee \$TEE_MODE "\$REMOTE_LOG_PATH"
fi
EOFCMD

echo "Sending analysis command to target worker(s): ${TARGET_WORKERS[*]}"
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
declare -a LAUNCH_PIDS=()
declare -a LAUNCH_LOGS=()
for worker in "${TARGET_WORKERS[@]}"; do
    log_file="launch_dawn_srw_analysis_${TPU_NAME}_${LAUNCH_TS}_worker_${worker}.log"
    LAUNCH_LOGS[$worker]="$log_file"
    echo "  Worker $worker launch starting (log: $log_file)"
    (
        run_worker_command "$worker" "$REMOTE_CMD"
    ) >"$log_file" 2>&1 &
    LAUNCH_PIDS[$worker]=$!
done

declare -a FAILED_WORKERS=()
for worker in "${TARGET_WORKERS[@]}"; do
    if ! wait "${LAUNCH_PIDS[$worker]}"; then
        FAILED_WORKERS+=("$worker")
        echo "ERROR: worker $worker setup/start failed. See ${LAUNCH_LOGS[$worker]}" >&2
    fi
done

if [ "${#FAILED_WORKERS[@]}" -gt 0 ]; then
    echo "ERROR: launch failed on worker(s): ${FAILED_WORKERS[*]}. Cleaning up target workers." >&2
    cleanup_target_workers || true
    exit 1
fi

echo ""
if [[ "$MODE" == "train_analysis" && "$DETACH" == "0" ]]; then
    SUMMARY_LOG=""
    SUMMARY_WORKER=""
    for worker in "${TARGET_WORKERS[@]}"; do
        log_file="${LAUNCH_LOGS[$worker]}"
        if grep -q "DAWN-SRW .* TRAIN ANALYSIS" "$log_file"; then
            SUMMARY_LOG="$log_file"
            SUMMARY_WORKER="$worker"
            break
        fi
    done
    echo "Train analysis complete."
    if [[ -n "$SUMMARY_LOG" ]]; then
        echo "  Summary worker : $SUMMARY_WORKER"
        echo "  Summary log    : $SUMMARY_LOG"
        echo ""
        awk 'BEGIN{show=0} /^============================================================$/ {show=1} show {print}' "$SUMMARY_LOG"
    else
        echo "WARNING: train_analysis summary block was not found in local launch logs." >&2
        echo "Local launch logs:"
        for worker in "${TARGET_WORKERS[@]}"; do
            echo "  worker $worker: ${LAUNCH_LOGS[$worker]}"
        done
    fi
    echo ""
    echo "Copy-paste:"
    echo "  $COPY_CMD"
    echo ""
    echo "Watch logs:"
    echo "  $WATCH_LOG_CMD"
    exit 0
fi

echo "Launch complete. Analysis is running in tmux session '$TMUX_SESSION' on target workers."
echo "  Primary log:     bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT"
echo "  Primary pane:    bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --pane"
echo "  Attach primary:  bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --attach"
echo "  Primary summary: bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --summary"
echo "  All summary:     bash scripts/watch_tpu_logs.sh --tpu $TPU_NAME --zone $ZONE --project $PROJECT --all --summary"
echo "  Literal worker 0 log: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tail -f ~/train.log'"
echo "  Attach literal worker 0: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='tmux attach -t train'"
echo "  Kill:    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all --command='tmux kill-session -t train'"
echo "  Your capture flow works on whichever worker you attach: tmux pipe-pane -t train 'cat >> ~/rebuttal_log.txt'"
