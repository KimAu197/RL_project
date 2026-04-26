#!/bin/bash
set -eo pipefail

################################################################################
# SFT training for one of the two prompt variants.
#
# Required parameters:
#   MODEL: base model path or HF repo id (overrides configs/base.yaml)
#   ROUND_NAME: short tag used in the output directory
#   DATASET: dataset tag used in the output directory (spider by default)
#   COUNT: max training examples (0 = full train split)
#   DETAILED: true|false, kept for parity with evaluation script
#   VARIANT: direct | sketch (chooses configs/sft_<variant>.yaml)
################################################################################

# Base model (HF repo id or local path). Set this before running.
MODEL="Qwen/Qwen2.5-7B-Instruct"

# Short identifier used in the run directory name.
ROUND_NAME="sft_baseline"

# Dataset tag used in the output directory (e.g. spider, spider_sub).
DATASET="spider"

# Number of training examples. 0 = full split.
COUNT=0

# Whether to write detailed per-example outputs. Matches the evaluation script.
DETAILED="true"

# Upload Trainer metrics to Weights & Biases. Requires `wandb login` first.
WANDB_ENABLED="false"
WANDB_PROJECT="sketch-to-sql-rl"
WANDB_ENTITY=""

# Which variant to train. Options: direct | sketch
VARIANT="direct"


################################################################################
# Run

cd "$(dirname "$0")/.."
PROJECT_NAME="$(basename "$PWD")"
PROJECT_PARENT="$(dirname "$PWD")"
export PYTHONPATH="${PROJECT_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="configs/sft_${VARIANT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

mkdir -p runs

CMD=(python -m "${PROJECT_NAME}.training.sft_train"
     --config "$CONFIG"
     --model "$MODEL"
     --round "$ROUND_NAME"
     --wandb-enabled "$WANDB_ENABLED"
     --wandb-project "$WANDB_PROJECT")
if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ "$COUNT" != "0" ]]; then
    CMD+=(--max-train "$COUNT")
fi

"${CMD[@]}" 2>&1 | tee -a "runs/${ROUND_NAME}_launch.log"

echo "[train_sft] finished variant=$VARIANT model=$MODEL dataset=$DATASET count=$COUNT detailed=$DETAILED"
