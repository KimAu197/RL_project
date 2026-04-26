#!/bin/bash
set -eo pipefail

################################################################################
# GRPO post-training for one of the two prompt variants. Expects an SFT
# checkpoint as the starting point.
#
# Required parameters:
#   MODEL: path to the SFT checkpoint (the GRPO starting point)
#   ROUND_NAME: short tag used in the output directory
#   DATASET: dataset tag used in the output directory
#   COUNT: max training examples (0 = full train split)
#   DETAILED: true|false
#   VARIANT: direct | sketch (chooses configs/grpo_<variant>.yaml)
################################################################################

# Starting SFT checkpoint (produced by scripts/train_sft.sh). Must be the
# concrete path to the `final/` directory — no wildcards.
MODEL="runs/sft_baseline_.../final"

# Short identifier used in the run directory name.
ROUND_NAME="grpo_baseline"

# Dataset tag used in the output directory.
DATASET="spider"

# Number of training examples. 0 = full split.
COUNT=0

# Whether to write detailed outputs (parity with evaluation script).
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

CONFIG="configs/grpo_${VARIANT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

mkdir -p runs

CMD=(python -m "${PROJECT_NAME}.training.grpo_train"
     --config "$CONFIG"
     --round "$ROUND_NAME"
     --init-ckpt "$MODEL"
     --wandb-enabled "$WANDB_ENABLED"
     --wandb-project "$WANDB_PROJECT")
if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ "$COUNT" != "0" ]]; then
    CMD+=(--max-train "$COUNT")
fi

"${CMD[@]}" 2>&1 | tee -a "runs/${ROUND_NAME}_launch.log"

echo "[train_grpo] finished variant=$VARIANT model=$MODEL dataset=$DATASET count=$COUNT detailed=$DETAILED"
