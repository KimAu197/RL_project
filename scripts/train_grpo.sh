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

# Which variant to train. Options: direct | sketch
VARIANT="direct"


################################################################################
# Run

cd "$(dirname "$0")/.."

CONFIG="configs/grpo_${VARIANT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

mkdir -p runs

CMD=(python -m project.training.grpo_train
     --config "$CONFIG"
     --round "$ROUND_NAME"
     --init-ckpt "$MODEL")
if [[ "$COUNT" != "0" ]]; then
    CMD+=(--max-train "$COUNT")
fi

"${CMD[@]}" 2>&1 | tee -a "runs/${ROUND_NAME}_launch.log"

echo "[train_grpo] finished variant=$VARIANT model=$MODEL dataset=$DATASET count=$COUNT detailed=$DETAILED"
