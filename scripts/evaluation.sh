#!/bin/bash
set -eo pipefail

################################################################################
# Evaluate a trained Text-to-SQL model on Spider dev.
#
# Required parameters:
#   MODEL: model path or HF repo id to evaluate
#   ROUND_NAME: short tag used in the results directory
#   DATASET: dataset tag used in the results directory
#   COUNT: number of dev examples (0 = full dev split)
#   DETAILED: true|false, whether answer.json keeps pred_text
#   VARIANT: direct | sketch (chooses the prompt mode via configs/eval_default.yaml)
################################################################################

# Model to evaluate (path to SFT/GRPO final checkpoint or HF repo id).
MODEL="runs/sft_baseline_.../final"

# Short identifier used in the results directory name.
ROUND_NAME="eval_sft_baseline"

# Dataset tag (e.g. spider, spider_dev_sub).
DATASET="spider"

# Number of dev examples. 0 = full dev split.
COUNT=0

# Output verbosity in answer.json.
DETAILED="true"

# Prompt mode. Options: direct | sketch
VARIANT="direct"


################################################################################
# Run

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
PKG="$(basename "$REPO_ROOT")"
cd "$REPO_ROOT"

CONFIG="configs/eval_default.yaml"

python -m "${PKG}.evaluation.eval_pipeline" \
    --config "$CONFIG" \
    --model "$MODEL" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --detailed "$DETAILED"

echo "[evaluation] finished variant=$VARIANT model=$MODEL dataset=$DATASET count=$COUNT detailed=$DETAILED"
