#!/bin/bash
set -eo pipefail

################################################################################
# Difficulty Filtering Settings

# SFT model to sample from before GRPO.
# Use the matching SFT checkpoint for the selected VARIANT.
MODEL="runs/sft_baseline_.../final"

# Prompt mode. Options: direct | sketch
VARIANT="sketch"

# Dataset tag used for bookkeeping in output paths.
DATASET="spider"

# Number of Spider train examples to scan (0 = full train split).
COUNT=0

# Number of sampled completions per prompt.
SAMPLES=8

# Number of sampled completions requested in each model.generate() call.
# Keep this at 1 if CUDA is unstable; total samples still equals SAMPLES.
SAMPLES_PER_CALL=1

# Prompts per generation batch. Actual generated sequences per step are
# BATCH_SIZE * SAMPLES_PER_CALL, so keep this small for 7B models.
BATCH_SIZE=1

# Sampling settings for estimating pass rate. These should be stochastic,
# typically matching GRPO generation settings.
TEMPERATURE=0.7
TOP_P=0.95

# Keep examples whose sampled execution pass rate falls in this range.
MIN_PASS_RATE=0.125
MAX_PASS_RATE=0.875

# Basic format quality filters.
MIN_HAS_SQL_RATE=0.5
MIN_VALID_SQL_RATE=0.0

# Output directory for filtered JSONL and difficulty statistics.
OUTPUT_DIR="datasets/spider/processed/difficulty_filter_${VARIANT}_${SAMPLES}x"


################################################################################
# Run

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$(dirname "$REPO_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
PKG="$(basename "$REPO_ROOT")"
cd "$REPO_ROOT"

CONFIG="configs/grpo_${VARIANT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

python -m "${PKG}.data.difficulty_filter" \
    --config "$CONFIG" \
    --model "$MODEL" \
    --variant "$VARIANT" \
    --count "$COUNT" \
    --samples "$SAMPLES" \
    --samples-per-call "$SAMPLES_PER_CALL" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --min-pass-rate "$MIN_PASS_RATE" \
    --max-pass-rate "$MAX_PASS_RATE" \
    --min-has-sql-rate "$MIN_HAS_SQL_RATE" \
    --min-valid-sql-rate "$MIN_VALID_SQL_RATE" \
    --output-dir "$OUTPUT_DIR"

echo "[filter_difficulty] finished variant=$VARIANT model=$MODEL dataset=$DATASET count=$COUNT samples=$SAMPLES samples_per_call=$SAMPLES_PER_CALL batch_size=$BATCH_SIZE temperature=$TEMPERATURE output=$OUTPUT_DIR"
