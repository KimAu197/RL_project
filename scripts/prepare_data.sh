#!/bin/bash
set -eo pipefail

################################################################################
# Prepare Spider for Sketch-to-SQL training.
#
# Steps:
#   1. Download + unzip Spider (skipped if already present).
#   2. Materialize processed JSONL files for every (split, mode) pair.
#
# Required parameters:
#   SPIDER_ROOT: path where Spider will be stored
#   PROCESSED_DIR: where processed JSONLs are written
################################################################################

# Spider dataset root (will be created if missing)
SPIDER_ROOT="datasets/spider"

# Processed JSONL directory (used by SFT / GRPO / eval)
PROCESSED_DIR="datasets/spider/processed"

# Max characters per schema injected into prompts
MAX_SCHEMA_CHARS=4000


################################################################################
# Run

cd "$(dirname "$0")/.."

python -m project.data.download_spider --dest "$(dirname "$SPIDER_ROOT")"

python -m project.data.prepare \
    --spider-root "$SPIDER_ROOT" \
    --out "$PROCESSED_DIR" \
    --max-schema-chars "$MAX_SCHEMA_CHARS"
