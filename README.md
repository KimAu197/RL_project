# Sketch-to-SQL with Execution-Guided RL for Small Language Models

Implementation scaffold of the course project proposal: studying whether an
intermediate sketch representation improves RL-based post-training for small
Text-to-SQL models.

Four systems are supported end-to-end in the same codebase and are selected
purely by YAML config:

| Variant | Description |
|---|---|
| `sft_direct` | SFT, model generates SQL directly from question + schema |
| `sft_sketch` | SFT, model generates sketch -> SQL |
| `grpo_direct` | GRPO RL on top of `sft_direct`, reward = execution match |
| `grpo_sketch` | GRPO RL on top of `sft_sketch`, reward = execution match |

## Layout

```
project/
  configs/        YAML configs for every variant, driven by base.yaml
  data/           Spider downloader, sketch extractor, schema linker, prompt builder
  sql/            SQLite execution, SQL validation, execution-based metrics
  models/         Model / tokenizer loading (incl. optional LoRA + 4-bit)
  training/       SFT and GRPO training loops (TRL) and reward functions
  evaluation/     Evaluation pipeline producing logging.log / metrics.csv / summary.txt / answer.json
  scripts/        Shell runners with standard parameters
```

All heavy settings (base model, LoRA, batch size, max lengths, rollouts per
prompt, reward shaping, etc.) live in YAML so the same code runs on a laptop,
a single GPU, or a multi-GPU box.

## Install

```bash
pip install -r requirements.txt
```

`bitsandbytes` is only needed if you enable 4-bit loading. Pin CUDA-matching
`torch` / `bitsandbytes` wheels separately on GPU boxes.

## Prepare Spider

```bash
bash scripts/prepare_data.sh
```

This downloads the Spider dataset (if not cached), extracts sketches from every
gold SQL using `sqlglot`, and writes JSONL files under `datasets/spider/processed/`.

## Train (SFT)

```bash
bash scripts/train_sft.sh        # uses configs/sft_direct.yaml by default
```

Edit the top of `scripts/train_sft.sh` to switch `CONFIG` between `sft_direct.yaml`
and `sft_sketch.yaml`.

Training metrics can be uploaded to Weights & Biases by running `wandb login`
and setting `WANDB_ENABLED="true"` at the top of the training script. W&B is
disabled by default so local training works without an account.

## Train (GRPO)

```bash
bash scripts/train_grpo.sh
```

Point `INIT_CKPT` to the checkpoint produced by the matching SFT run
(`sft_direct` -> `grpo_direct`, `sft_sketch` -> `grpo_sketch`).

## Evaluate

```bash
bash scripts/evaluation.sh
```

Produces:

```
results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/
  logging.log
  metrics.csv
  summary.txt
  answer.json
  log/
```

## Notes

- The pipeline is framework-agnostic on compute: it runs on CPU/MPS for sanity
  checks, and on CUDA for real training. Enable LoRA + 4-bit in the config for
  tight VRAM budgets.
- The reward is intentionally simple (1.0 on execution match, else 0.0). A
  shaped reward variant is wired through `training/reward.py` for ablations on
  reward hacking, but disabled by default.
- Base model choice is intentionally not hard-coded. Set `model.name_or_path`
  in `configs/base.yaml`.
