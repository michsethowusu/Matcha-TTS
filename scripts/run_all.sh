#!/usr/bin/env bash
# End-to-end finetuning on ghananlpcommunity/new-twi-tts-aligned.
# Usage:
#   scripts/run_all.sh /path/to/checkpoint.ckpt
#
# Runs:
#   1. data prep (resumable)
#   2. Matcha-TTS finetuning with early stopping (patience=5)
#   3. Vocos vocoder finetuning with early stopping (patience=5)
set -euo pipefail

FINETUNE_CKPT="${1:-}"
if [ -z "$FINETUNE_CKPT" ]; then
    echo "Usage: $0 <path-to-checkpoint>"
    exit 1
fi

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate
export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

if [ -n "${HF_TOKEN:-}" ]; then
    echo "[run_all] HF token set; checkpoints will be pushed to HF if repo_id is configured."
fi

echo "[run_all] === Stage 1/3: data prep ==="
scripts/run_data_prep.sh

echo "[run_all] === Stage 2/3: Matcha-TTS finetuning ==="
scripts/run_matcha_finetune.sh "$FINETUNE_CKPT"

echo "[run_all] === Stage 3/3: Vocos vocoder finetuning ==="
scripts/run_vocos_finetune.sh

echo "[run_all] ALL DONE"
