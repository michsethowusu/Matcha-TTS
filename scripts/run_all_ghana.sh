#!/usr/bin/env bash
# End-to-end multilingual finetuning on ghananlpcommunity/ghana-speech.
# Usage:
#   scripts/run_all_ghana.sh /path/to/checkpoint.ckpt
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
    echo "[run_all] HF token set; manifests will be pushed and checkpoints mirrored to HF if repo_id configured."
fi

echo "[run_all] === Stage 1/3: data prep ==="
MAX_ATTEMPTS=5
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "[run_all] data prep attempt $attempt/$MAX_ATTEMPTS"
    if scripts/run_data_prep_ghana.sh; then
        echo "[run_all] data prep succeeded"
        break
    fi
    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
        echo "[run_all] data prep FAILED after $MAX_ATTEMPTS attempts"
        exit 1
    fi
    echo "[run_all] data prep failed, retrying in 120s..."
    sleep 120
done

echo "[run_all] === Stage 2/3: Matcha-TTS finetuning ==="
scripts/run_matcha_finetune_ghana.sh "$FINETUNE_CKPT"

echo "[run_all] === Stage 3/3: Vocos vocoder finetuning ==="
scripts/run_vocos_finetune_ghana.sh

echo "[run_all] ALL DONE"
