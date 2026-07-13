#!/usr/bin/env bash
# Finetune Matcha-TTS from a previous Twi checkpoint on ghananlpcommunity/new-twi-tts-aligned.
# Usage:
#   export HF_TOKEN=hf_xxx          # optional: only if you want HF checkpoint uploads
#   export HF_REPO_ID=your/repo     # optional: defaults to config value
#   scripts/run_matcha_finetune.sh /path/to/twi_045.ckpt
#
# Training runs until convergence or 500 epochs, with EarlyStopping patience=5.
set -euo pipefail

FINETUNE_CKPT="${1:-}"
if [ -z "$FINETUNE_CKPT" ]; then
    echo "Usage: $0 <path-to-checkpoint>"
    exit 1
fi

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate

# Load the computed statistics for the new dataset.
STATS=$(cat /mnt/volume_d2wey28/data/twi_new/twi_new.json)
MEAN=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_mean'])")
STD=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_std'])")

echo "[matcha-ft] finetuning from $FINETUNE_CKPT"
echo "[matcha-ft] data stats: mean=$MEAN std=$STD"

export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

python matcha/train.py \
    experiment=twi_new \
    finetune_ckpt="$FINETUNE_CKPT" \
    data.data_statistics.mel_mean="$MEAN" \
    data.data_statistics.mel_std="$STD" \
    data.batch_size=64 \
    data.num_workers=8 \
    trainer.devices=[0] \
    trainer.precision=bf16-mixed \
    test=false

echo "[matcha-ft] done"
