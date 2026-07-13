#!/usr/bin/env bash
# Finetune Matcha-TTS from a previous Twi checkpoint on ghananlpcommunity/ghana-speech.
# Usage:
#   export HF_TOKEN=hf_xxx
#   export HF_REPO_ID=your/repo
#   scripts/run_matcha_finetune_ghana.sh /path/to/nano_twi_045.ckpt
set -euo pipefail

FINETUNE_CKPT="${1:-}"
if [ -z "$FINETUNE_CKPT" ]; then
    echo "Usage: $0 <path-to-checkpoint>"
    exit 1
fi

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate
export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

STATS=$(cat /mnt/volume_d2wey28/data/ghana_speech/ghana_speech.json)
MEAN=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_mean'])")
STD=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_std'])")

echo "[matcha-ft] finetuning from $FINETUNE_CKPT"
echo "[matcha-ft] data stats: mean=$MEAN std=$STD"

python matcha/train.py \
    experiment=ghana_speech \
    finetune_ckpt="$FINETUNE_CKPT" \
    data.data_statistics.mel_mean="$MEAN" \
    data.data_statistics.mel_std="$STD" \
    data.batch_size=64 \
    data.num_workers=8 \
    trainer.devices=[0] \
    trainer.precision=bf16-mixed \
    test=false

echo "[matcha-ft] done"
