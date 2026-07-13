#!/usr/bin/env bash
# Finetune the Vocos vocoder on the new Twi dataset.
# Runs until convergence with EarlyStopping patience=5.
set -euo pipefail

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate

export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

DATA_DIR=/mnt/volume_d2wey28/data/twi_new
OUT_DIR=/mnt/volume_d2wey28/projects/matcha-twi/outputs/vocos_twi_new

echo "[vocos-ft] finetuning Vocos on $DATA_DIR -> $OUT_DIR"

python -m matcha.vocos.train \
    --input-wavs-dir "$DATA_DIR/wavs" \
    --input-mels-dir "$DATA_DIR/mels" \
    --stats-json "$DATA_DIR/twi_new.json" \
    --train-filelist "$DATA_DIR/train.txt" \
    --val-filelist "$DATA_DIR/val.txt" \
    --checkpoint-path "$OUT_DIR" \
    --pretrained BSC-LT/vocos-mel-22khz \
    --batch-size 32 \
    --segment-size 172 \
    --learning-rate 2e-4 \
    --max-epochs 500 \
    --patience 5 \
    --pretrain-mel-epochs 1 \
    --num-workers 8 \
    --resume

echo "[vocos-ft] done"
