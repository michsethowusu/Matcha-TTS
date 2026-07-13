#!/usr/bin/env bash
# Prepare ghananlpcommunity/ghana-speech for multilingual Matcha-TTS + Vocos finetuning.
# Run on the H200 in /mnt/volume_d2wey28/projects/matcha-twi.
set -euo pipefail

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate
export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

DATA_DIR=/mnt/volume_d2wey28/data/ghana_speech

echo "[prep-ghana] preparing ghana-speech -> $DATA_DIR"
python scripts/prep_ghana_speech.py \
    --data-dir "$DATA_DIR" \
    --val-per-lang 32 \
    --num-workers 16 \
    --phonemize-batch 256 \
    --push-manifest-repo "ghananlpcommunity/ghana-speech-lfn-phonemized" \
    --push-manifest-private

echo "[prep-ghana] done. Data is in $DATA_DIR"
