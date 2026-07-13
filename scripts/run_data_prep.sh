#!/usr/bin/env bash
# Prepare ghananlpcommunity/new-twi-tts-aligned for Matcha-TTS + Vocos finetuning.
# Run on the H200 in /mnt/volume_d2wey28/projects/matcha-twi.
set -euo pipefail

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate

export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH

DATA_DIR=/mnt/volume_d2wey28/data/twi_new

echo "[prep] streaming, filtering, resampling, phonemizing..."
python scripts/prep_new_twi_data.py prep --data-dir "$DATA_DIR"

echo "[prep] building train/val filelists..."
python scripts/prep_new_twi_data.py filelists --data-dir "$DATA_DIR" --val-size 2048

echo "[prep] precomputing mel spectrograms..."
python scripts/prep_new_twi_data.py mels --data-dir "$DATA_DIR" --num-workers 16

echo "[prep] computing mel statistics..."
python scripts/prep_new_twi_data.py stats --data-dir "$DATA_DIR" --num-workers 16

echo "[prep] done. Data is in $DATA_DIR"
