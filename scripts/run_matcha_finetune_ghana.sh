#!/usr/bin/env bash
# Finetune Matcha-TTS from a previous Twi checkpoint on ghananlpcommunity/ghana-speech.
# Uses filtered filelists (1-10s, 10h/lang cap).
# Usage:
#   scripts/run_matcha_finetune_ghana.sh /path/to/nano_twi_045.ckpt
set -euo pipefail

FINETUNE_CKPT="${1:-}"
if [ -z "$FINETUNE_CKPT" ]; then
    echo "Usage: $0 <path-to-checkpoint>"
    exit 1
fi

cd /mnt/volume_d2wey28/projects/matcha-twi
source .venv/bin/activate
export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:${PYTHONPATH:-}

# Push each epoch's checkpoint to this HF model repo (HF_TOKEN must be set in the
# environment; it is intentionally not stored in this script).
export HF_REPO_ID="${HF_REPO_ID:-ghananlpcommunity/ghana-speech-nano}"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[matcha-ft] WARNING: HF_TOKEN not set — checkpoints will NOT be pushed to the Hub."
fi

STATS=$(cat /mnt/volume_d2wey28/data/ghana_speech/ghana_speech_filtered.json)
MEAN=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_mean'])")
STD=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['mel_std'])")
N_TRAIN=$(echo "$STATS" | python -c "import sys,json; print(json.load(sys.stdin)['n_train'])")

echo "[matcha-ft] finetuning from $FINETUNE_CKPT"
echo "[matcha-ft] filtered data: $N_TRAIN train clips, mean=$MEAN std=$STD"

# Optional full resume: set RESUME_CKPT to a Lightning checkpoint (e.g. last.ckpt) to
# restore optimizer/epoch state via ckpt_path. When set, finetune_ckpt is ignored by
# train.py (it only weight-loads when ckpt_path is absent).
RESUME_ARG=()
if [ -n "${RESUME_CKPT:-}" ]; then
    echo "[matcha-ft] RESUMING training from $RESUME_CKPT"
    RESUME_ARG=(ckpt_path="$RESUME_CKPT")
fi

python matcha/train.py \
    experiment=ghana_speech \
    finetune_ckpt="$FINETUNE_CKPT" \
    data.data_statistics.mel_mean="$MEAN" \
    data.data_statistics.mel_std="$STD" \
    data.batch_size=64 \
    data.num_workers=16 \
    trainer.devices=[0] \
    trainer.precision=bf16-mixed \
    test=false \
    "${RESUME_ARG[@]}"

echo "[matcha-ft] done"
