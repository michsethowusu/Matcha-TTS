#!/usr/bin/env bash
# One-time environment setup for finetuning on the Ghana NLP H200 GPU.
# Run on the H200 in /mnt/volume_d2wey28/projects/matcha-twi.
set -euo pipefail

cd /mnt/volume_d2wey28/projects/matcha-twi

# The stock venv may ship torch compiled for a newer CUDA driver than the host's 12.8.
# Re-install PyTorch for CUDA 12.8 so the H200 is usable.
echo "[setup] installing CUDA 12.8 PyTorch wheels..."
pip install --upgrade \
    torch==2.7.1+cu128 \
    torchaudio==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Vocos vocoder + progress bars for data prep.
echo "[setup] installing vocos and helpers..."
pip install vocos tqdm

# The local package is used via PYTHONPATH in the run scripts (editable install fails
# on Python 3.12 because setup.py pins old numpy/cython build deps).
echo "[setup] verifying imports and GPU..."
export PYTHONPATH=/mnt/volume_d2wey28/projects/matcha-twi:$PYTHONPATH
python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
PY

echo "[setup] done"
