#!/usr/bin/env python3
"""Push the filtered ghana-speech prepped data (mels + filelists + stats) to HuggingFace.

Packs precomputed mel spectrograms into Parquet shards so the entire training-ready
dataset can be downloaded from HF and reconstructed on any machine, even if the
original disk data is deleted.

Output HF dataset: gghananlpcommunity/ghana-speech-prepped-filtered
  - data/train-00000-of-NNNNN.parquet   (mel bytes + metadata for train split)
  - data/val-00000-of-NNNNN.parquet      (mel bytes + metadata for val split)
  - ghana_speech_filtered.json           (mel stats, per-lang stats)
  - lang_map.json                        (language name -> lang_id)
  - README.md                            (documentation + reconstruction instructions)

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/push_filtered_to_hf.py \
        --data-dir /mnt/volume_d2wey28/data/ghana_speech \
        --repo-id ghananlpcommunity/ghana-speech-prepped-filtered \
        --shard-size 5000
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo


def parse_filelist(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            wav_path, lang_id, phonemes = parts[0], int(parts[1]), parts[2]
            stem = Path(wav_path).stem
            rows.append((wav_path, lang_id, phonemes, stem))
    return rows


def build_shard(rows, mel_dir, shard_idx):
    """Build a Parquet table for one shard of rows."""
    ids = []
    lang_ids = []
    phonemes_list = []
    mel_shapes = []
    mel_bytes_list = []
    missing = 0

    for wav_path, lang_id, phonemes, stem in rows:
        mel_path = mel_dir / (stem + ".npy")
        try:
            mel = np.load(mel_path)
            buf = io.BytesIO()
            np.save(buf, mel)
            mel_bytes = buf.getvalue()
        except FileNotFoundError:
            missing += 1
            continue
        ids.append(stem)
        lang_ids.append(lang_id)
        phonemes_list.append(phonemes)
        mel_shapes.append(list(mel.shape))
        mel_bytes_list.append(mel_bytes)

    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "lang_id": pa.array(lang_ids, type=pa.int32()),
        "phonemes": pa.array(phonemes_list, type=pa.string()),
        "mel_shape": pa.array(mel_shapes, type=pa.list_(pa.int32())),
        "mel": pa.array(mel_bytes_list, type=pa.binary()),
    })
    return table, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="ghananlpcommunity/ghana-speech-prepped-filtered")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--local-cache", type=str, default="/mnt/volume_d2wey28/data/hf_push_cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mel_dir = data_dir / "mels"
    cache_dir = Path(args.local_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load filelists
    train_rows = parse_filelist(data_dir / "train_filtered.txt")
    val_rows = parse_filelist(data_dir / "val_filtered.txt")
    print(f"[push] Train: {len(train_rows)} clips, Val: {len(val_rows)} clips")

    # Create HF repo
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, token=os.environ.get("HF_TOKEN"))
    print(f"[push] HF repo: {args.repo_id}")

    # Build and upload shards
    all_files = []

    for split_name, rows in [("train", train_rows), ("val", val_rows)]:
        n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
        print(f"\n[push] Building {split_name} split: {len(rows)} clips in {n_shards} shards")

        total_missing = 0
        for shard_idx in range(n_shards):
            start = shard_idx * args.shard_size
            end = min(start + args.shard_size, len(rows))
            shard_rows = rows[start:end]

            t0 = time.time()
            table, missing = build_shard(shard_rows, mel_dir, shard_idx)
            total_missing += missing
            elapsed = time.time() - t0

            shard_file = cache_dir / f"{split_name}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
            pq.write_table(table, shard_file, compression="zstd")
            size_mb = shard_file.stat().st_size / 1e6
            print(f"[push]   {split_name} shard {shard_idx+1}/{n_shards}: "
                  f"{len(table)} clips, {size_mb:.1f} MB, {elapsed:.1f}s, missing={missing}")

            all_files.append((shard_file, f"data/{split_name}-{shard_idx:05d}-of-{n_shards:05d}.parquet"))

        print(f"[push] {split_name} total missing mels: {total_missing}")

    # Upload metadata files
    meta_files = [
        (data_dir / "ghana_speech_filtered.json", "ghana_speech_filtered.json"),
        (data_dir / "lang_map.json", "lang_map.json"),
    ]

    # Write README
    readme_path = cache_dir / "README.md"
    stats = json.loads((data_dir / "ghana_speech_filtered.json").read_text())
    n_train = stats["n_train"]
    n_val = stats["n_val"]
    total_hours = stats["total_train_hours"]
    mel_mean = stats["mel_mean"]
    mel_std = stats["mel_std"]

    readme_path.write_text(f"""# Ghana Speech Prepped (Filtered)

Precomputed mel spectrograms + phonemized text for multilingual Matcha-TTS training,
filtered to clips in [1.0, 10.0] seconds and capped at 10 hours per language.

## Stats

| Metric | Value |
|--------|-------|
| Train clips | {n_train:,} |
| Val clips | {n_val:,} |
| Total train audio | {total_hours:.1f} hours |
| Languages | 42 |
| Mel mean | {mel_mean} |
| Mel std | {mel_std} |
| Duration filter | [1.0, 10.0] seconds |
| Cap | 10 hours per language |

## Format

Each Parquet shard contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Clip identifier (wav stem) |
| `lang_id` | int32 | Language ID (0-41), repurposes speaker slot |
| `phonemes` | string | LFN phonemized text |
| `mel_shape` | list[int32] | Shape of the mel spectrogram |
| `mel` | binary | Raw numpy bytes of the (un-normalized) mel spectrogram |

## Reconstruction

```python
import io, json, numpy as np
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Download metadata
stats = json.loads(hf_hub_download("ghananlpcommunity/ghana-speech-prepped-filtered", "ghana_speech_filtered.json", repo_type="dataset"))
lang_map = json.loads(hf_hub_download("ghananlpcommunity/ghana-speech-prepped-filtered", "lang_map.json", repo_type="dataset"))

# Load parquet shards
ds = load_dataset("ghananlpcommunity/ghana-speech-prepped-filtered")

# Reconstruct mel directory + filelists
out_dir = Path("ghana_speech_data")
mels_dir = out_dir / "mels"
mels_dir.mkdir(parents=True, exist_ok=True)

train_lines, val_lines = [], []
for split, lines in [("train", train_lines), ("val", val_lines)]:
    for row in ds[split]:
        mel = np.load(io.BytesIO(row["mel"]))
        np.save(mels_dir / f"{{row['id']}}.npy", mel)
        wav_path = str(out_dir / "wavs" / f"{{row['id']}}.wav")
        lines.append(f"{{wav_path}}|{{row['lang_id']}}|{{row['phonemes']}}")

(out_dir / "train_filtered.txt").write_text("\\n".join(train_lines))
(out_dir / "val_filtered.txt").write_text("\\n".join(val_lines))
```

## Source

- Original dataset: `ghananlpcommunity/ghana-speech` (42 language subsets, ~1.41M clips)
- Phonemization: espeak `lfn` voice via `twi_cleaners`
- Mel: n_fft=1024, n_mels=80, sample_rate=22050, hop_length=256, win_length=1024, f_min=0, f_max=8000
""")
    meta_files.append((readme_path, "README.md"))

    # Upload everything
    print(f"\n[push] Uploading {len(all_files)} Parquet shards + {len(meta_files)} metadata files ...")
    for local_path, hf_path in all_files + meta_files:
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        elapsed = time.time() - t0
        size_mb = local_path.stat().st_size / 1e6
        print(f"[push]   uploaded {hf_path}: {size_mb:.1f} MB in {elapsed:.1f}s")

    print(f"\n[push] Done! Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
