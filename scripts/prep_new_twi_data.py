#!/usr/bin/env python3
"""Prepare ghananlpcommunity/new-twi-tts-aligned for Matcha-TTS finetuning.

Stages (run in order, each is resumable):
  prep       - stream the HF dataset, filter by duration, resample to 22.05 kHz mono,
               phonemize with the lfn espeak voice, write wavs/ + manifest.jsonl.
  filelists  - manifest.jsonl -> train.txt / val.txt (wav_path|phonemes).
  mels       - precompute raw (un-normalised) mel-spectrograms for every clip.
  stats      - compute mel mean/std over the train set.
  all        - run prep, filelists, mels, stats in sequence.

Example:
  python scripts/prep_new_twi_data.py all --data-dir /mnt/volume_d2wey28/data/twi_new
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow importing the local `matcha` package without installing it.
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matcha.text.cleaners import twi_cleaners
from matcha.text.symbols import symbols
from matcha.utils.audio import mel_spectrogram

SAMPLE_RATE = 22050
N_FFT = 1024
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0
F_MAX = 8000


def get_args():
    p = argparse.ArgumentParser(description="Prepare new Twi TTS dataset for Matcha-TTS")
    p.add_argument("stage", choices=["prep", "filelists", "mels", "stats", "all"])
    p.add_argument("--data-dir", default="/mnt/volume_d2wey28/data/twi_new")
    p.add_argument("--dataset", default="ghananlpcommunity/new-twi-tts-aligned")
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--val-size", type=int, default=2048)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--commit-every", type=int, default=1000)
    p.add_argument("--max-clips", type=int, default=0, help="stop after this many kept clips (0 = no limit)")
    p.add_argument("--streaming", action="store_true", help="stream from HF (slow; only use if disk is tight)")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def dirs(data_dir):
    data_dir = Path(data_dir)
    wavs_dir = data_dir / "wavs"
    mels_dir = data_dir / "mels"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    mels_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, wavs_dir, mels_dir


def stage_prep(args):
    data_dir, wavs_dir, _ = dirs(args.data_dir)
    manifest_path = data_dir / "manifest.jsonl"
    state_path = data_dir / "prep_state.json"

    symbol_set = set(symbols)

    state = {"seen": 0, "kept": 0}
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"[prep] resuming: seen={state['seen']} kept={state['kept']}")

    print(f"[prep] loading dataset {args.dataset} (streaming={args.streaming})...")
    load_kwargs = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token
        print("[prep] using HF_TOKEN for authenticated download")
    ds = load_dataset(args.dataset, split=args.train_split, streaming=args.streaming, **load_kwargs)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if state["seen"]:
        ds = ds.skip(state["seen"])

    manifest_f = open(manifest_path, "a", encoding="utf-8")
    batch, errors, skipped = [], 0, 0

    def flush(force=False):
        nonlocal batch
        if batch and (force or len(batch) >= args.commit_every):
            for line in batch:
                manifest_f.write(line + "\n")
            manifest_f.flush()
            os.fsync(manifest_f.fileno())
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
            print(f"[prep] committed: seen={state['seen']} kept={state['kept']} skipped={skipped} errors={errors}")
            batch = []

    try:
        for row in tqdm(ds, desc="[prep]"):
            state["seen"] += 1
            try:
                audio = row["audio"]
                arr = np.asarray(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                dur = arr.shape[0] / sr

                text = (row.get("text") or "").strip()
                if not text or not (args.min_seconds <= dur <= args.max_seconds):
                    skipped += 1
                    continue

                phon = twi_cleaners(text)
                phon = "".join(c for c in phon if c in symbol_set)
                phon = " ".join(phon.split())
                if len(phon.replace(" ", "")) < 2:
                    skipped += 1
                    continue

                # Resample to target sample rate if needed (datasets already resampled via cast_column).
                if sr != SAMPLE_RATE:
                    arr = ta.functional.resample(torch.from_numpy(arr), sr, SAMPLE_RATE).numpy()

                clip_id = row.get("id") or f"twi_new_{state['seen']:09d}"
                wav_path = wavs_dir / f"{clip_id}.wav"
                if not wav_path.exists():
                    sf.write(wav_path, arr, SAMPLE_RATE, subtype="PCM_16")

                entry = json.dumps({"id": clip_id, "wav": str(wav_path), "phon": phon, "dur": dur})
                batch.append(entry)
                state["kept"] += 1
                flush()
                if args.max_clips and state["kept"] >= args.max_clips:
                    print(f"[prep] reached max_clips={args.max_clips}")
                    break
            except Exception as e:  # pylint: disable=broad-except
                errors += 1
                if errors <= 20:
                    print(f"[prep] row {state['seen']} error: {e}")
    finally:
        flush(force=True)
        manifest_f.close()

    print(f"[prep] DONE: seen={state['seen']} kept={state['kept']} skipped={skipped} errors={errors}")
    return {"seen": state["seen"], "kept": state["kept"], "skipped": skipped, "errors": errors}


def stage_filelists(args):
    data_dir, _, _ = dirs(args.data_dir)
    manifest_path = data_dir / "manifest.jsonl"
    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"

    rows, ids = [], set()
    print(f"[filelists] reading {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids or not Path(r["wav"]).exists():
                continue
            ids.add(r["id"])
            rows.append(r)

    random.Random(args.seed).shuffle(rows)
    val = rows[: args.val_size]
    train = rows[args.val_size :]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(f"{r['wav']}|{r['phon']}\n")

    dump(train_path, train)
    dump(val_path, val)
    print(f"[filelists] train={len(train)} val={len(val)} (total {len(rows)})")
    return {"train": len(train), "val": len(val)}


def stage_mels(args):
    _, wavs_dir, mels_dir = dirs(args.data_dir)
    wav_paths = sorted(wavs_dir.glob("*.wav"))
    print(f"[mels] {len(wav_paths)} clips to consider")

    class MelPrecompute(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __getitem__(self, i):
            wav = self.paths[i]
            mp = mels_dir / f"{wav.stem}.npy"
            if mp.exists():
                return 1
            data, sr = sf.read(wav, dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T)
            assert sr == SAMPLE_RATE
            mel = mel_spectrogram(
                audio,
                N_FFT,
                N_MELS,
                SAMPLE_RATE,
                HOP_LENGTH,
                WIN_LENGTH,
                F_MIN,
                F_MAX,
                center=False,
            ).squeeze()
            np.save(mp, mel.numpy())
            return 0

        def __len__(self):
            return len(self.paths)

    loader = DataLoader(
        MelPrecompute(wav_paths),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda b: sum(b),
    )

    done = 0
    for n in tqdm(loader, desc="[mels]"):
        done += n
    print(f"[mels] DONE: {done}/{len(wav_paths)} clips have mels in {mels_dir}")
    return {"clips": len(wav_paths), "processed": done}


def stage_stats(args):
    data_dir, _, _ = dirs(args.data_dir)
    train_path = data_dir / "train.txt"
    stats_path = data_dir / "twi_new.json"
    progress_path = data_dir / "stats_progress.json"

    with open(train_path, encoding="utf-8") as f:
        wavs = [ln.split("|")[0] for ln in f if ln.strip()]
    total = len(wavs)

    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            prog = json.load(f)
    else:
        prog = {"processed": 0, "sum": 0.0, "sq": 0.0, "frames": 0}
    print(f"[stats] resuming at {prog['processed']}/{total} clips")

    if prog["processed"] >= total and stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            params = json.load(f)
        print(f"[stats] already complete: {params}")
        return params

    class MelStatsDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __getitem__(self, i):
            data, _ = sf.read(self.paths[i], dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T)
            mel = mel_spectrogram(
                audio,
                N_FFT,
                N_MELS,
                SAMPLE_RATE,
                HOP_LENGTH,
                WIN_LENGTH,
                F_MIN,
                F_MAX,
                center=False,
            ).squeeze()
            return torch.tensor(
                [float(mel.sum()), float((mel**2).sum()), float(mel.shape[-1])],
                dtype=torch.float64,
            )

        def __len__(self):
            return len(self.paths)

    remaining = wavs[prog["processed"] :]
    loader = DataLoader(
        MelStatsDataset(remaining),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    acc = torch.tensor([prog["sum"], prog["sq"], float(prog["frames"])], dtype=torch.float64)
    processed = prog["processed"]
    last_commit = processed

    def checkpoint():
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed": processed,
                    "sum": acc[0].item(),
                    "sq": acc[1].item(),
                    "frames": int(acc[2].item()),
                },
                f,
            )

    for batch in tqdm(loader, desc="[stats]"):
        acc += batch.sum(dim=0)
        processed += batch.shape[0]
        if processed - last_commit >= args.commit_every:
            checkpoint()
            last_commit = processed

    checkpoint()
    denom = acc[2].item() * N_MELS
    mean = acc[0].item() / denom
    std = math.sqrt(acc[1].item() / denom - mean**2)
    params = {"mel_mean": mean, "mel_std": std}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(params, f)
    print(f"[stats] DONE over {total} clips: {params}")
    return params


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.stage == "prep":
        stage_prep(args)
    elif args.stage == "filelists":
        stage_filelists(args)
    elif args.stage == "mels":
        stage_mels(args)
    elif args.stage == "stats":
        stage_stats(args)
    elif args.stage == "all":
        stage_prep(args)
        stage_filelists(args)
        stage_mels(args)
        stage_stats(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
