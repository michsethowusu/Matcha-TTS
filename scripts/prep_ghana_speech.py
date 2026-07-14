#!/usr/bin/env python3
"""Prepare ghananlpcommunity/ghana-speech for multilingual Matcha-TTS finetuning.

Streams each language subset, filters/resamples/phonemizes, and writes a single
mixed dataset with language IDs repurposing Matcha's speaker-conditioning slot.

Output layout under --data-dir:
  wavs/           22.05 kHz mono PCM16 .wav files
  mels/           precomputed raw (un-normalised) mel-spectrograms (.npy)
  manifest.jsonl  metadata for every kept clip
  train.txt       filelist: wav_path|lang_id|phonemes
  val.txt         filelist: wav_path|lang_id|phonemes
  ghana_speech.json   mel mean/std
  lang_map.json   language_name -> lang_id

Resumable: each stage skips already-completed work (existing wavs/mels/state).
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matcha.text.cleaners import twi_cleaners
from matcha.text.symbols import symbols
from matcha.utils.audio import mel_spectrogram

def _retry(fn, retries=5, backoff=30, label="operation"):
    """Retry *fn()* up to *retries* times with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                print(f"[prep] {label} failed after {retries} attempts: {e}")
                raise
            wait = backoff * attempt
            print(f"[prep] {label} attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


SAMPLE_RATE = 22050
N_FFT = 1024
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0
F_MAX = 8000


def get_args():
    p = argparse.ArgumentParser(description="Prepare ghana-speech dataset")
    p.add_argument("--data-dir", default="/mnt/volume_d2wey28/data/ghana_speech")
    p.add_argument("--dataset", default="ghananlpcommunity/ghana-speech")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--val-per-lang", type=int, default=32,
                   help="number of validation clips per language (stratified split)")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--phonemize-batch", type=int, default=256,
                   help="texts per espeak phonemization batch")
    p.add_argument("--max-clips-per-lang", type=int, default=0,
                   help="stop after N kept clips per language (0 = no limit)")
    p.add_argument("--streaming", action="store_true",
                   help="stream subsets instead of downloading full parquet first (slow; for testing)")
    p.add_argument("--commit-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--push-manifest-repo", default="ghananlpcommunity/ghana-speech-lfn-phonemized",
                   help="HF dataset repo to push phoneme manifests (no audio). Set empty to skip.")
    p.add_argument("--push-manifest-private", action="store_true", default=False,
                   help="make the pushed manifest dataset private")
    return p.parse_args()


def ensure_dirs(data_dir):
    data_dir = Path(data_dir)
    wavs_dir = data_dir / "wavs"
    mels_dir = data_dir / "mels"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    mels_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, wavs_dir, mels_dir


def get_hf_token():
    return os.environ.get("HF_TOKEN")


def load_subset_configs(dataset_name):
    """Discover all language subset names from the dataset card."""
    from huggingface_hub import HfApi
    token = get_hf_token()
    api = HfApi(token=token) if token else HfApi()
    info = api.dataset_info(dataset_name)
    configs = [c["config_name"] for c in info.cardData.get("dataset_info", [])]
    if not configs:
        raise RuntimeError("Could not discover dataset configs")
    return sorted(set(configs))


def phonemize_batch(texts):
    """Phonemize a list of raw texts with the lfn espeak voice."""
    from matcha.text.cleaners import _get_twi_phonemizer

    symbol_set = set(symbols)
    phones = _get_twi_phonemizer().phonemize(texts, strip=True, njobs=1)
    out = []
    for phon in phones:
        phon = "".join(c for c in phon if c in symbol_set)
        phon = " ".join(phon.split())
        out.append(phon)
    return out


def stage_prep(args):
    data_dir, wavs_dir, _ = ensure_dirs(args.data_dir)
    manifest_path = data_dir / "manifest.jsonl"
    per_lang_dir = data_dir / "manifests_per_lang"
    per_lang_dir.mkdir(exist_ok=True)
    state_path = data_dir / "prep_state.json"
    lang_map_path = data_dir / "lang_map.json"

    subsets = load_subset_configs(args.dataset)
    print(f"[prep] discovered {len(subsets)} language subsets")
    lang_to_id = {lang: i for i, lang in enumerate(subsets)}
    with open(lang_map_path, "w", encoding="utf-8") as f:
        json.dump(lang_to_id, f, indent=2)

    symbol_set = set(symbols)
    state = {"done_subsets": [], "kept": 0, "skipped": 0, "errors": 0}
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"[prep] resuming: {len(state['done_subsets'])} subsets done, kept={state['kept']}")

    manifest_f = open(manifest_path, "a", encoding="utf-8")
    per_lang_fs = {}

    load_kwargs = {}
    token = get_hf_token()
    if token:
        load_kwargs["token"] = token
        print("[prep] using HF_TOKEN for authenticated download")

    pending = [s for s in subsets if s not in state["done_subsets"]]

    for subset in pending:
        print(f"\n[prep] === subset: {subset} (lang_id={lang_to_id[subset]}) ===")
        ds = _retry(
            lambda s=subset: load_dataset(args.dataset, s, split="train", streaming=args.streaming, **load_kwargs),
            retries=5, backoff=60, label=f"load_dataset({subset})",
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

        per_lang_path = per_lang_dir / f"{subset}.jsonl"
        per_lang_fs[subset] = open(per_lang_path, "a", encoding="utf-8")

        batch_texts, batch_meta = [], []
        subset_kept = 0
        pbar = tqdm(ds, desc=f"[prep {subset}]")
        for row in pbar:
            try:
                dur = float(row.get("duration") or 0.0)
                text = (row.get("text") or "").strip()
                if not text or not (args.min_seconds <= dur <= args.max_seconds):
                    state["skipped"] += 1
                    continue

                clip_id = row.get("id") or f"{subset}_{len(batch_texts):09d}"
                wav_path = wavs_dir / f"{clip_id}.wav"

                # If wav already exists, just append manifest without re-phonemizing.
                if wav_path.exists():
                    audio, sr = sf.read(wav_path, dtype="float32")
                    assert sr == SAMPLE_RATE
                else:
                    audio = np.asarray(row["audio"]["array"], dtype="float32")
                    sr = row["audio"]["sampling_rate"]
                    if sr != SAMPLE_RATE:
                        audio = ta.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
                    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")

                batch_texts.append(text)
                batch_meta.append({
                    "id": clip_id,
                    "wav": str(wav_path),
                    "lang": subset,
                    "lang_id": lang_to_id[subset],
                    "text": text,
                    "source_file": row.get("source_file", ""),
                    "dur": dur,
                })

                # Phonemize in batches for speed.
                if len(batch_texts) >= args.phonemize_batch:
                    _flush_phoneme_batch(batch_texts, batch_meta, manifest_f, per_lang_fs, symbol_set, state)
                    batch_texts, batch_meta = [], []

                state["kept"] += 1
                subset_kept += 1
                if state["kept"] % args.commit_every == 0:
                    _commit(state_path, state, manifest_f)

                pbar.set_postfix({"kept": state["kept"], "skip": state["skipped"], "err": state["errors"]})

                if args.max_clips_per_lang and subset_kept >= args.max_clips_per_lang:
                    print(f"[prep] reached max_clips_per_lang={args.max_clips_per_lang} for {subset}")
                    break

            except Exception as e:
                state["errors"] += 1
                if state["errors"] <= 20:
                    print(f"[prep] error in {subset}: {e}")

        # Flush remaining texts for this subset.
        if batch_texts:
            _flush_phoneme_batch(batch_texts, batch_meta, manifest_f, per_lang_fs, symbol_set, state)

        if subset in per_lang_fs:
            per_lang_fs[subset].flush()
            per_lang_fs[subset].close()
            del per_lang_fs[subset]
        _push_manifest_subset(data_dir, subset, args.push_manifest_repo, args.push_manifest_private)

        manifest_f.flush()
        state["done_subsets"].append(subset)
        _commit(state_path, state, manifest_f)
        print(f"[prep] subset {subset} done. cumulative kept={state['kept']}")

    manifest_f.close()
    for f in per_lang_fs.values():
        f.close()
    print(f"[prep] DONE: kept={state['kept']} skipped={state['skipped']} errors={state['errors']}")
    return state


def _push_manifest_subset(data_dir, subset, repo_id, private):
    if not repo_id:
        return
    token = get_hf_token()
    if not token:
        print(f"[prep] skip HF push for {subset}: HF_TOKEN not set")
        return
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi

        per_lang_path = data_dir / "manifests_per_lang" / f"{subset}.jsonl"
        if not per_lang_path.exists():
            return
        rows = []
        with open(per_lang_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "id": r["id"],
                    "language": r["lang"],
                    "text": r.get("text", ""),
                    "phonemes": r["phon"],
                    "duration": r["dur"],
                    "source_file": r.get("source_file", ""),
                    "lang_id": r["lang_id"],
                    "wav_filename": Path(r["wav"]).name,
                })
        if not rows:
            return
        ds = Dataset.from_list(rows)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        ds.push_to_hub(repo_id, config_name=subset, token=token, private=private)
        print(f"[prep] pushed {len(rows)} manifest rows to {repo_id} (config={subset})")
    except Exception as e:
        print(f"[prep] failed to push manifest for {subset}: {e}")


def _flush_phoneme_batch(texts, metas, manifest_f, per_lang_fs, symbol_set, state):
    try:
        phones = phonemize_batch(texts)
    except Exception as e:
        print(f"[prep] phonemization batch error: {e}")
        state["errors"] += len(texts)
        return
    for meta, phon in zip(metas, phones):
        if len(phon.replace(" ", "")) < 2:
            state["skipped"] += 1
            continue
        entry = {**meta, "phon": phon}
        manifest_f.write(json.dumps(entry) + "\n")
        lang_f = per_lang_fs.get(meta["lang"])
        if lang_f:
            lang_f.write(json.dumps(entry) + "\n")


def _commit(state_path, state, manifest_f):
    manifest_f.flush()
    os.fsync(manifest_f.fileno())
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def stage_filelists(args):
    data_dir, _, _ = ensure_dirs(args.data_dir)
    manifest_path = data_dir / "manifest.jsonl"
    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"

    # Group by language.
    by_lang = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not Path(r["wav"]).exists():
                continue
            by_lang.setdefault(r["lang"], []).append(r)

    train_rows, val_rows = [], []
    rng = random.Random(args.seed)
    for lang, rows in by_lang.items():
        rng.shuffle(rows)
        n_val = min(args.val_per_lang, len(rows))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(f"{r['wav']}|{r['lang_id']}|{r['phon']}\n")

    dump(train_path, train_rows)
    dump(val_path, val_rows)
    print(f"[filelists] train={len(train_rows)} val={len(val_rows)} languages={len(by_lang)}")
    return {"train": len(train_rows), "val": len(val_rows)}


def stage_mels(args):
    _, wavs_dir, mels_dir = ensure_dirs(args.data_dir)
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
    data_dir, _, _ = ensure_dirs(args.data_dir)
    train_path = data_dir / "train.txt"
    stats_path = data_dir / "ghana_speech.json"
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
            return json.load(f)

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

    remaining = wavs[prog["processed"]:]
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
            json.dump({
                "processed": processed,
                "sum": acc[0].item(),
                "sq": acc[1].item(),
                "frames": int(acc[2].item()),
            }, f)

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
    # Process stages in order. Each stage is resumable.
    stage_prep(args)
    stage_filelists(args)
    stage_mels(args)
    stage_stats(args)


if __name__ == "__main__":
    main()
