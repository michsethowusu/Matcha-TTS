#!/usr/bin/env python3
"""Filter the ghana-speech training set for reasonable durations and a per-language cap.

Reads train.txt + val.txt (pipe-delimited: wav_path|lang_id|phonemes), computes each
clip's duration from the wav header, and writes filtered versions:

  train_filtered.txt   clips in [min_dur, max_dur], capped at max_hours per language
  val_filtered.txt     32 shortest qualifying clips per language (held out from train cap)

Also recomputes mel mean/std over the filtered training set.

Usage:
    python scripts/filter_ghana_speech.py \
        --data-dir /mnt/volume_d2wey28/data/ghana_speech \
        --min-duration 1.0 \
        --max-duration 10.0 \
        --max-hours-per-lang 10 \
        --val-per-lang 32 \
        --workers 16
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_filelist(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            lines.append((parts[0], parts[1], parts[2]))  # wav_path, lang_id, phonemes
    return lines


def get_duration(item):
    wav_path, lang_id, phonemes = item
    try:
        info = sf.info(wav_path)
        return (wav_path, lang_id, phonemes, info.duration)
    except Exception as e:
        return (wav_path, lang_id, phonemes, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--max-hours-per-lang", type=float, default=10.0)
    parser.add_argument("--val-per-lang", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    max_secs = args.max_hours_per_lang * 3600

    # Load both filelists
    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"

    print(f"[filter] Loading {train_path} ...")
    train_items = parse_filelist(train_path)
    print(f"[filter] Loading {val_path} ...")
    val_items = parse_filelist(val_path)

    all_items = train_items + val_items
    print(f"[filter] Combined pool: {len(all_items)} clips ({len(train_items)} train + {len(val_items)} val)")

    # Compute durations in parallel
    print(f"[filter] Computing durations with {args.workers} workers ...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_duration, item): item for item in all_items}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"[filter]   {done}/{len(all_items)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[filter] Duration computation done in {elapsed:.1f}s")

    # Split good vs bad
    good = []
    bad_duration = 0
    bad_missing = 0
    for wav_path, lang_id, phonemes, dur in results:
        if dur is None:
            bad_missing += 1
            continue
        if dur < args.min_duration or dur > args.max_duration:
            bad_duration += 1
            continue
        good.append((wav_path, lang_id, phonemes, dur))

    print(f"[filter] Kept {len(good)} clips after [{args.min_duration}, {args.max_duration}]s filter")
    print(f"[filter] Dropped: {bad_duration} out-of-range, {bad_missing} unreadable")

    # Group by language
    by_lang = defaultdict(list)
    for item in good:
        by_lang[item[1]].append(item)

    print(f"[filter] Languages with qualifying clips: {len(by_lang)}")

    # Sort each language by duration (shortest first), then cap
    val_clips = []
    train_clips = []
    lang_stats = {}

    for lang_id in sorted(by_lang.keys(), key=int):
        clips = sorted(by_lang[lang_id], key=lambda x: x[3])

        # First val_per_lang go to validation (shortest, most uniform)
        n_val = min(args.val_per_lang, len(clips))
        val_clips.extend(clips[:n_val])

        # Remaining go to training, capped at max_secs
        remaining = clips[n_val:]
        cumsum = 0
        kept = []
        for item in remaining:
            if cumsum + item[3] > max_secs:
                break
            cumsum += item[3]
            kept.append(item)
        train_clips.extend(kept)

        lang_stats[lang_id] = {
            "total_qualifying": len(clips),
            "val": n_val,
            "train": len(kept),
            "train_hours": cumsum / 3600,
            "capped": len(remaining) > len(kept),
        }

    print(f"\n[filter] Final: {len(train_clips)} train, {len(val_clips)} val")
    total_train_hours = sum(s["train_hours"] for s in lang_stats.values())
    print(f"[filter] Total training audio: {total_train_hours:.1f} hours")

    # Write filtered filelists
    def write_filelist(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for wav_path, lang_id, phonemes, _dur in items:
                f.write(f"{wav_path}|{lang_id}|{phonemes}\n")

    train_out = data_dir / "train_filtered.txt"
    val_out = data_dir / "val_filtered.txt"
    write_filelist(train_out, train_clips)
    write_filelist(val_out, val_clips)
    print(f"[filter] Wrote {train_out}")
    print(f"[filter] Wrote {val_out}")

    # Recompute mel mean/std from the filtered training set mels
    print(f"\n[filter] Computing mel stats for filtered training set ...")
    mel_dir = data_dir / "mels"
    mel_sum = 0.0
    mel_sq_sum = 0.0
    mel_count = 0
    missing_mels = 0

    t0 = time.time()
    for i, (wav_path, _, _, _) in enumerate(train_clips):
        mel_path = mel_dir / (Path(wav_path).stem + ".npy")
        try:
            mel = np.load(mel_path)
            mel_sum += mel.sum()
            mel_sq_sum += (mel ** 2).sum()
            mel_count += mel.size
        except FileNotFoundError:
            missing_mels += 1
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - t0
            print(f"[filter]   mel stats: {i+1}/{len(train_clips)} ({elapsed:.1f}s)")

    mel_mean = mel_sum / mel_count
    mel_std = float(np.sqrt(mel_sq_sum / mel_count - mel_mean ** 2))
    elapsed = time.time() - t0
    print(f"[filter] Mel stats computed in {elapsed:.1f}s (missing mels: {missing_mels})")
    print(f"[filter] mel_mean={mel_mean}, mel_std={mel_std}")

    # Save stats
    stats = {
        "mel_mean": float(mel_mean),
        "mel_std": float(mel_std),
        "n_train": len(train_clips),
        "n_val": len(val_clips),
        "max_hours_per_lang": args.max_hours_per_lang,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "total_train_hours": total_train_hours,
        "lang_stats": {str(k): v for k, v in lang_stats.items()},
    }
    stats_path = data_dir / "ghana_speech_filtered.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[filter] Wrote {stats_path}")

    # Print per-language summary
    print(f"\n{'lang_id':>7} {'lang':>5} {'kept':>6} {'total':>6} {'hours':>6} {'capped'}")
    print("-" * 50)
    for lang_id in sorted(lang_stats.keys(), key=int):
        s = lang_stats[lang_id]
        flag = "YES" if s["capped"] else ""
        print(f"{lang_id:>7} {s['train']:>5} {s['train']:>6} {s['total_qualifying']:>6} {s['train_hours']:>6.2f} {flag}")

    print(f"\n[filter] Done!")


if __name__ == "__main__":
    main()
