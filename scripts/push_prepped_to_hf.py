#!/usr/bin/env python3
"""Push the preprocessed ghana-speech data to Hugging Face.

Uploads per-language parquet configs with phonemized metadata (no local paths)
plus lang_map and mel stats as repo-level files.

Usage:
    python scripts/push_prepped_to_hf.py --data-dir /mnt/volume_d2wey28/data/ghana_speech
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi


REPO_ID = "ghananlpcommunity/ghana-speech-phonemized"
README = """\
---
language:
- akuapem-twi
- anyin
- asante-twi
- avatime
- bassar
- bimoba
- birifor
- bissa
- buli
- chumburung
- dagaare
- dagbani
- dangme
- deg
- ewe
- fante
- fulfulde
- gikyode
- gonja
- hausa
- kabiye
- kasem
- konkomba
- konni
- kusaal
- lelemi
- mampruli
- nawuri
- ninkare
- nkonya
- ntrubo
- nzema
- paasaal
- sehwi
- sekpele
- selee
- sisala
- siwu
- tampulma
- tem
- tuwuli
- vagla
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- speech
- multilingual
- african-languages
- ghana
- phonemes
- tts
---

# Ghana Speech (Phonemized)

Preprocessed multilingual speech dataset from
[`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech),
covering **42 Ghanaian and West African languages** with IPA phoneme transcriptions.

This dataset contains only the **metadata and phonemes** — no audio.
Use it together with the source dataset to load audio on-the-fly.

## Contents

Each language is a separate **config** (Parquet). Columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique clip ID (matches audio IDs in the source dataset) |
| `text` | string | Raw orthographic transcription |
| `phon` | string | IPA phoneme transcription (espeak `lfn` voice) |
| `lang` | string | Language subset name (e.g. `Akuapem_Twi_twi`) |
| `lang_id` | int | Integer language ID (0-41) for speaker-conditioning slot |
| `dur` | float | Clip duration in seconds |
| `source_file` | string | Original source filename |
| `split` | string | `train` or `val` |

Repo-level files:

- **`lang_map.json`** — language name to lang_id mapping
- **`ghana_speech.json`** — mel normalisation stats (mel_mean, mel_std)
- **`lang_stats.json`** — per-language clip counts
- **`val_ids.json`** — list of validation clip IDs

## Filtering criteria

- Duration: 1.0s to 15.0s
- Phonemes: at least 2 non-space characters after phonemization
- Empty text: excluded

## Validation split

32 clips per language, stratified. Val clips have `split="val"` in the Parquet configs.

## Usage with Matcha-TTS

```python
from datasets import load_dataset, Audio

# Load phonemes for a specific language
ds = load_dataset("ghananlpcommunity/ghana-speech-phonemized", "Akuapem_Twi_twi", split="train")

# Load audio from the source dataset on-the-fly
audio_ds = load_dataset("ghananlpcommunity/ghana-speech", "Akuapem_Twi_twi", split="train")

# Combine: use phonemes from this dataset, audio from source
for phon_row, audio_row in zip(ds, audio_ds):
    phonemes = phon_row["phon"]
    audio = audio_row["audio"]["array"]
```

## Mel normalisation stats

```json
{"mel_mean": -5.728251241639019, "mel_std": 3.3875457364162376}
```

## Statistics

- **Total clips**: ~1,195,000
- **Languages**: 42
- **Train clips**: ~1,194,000
- **Val clips**: 1,344 (32 per language)

## License

Same as source dataset.
"""


def collect_val_ids(data_dir):
    """Extract val clip IDs from val.txt."""
    val_ids = set()
    val_path = data_dir / "val.txt"
    if val_path.exists():
        with open(val_path, encoding="utf-8") as f:
            for line in f:
                wav_path = line.split("|")[0]
                clip_id = Path(wav_path).stem
                val_ids.add(clip_id)
    return val_ids


def push_lang_stats(api, data_dir, token):
    """Push lang_stats.json to repo root."""
    manifests_dir = data_dir / "manifests_per_lang"
    stats = {}
    for manifest in sorted(manifests_dir.glob("*.jsonl")):
        name = manifest.stem
        count = sum(1 for _ in open(manifest, encoding="utf-8"))
        stats[name] = count
    content = json.dumps(stats, indent=2)
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo="lang_stats.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )
    return stats


def push_file(data_dir, filename, api, token):
    """Push a single file from data_dir to repo root."""
    path = data_dir / filename
    if not path.exists():
        return
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )
    print(f"  pushed {filename}")


def push_lang(data_dir, lang, lang_id, val_ids, api, token):
    """Read a per-lang manifest, strip wav paths, add split, push as parquet."""
    manifest_path = data_dir / "manifests_per_lang" / f"{lang}.jsonl"
    if not manifest_path.exists():
        print(f"  SKIP {lang}: no manifest")
        return

    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            clip_id = r["id"]
            rows.append({
                "id": clip_id,
                "text": r.get("text", ""),
                "phon": r.get("phon", ""),
                "lang": lang,
                "lang_id": lang_id,
                "dur": r.get("dur", 0.0),
                "source_file": r.get("source_file", ""),
                "split": "val" if clip_id in val_ids else "train",
            })

    if not rows:
        print(f"  SKIP {lang}: 0 rows")
        return

    ds = Dataset.from_list(rows)
    ds.push_to_hub(
        REPO_ID,
        config_name=lang,
        token=token,
        commit_message=f"Push {lang}: {len(rows)} clips",
    )
    print(f"  pushed {lang}: {len(rows)} clips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/volume_d2wey28/data/ghana_speech")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    token = open("/dev/stdin").read().strip() if False else None

    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)

    # Push repo-level files
    print("=== Pushing repo-level files ===")
    push_file(data_dir, "lang_map.json", api, token)
    push_file(data_dir, "ghana_speech.json", api, token)
    stats = push_lang_stats(api, data_dir, token)

    # Collect val IDs
    val_ids = collect_val_ids(data_dir)
    print(f"  {len(val_ids)} val clips identified")

    # Push val_ids.json
    val_ids_list = sorted(val_ids)
    api.upload_file(
        path_or_fileobj=json.dumps(val_ids_list, indent=2).encode(),
        path_in_repo="val_ids.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )
    print("  pushed val_ids.json")

    # Push per-language Parquet configs
    lang_map_path = data_dir / "lang_map.json"
    with open(lang_map_path, encoding="utf-8") as f:
        lang_map = json.load(f)

    print(f"\n=== Pushing {len(lang_map)} language configs ===")
    for lang, lang_id in sorted(lang_map.items(), key=lambda x: x[1]):
        push_lang(data_dir, lang, lang_id, val_ids, api, token)

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
