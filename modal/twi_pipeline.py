"""Modal pipeline to finetune Matcha-TTS on Asante Twi.

Stages (each is independently runnable and RESUMABLE — state lives on a Modal Volume so a
crash/timeout just means re-running the same command):

  modal run modal/twi_pipeline.py::prep_data        # download + filter + resample + phonemize
  modal run modal/twi_pipeline.py::build_filelists  # manifest.jsonl -> train.txt / val.txt
  modal run modal/twi_pipeline.py::compute_stats    # mel_mean / mel_std -> twi.json
  modal run modal/twi_pipeline.py::train            # finetune from English ckpt, push to HF

Or run the whole chain:  modal run modal/twi_pipeline.py

Prerequisites (once):
  modal secret create huggingface HF_TOKEN=hf_xxx     # needs read (dataset) + write (push) scope
Data source: michsethowusu/ghana-speech, subset "Asante_Twi_twi" (~143k rows).
Checkpoints are pushed to $HF_REPO_ID (default michsethowusu/matcha-twi).
"""
import os
import pathlib

import modal

# ----------------------------------------------------------------------------- config
REPO_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = "/data"                       # Modal Volume mount
TWI_DIR = f"{DATA_DIR}/twi"
WAVS_DIR = f"{TWI_DIR}/wavs"
MELS_DIR = f"{TWI_DIR}/mels"  # precomputed raw mel-spectrograms (.npy), loaded by get_mel
ONNX_DIR = f"{TWI_DIR}/onnx"  # sherpa-onnx export bundle (acoustic onnx per epoch + vocoder + tokens)
VOCODER_ONNX_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx"
MANIFEST = f"{TWI_DIR}/manifest.jsonl"
STATE = f"{TWI_DIR}/state.json"
TRAIN_TXT = f"{TWI_DIR}/train.txt"
VAL_TXT = f"{TWI_DIR}/val.txt"
STATS_JSON = f"{TWI_DIR}/twi.json"
STATS_PROGRESS = f"{TWI_DIR}/stats_progress.json"  # resumable accumulator checkpoint
PRETRAINED = f"{DATA_DIR}/pretrained/matcha_ljspeech.ckpt"
PRETRAINED_URL = (
    "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt"
)
RUN_DIR = f"{TWI_DIR}/runs/twi"          # hydra output dir; checkpoints + last.ckpt live here
LAST_CKPT = f"{RUN_DIR}/checkpoints/last.ckpt"

DATASET = "michsethowusu/ghana-speech"
SUBSET = "Asante_Twi_twi"
SAMPLE_RATE = 22050
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0
HF_REPO_ID = "michsethowusu/matcha-twi"

app = modal.App("matcha-twi")
volume = modal.Volume.from_name("matcha-twi", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("espeak-ng", "ffmpeg", "git", "build-essential")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    # datasets<3.0 decodes Audio via soundfile/librosa; datasets>=3.0 requires torchcodec.
    .pip_install("datasets==2.21.0", "soundfile", "huggingface_hub", "onnx", "onnxruntime")
    .env({"HF_REPO_ID": HF_REPO_ID})
    # Copy the repo into the image and install it so `matcha` + the configs are importable.
    .add_local_dir(
        str(REPO_ROOT),
        "/root/Matcha-TTS",
        copy=True,
        ignore=["data", "logs", "outputs", "**/__pycache__", ".git", "*.ckpt"],
    )
    .run_commands("cd /root/Matcha-TTS && pip install -e . --no-deps")
    .workdir("/root/Matcha-TTS")
)


# --------------------------------------------------------------------------- helpers
def _read_state():
    import json

    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"seen": 0, "kept": 0}


def _write_state(state):
    import json

    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f)


# ------------------------------------------------------------------------- prep_data
@app.function(image=image, volumes={DATA_DIR: volume}, secrets=[hf_secret], timeout=86400)
def prep_data(commit_every: int = 500, max_clips: int = 0):
    """Stream the Asante Twi subset, filter by duration/text, resample to 22.05 kHz mono,
    phonemize with the lfn espeak voice, and append to manifest.jsonl. Resumable: re-running
    skips the rows already consumed (tracked in state.json) and commits every `commit_every`
    kept clips so a crash loses at most one batch. max_clips>0 stops after that many kept
    (useful for a vocoder-finetuning subset)."""
    import json

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    from matcha.text.cleaners import twi_cleaners
    from matcha.text.symbols import symbols

    symbol_set = set(symbols)
    os.makedirs(WAVS_DIR, exist_ok=True)

    state = _read_state()
    seen, kept = state["seen"], state["kept"]
    print(f"[prep] resuming: {seen} source rows already consumed, {kept} clips kept")

    ds = load_dataset(DATASET, SUBSET, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if seen:
        ds = ds.skip(seen)

    manifest_f = open(MANIFEST, "a", encoding="utf-8")  # noqa: SIM115  (long-lived append handle)
    batch, errors, skipped = 0, 0, 0

    def flush(force=False):
        nonlocal batch
        if batch and (force or batch >= commit_every):
            manifest_f.flush()
            os.fsync(manifest_f.fileno())
            _write_state({"seen": seen, "kept": kept})
            volume.commit()
            print(f"[prep] committed: seen={seen} kept={kept} skipped={skipped} errors={errors}")
            batch = 0

    try:
        for row in ds:
            seen += 1
            try:
                dur = float(row.get("duration") or 0.0)
                text = (row.get("text") or "").strip()
                if not text or not (MIN_SECONDS <= dur <= MAX_SECONDS):
                    skipped += 1
                    continue

                phon = twi_cleaners(text)
                phon = "".join(c for c in phon if c in symbol_set)  # drop any OOV symbol
                phon = " ".join(phon.split())
                if len(phon.replace(" ", "")) < 2:
                    skipped += 1
                    continue

                clip_id = row.get("id") or f"twi_{seen:08d}"
                audio = row["audio"]
                wav = np.asarray(audio["array"], dtype="float32")
                wav_path = f"{WAVS_DIR}/{clip_id}.wav"
                sf.write(wav_path, wav, SAMPLE_RATE, subtype="PCM_16")

                manifest_f.write(json.dumps({"id": clip_id, "wav": wav_path, "phon": phon, "dur": dur}) + "\n")
                kept += 1
                batch += 1
                flush()
                if max_clips and kept >= max_clips:
                    print(f"[prep] reached max_clips={max_clips}")
                    break
            except Exception as e:  # pylint: disable=broad-except
                errors += 1
                if errors <= 20:
                    print(f"[prep] row {seen} error: {e}")
    finally:
        flush(force=True)
        manifest_f.close()

    print(f"[prep] DONE: seen={seen} kept={kept} skipped={skipped} errors={errors}")
    return {"seen": seen, "kept": kept, "skipped": skipped, "errors": errors}


# -------------------------------------------------------------------- build_filelists
@app.function(image=image, volumes={DATA_DIR: volume}, timeout=3600)
def build_filelists(val_size: int = 256, seed: int = 1234):
    """Turn manifest.jsonl into train.txt / val.txt ("wav_path|phonemes"), de-duped by id."""
    import json
    import random

    rows, ids = [], set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids or not os.path.exists(r["wav"]):
                continue
            ids.add(r["id"])
            rows.append(r)

    random.Random(seed).shuffle(rows)
    val = rows[:val_size]
    train = rows[val_size:]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(f"{r['wav']}|{r['phon']}\n")

    dump(TRAIN_TXT, train)
    dump(VAL_TXT, val)
    volume.commit()
    print(f"[filelists] train={len(train)} val={len(val)} (total {len(rows)})")
    return {"train": len(train), "val": len(val)}


# -------------------------------------------------------------------- compute_stats
@app.function(image=image, volumes={DATA_DIR: volume}, cpu=16.0, memory=32768, timeout=86400)
def precompute_mels(num_workers: int = 16, commit_every: int = 4000):
    """Precompute raw (un-normalised) mel-spectrograms for every clip and store them as .npy next
    to the wavs (in /data/twi/mels). Training then loads these instead of reading the wav and
    running an FFT every step — turning the data-bound pipeline GPU-bound. RESUMABLE: each worker
    skips clips whose .npy already exists, so a restart only does the remaining ones. Mel params
    must match get_mel exactly (n_fft=1024, n_mels=80, sr=22050, hop=256, win=1024, fmin=0, fmax=8000)."""
    import glob
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch
    from torch.utils.data import DataLoader, Dataset

    from matcha.utils.audio import mel_spectrogram

    os.makedirs(MELS_DIR, exist_ok=True)
    wavs = sorted(glob.glob(f"{WAVS_DIR}/*.wav"))
    print(f"[mels] {len(wavs)} clips to consider")

    class MelPrecompute(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            wav = self.paths[i]
            mp = f"{MELS_DIR}/{Path(wav).stem}.npy"
            if os.path.exists(mp):
                return 1  # already done
            data, _ = sf.read(wav, dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T)
            mel = mel_spectrogram(audio, 1024, 80, SAMPLE_RATE, 256, 1024, 0, 8000, center=False).squeeze()
            np.save(mp, mel.numpy())
            return 0

    loader = DataLoader(
        MelPrecompute(wavs), batch_size=64, num_workers=num_workers, shuffle=False,
        collate_fn=lambda b: len(b),
    )
    seen = 0
    last_commit = 0
    for n in loader:
        seen += n
        if seen - last_commit >= commit_every:
            volume.commit()
            last_commit = seen
            print(f"[mels] {seen}/{len(wavs)} processed")
    volume.commit()
    print(f"[mels] DONE: {seen}/{len(wavs)} clips have mels in {MELS_DIR}")
    return {"clips": len(wavs)}


@app.function(image=image, volumes={DATA_DIR: volume}, cpu=16.0, memory=32768, timeout=86400)
def compute_stats(batch_size: int = 256, num_workers: int = 16, commit_every: int = 4000):
    """Compute mel mean/std over the FULL train set, RESUMABLE. Mel statistics are an online sum
    (mel_sum, mel_sq_sum, frame_count), so we walk the filelist in fixed order, checkpoint the
    running accumulators + processed count to the volume every `commit_every` clips, and on
    restart skip the already-processed prefix. Worker preemption costs at most one batch window,
    never the whole pass. n_feats=80, computed on raw (un-normalised) mels."""
    import json
    import math

    import torch
    from torch.utils.data import DataLoader, Dataset

    from matcha.utils.audio import mel_spectrogram

    N_FEATS = 80

    with open(TRAIN_TXT, encoding="utf-8") as f:
        wavs = [ln.split("|")[0] for ln in f if ln.strip()]
    total = len(wavs)

    # resume from checkpoint if present
    if os.path.exists(STATS_PROGRESS):
        with open(STATS_PROGRESS, encoding="utf-8") as f:
            prog = json.load(f)
    else:
        prog = {"processed": 0, "sum": 0.0, "sq": 0.0, "frames": 0}
    print(f"[stats] resuming at {prog['processed']}/{total} clips")
    if prog["processed"] >= total:
        params = _finalize_stats(prog, N_FEATS)
        print(f"[stats] already complete: {params}")
        return params

    class MelStatsDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            import soundfile as sf

            data, _ = sf.read(self.paths[i], dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T)
            mel = mel_spectrogram(audio, 1024, N_FEATS, SAMPLE_RATE, 256, 1024, 0, 8000, center=False).squeeze()
            return torch.tensor([float(mel.sum()), float((mel**2).sum()), float(mel.shape[-1])], dtype=torch.float64)

    remaining = wavs[prog["processed"] :]
    loader = DataLoader(
        MelStatsDataset(remaining), batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=False, drop_last=False,
    )

    acc = torch.tensor([prog["sum"], prog["sq"], float(prog["frames"])], dtype=torch.float64)
    processed = prog["processed"]
    last_commit = processed

    def checkpoint():
        with open(STATS_PROGRESS, "w", encoding="utf-8") as f:
            json.dump({"processed": processed, "sum": acc[0].item(), "sq": acc[1].item(),
                       "frames": int(acc[2].item())}, f)
        volume.commit()
        print(f"[stats] checkpoint {processed}/{total}")

    for batch in loader:
        acc += batch.sum(dim=0)
        processed += batch.shape[0]
        if processed - last_commit >= commit_every:
            checkpoint()
            last_commit = processed

    checkpoint()  # final accumulator state
    params = _finalize_stats({"sum": acc[0].item(), "sq": acc[1].item(), "frames": int(acc[2].item())}, N_FEATS)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f)
    volume.commit()
    print(f"[stats] DONE over {total} clips: {params}")
    return params


def _finalize_stats(prog, n_feats):
    import math

    denom = prog["frames"] * n_feats
    mean = prog["sum"] / denom
    std = math.sqrt(prog["sq"] / denom - mean**2)
    return {"mel_mean": mean, "mel_std": std}


# ---------------------------------------------------------------------------- train
@app.function(image=image, gpu="A10G", cpu=8.0, volumes={DATA_DIR: volume}, secrets=[hf_secret], timeout=86400)
def train(max_epochs: int = 500):
    """Finetune Matcha-TTS from the English LJSpeech checkpoint. Resumes from last.ckpt if it
    exists on the volume; otherwise loads the pretrained weights (fresh optimizer). The volume
    is committed every few minutes so a timeout/crash can resume, and every saved checkpoint is
    pushed to the HF Hub by HFModelCheckpoint."""
    import json
    import subprocess
    import threading
    import time
    import urllib.request

    # 1) pretrained English checkpoint (cached on the volume)
    if not os.path.exists(PRETRAINED):
        os.makedirs(os.path.dirname(PRETRAINED), exist_ok=True)
        print("[train] downloading pretrained English checkpoint...")
        urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED)
        volume.commit()

    # 2) data statistics
    with open(STATS_JSON, encoding="utf-8") as f:
        stats = json.load(f)
    print(f"[train] data statistics: {stats}")

    # 3) background commit so checkpoints persist for resume
    stop = threading.Event()

    def committer():
        while not stop.wait(180):
            try:
                volume.commit()
            except Exception as e:  # pylint: disable=broad-except
                print(f"[train] volume commit failed: {e}")

    threading.Thread(target=committer, daemon=True).start()

    # 4) resume vs finetune
    cmd = [
        "python", "matcha/train.py", "experiment=twi", "train=true", "test=false",
        f"trainer.max_epochs={max_epochs}",
        "data.batch_size=64",     # A100 40GB has headroom beyond the A10G's batch 32
        "data.num_workers=8",     # match the 8 reserved cores; more just oversubscribes the CPU
        f"data.data_statistics.mel_mean={stats['mel_mean']}",
        f"data.data_statistics.mel_std={stats['mel_std']}",
        f"paths.output_dir={RUN_DIR}",
        f"hydra.run.dir={RUN_DIR}",
    ]
    if os.path.exists(LAST_CKPT):
        print(f"[train] resuming from {LAST_CKPT}")
        cmd.append(f"ckpt_path={LAST_CKPT}")
    else:
        print(f"[train] finetuning from {PRETRAINED}")
        cmd.append(f"finetune_ckpt={PRETRAINED}")

    env = {**os.environ, "HF_REPO_ID": os.environ.get("HF_REPO_ID", HF_REPO_ID)}
    try:
        subprocess.run(cmd, cwd="/root/Matcha-TTS", env=env, check=True)
    finally:
        stop.set()
        volume.commit()
    print("[train] done")


@app.function(image=image, volumes={DATA_DIR: volume}, secrets=[hf_secret], cpu=8.0, memory=16384, timeout=14400)
def export_onnx(epochs: str = "41,42,43,44,45", n_timesteps: int = 5):
    """Convert the selected epoch checkpoints to a sherpa-onnx bundle: one acoustic ONNX per
    epoch (mel output) with sherpa metadata embedded, a shared tokens.txt (from symbols.py), a
    HiFi-GAN vocoder ONNX, and espeak-ng-data (lfn voice). Everything is pushed to HF under
    sherpa-onnx/ so it can be run with sherpa-onnx offline TTS."""
    import glob
    import shutil
    import subprocess
    import urllib.request

    import onnx

    from matcha.text.symbols import symbols

    os.makedirs(ONNX_DIR, exist_ok=True)
    eps = [e.strip() for e in epochs.split(",") if e.strip()]

    def add_meta(path, extra):
        m = onnx.load(path)
        meta = {
            "model_type": "matcha-tts",
            "language": "Twi",
            "voice": "lfn",            # espeak-ng voice used for phonemization (matches training)
            "has_espeak": 1,
            "jieba": 0,
            "n_speakers": 1,
            "sample_rate": SAMPLE_RATE,
            "version": 1,
            "pad_id": 0,
            "use_icefall": 0,
            "num_ode_steps": n_timesteps,
            "model_author": "michsethowusu",
            "dataset": "michsethowusu/ghana-speech (Asante_Twi)",
            "comment": "Matcha-TTS Asante Twi; lfn espeak phonemizer",
            **extra,
        }
        for k, v in meta.items():
            p = m.metadata_props.add()
            p.key, p.value = k, str(v)
        onnx.save(m, path)

    # 1) acoustic ONNX per epoch + metadata
    produced = []
    for ep in eps:
        ckpt = f"{RUN_DIR}/checkpoints/twi_epoch={int(ep):03d}.ckpt"
        if not os.path.exists(ckpt):
            print(f"[onnx] SKIP epoch {ep}: {ckpt} missing")
            continue
        out = f"{ONNX_DIR}/twi_ep{int(ep):03d}_steps{n_timesteps}.onnx"
        print(f"[onnx] exporting epoch {ep} -> {out}")
        subprocess.run(
            ["python", "-m", "matcha.onnx.export", ckpt, out, "--n-timesteps", str(n_timesteps)],
            cwd="/root/Matcha-TTS", check=True,
        )
        add_meta(out, {"epoch": int(ep)})
        produced.append(out)

    # 2) tokens.txt from symbols.py (id == index; space symbol -> "  <id>")
    with open(f"{ONNX_DIR}/tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(symbols):
            f.write(f"{s} {i}\n")
    print(f"[onnx] wrote tokens.txt ({len(symbols)} symbols)")

    # 3) vocoder ONNX
    voc = f"{ONNX_DIR}/hifigan_v2.onnx"
    if not os.path.exists(voc):
        print("[onnx] downloading vocoder hifigan_v2.onnx")
        urllib.request.urlretrieve(VOCODER_ONNX_URL, voc)

    # 4) espeak-ng-data (bundle so sherpa phonemizes Twi with lfn, self-contained)
    dst = f"{ONNX_DIR}/espeak-ng-data"
    if not os.path.isdir(dst):
        cand = [p for p in ["/usr/share/espeak-ng-data", "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
                             *glob.glob("/usr/**/espeak-ng-data", recursive=True)] if os.path.isdir(p)]
        if cand:
            shutil.copytree(cand[0], dst)
            print(f"[onnx] copied espeak-ng-data from {cand[0]}")
        else:
            print("[onnx] WARNING: espeak-ng-data not found")

    # 5) validate one acoustic onnx runs (x -> mel) with onnxruntime
    if produced:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(produced[0], providers=["CPUExecutionProvider"])
        x = np.random.randint(0, len(symbols), size=(1, 40), dtype=np.int64)
        feeds = {"x": x, "x_lengths": np.array([40], dtype=np.int64), "scales": np.array([0.667, 1.0], dtype=np.float32)}
        outs = sess.run(None, feeds)
        print(f"[onnx] validation OK: {os.path.basename(produced[0])} mel shape={outs[0].shape}")

    volume.commit()

    # 6) push bundle to HF under sherpa-onnx/
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=ONNX_DIR, path_in_repo="sherpa-onnx", repo_id=HF_REPO_ID, repo_type="model")
        print(f"[onnx] pushed sherpa-onnx bundle to https://huggingface.co/{HF_REPO_ID}/tree/main/sherpa-onnx")
    print(f"[onnx] DONE: {len(produced)} acoustic models + vocoder + tokens + espeak-ng-data in {ONNX_DIR}")
    return {"models": [os.path.basename(p) for p in produced]}


HIFIGAN_FT_DIR = f"{TWI_DIR}/hifigan_ft"  # finetuned vocoder checkpoints (g_*/do_*)
INIT_GENERATOR_URL = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000"


@app.function(image=image, gpu="A10G", cpu=8.0, volumes={DATA_DIR: volume}, secrets=[hf_secret], timeout=86400)
def finetune_vocoder(total_steps: int = 200000, checkpoint_interval: int = 5000, val_size: int = 64):
    """Finetune the universal HiFi-GAN on the Twi (wav, precomputed-mel) pairs so the vocoder
    matches the speaker. Resumes from g_*/do_* in HIFIGAN_FT_DIR; commits the volume every few
    minutes; pushes generator checkpoints (g_*) to HF under hifigan-ft/ as they are saved."""
    import glob
    import subprocess
    import threading
    import urllib.request

    os.makedirs(HIFIGAN_FT_DIR, exist_ok=True)

    # init generator (universal hifigan_univ_v1 == g_02500000), cached on the volume
    init_g = f"{HIFIGAN_FT_DIR}/g_02500000"
    if not os.path.exists(init_g):
        print("[hifi-ft] downloading universal generator init")
        urllib.request.urlretrieve(INIT_GENERATOR_URL, init_g)
        volume.commit()

    # build hifigan filelists (ids that have both wav + precomputed mel)
    train_fl, val_fl = f"{HIFIGAN_FT_DIR}/train.txt", f"{HIFIGAN_FT_DIR}/val.txt"
    if not (os.path.exists(train_fl) and os.path.exists(val_fl)):
        ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{MELS_DIR}/*.npy"))
        import random as _r
        _r.Random(1234).shuffle(ids)
        val, train = ids[:val_size], ids[val_size:]
        with open(train_fl, "w") as f:
            f.write("\n".join(train))
        with open(val_fl, "w") as f:
            f.write("\n".join(val))
        volume.commit()
        print(f"[hifi-ft] filelists: train={len(train)} val={len(val)}")

    # background: commit volume + push new g_* checkpoints to HF
    stop = threading.Event()
    token = os.environ.get("HF_TOKEN")
    pushed = set()

    def bg():
        from huggingface_hub import HfApi
        api = HfApi(token=token) if token else None
        if api:
            api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=True, exist_ok=True)
        while not stop.wait(180):
            try:
                volume.commit()
                if api:
                    for g in sorted(glob.glob(f"{HIFIGAN_FT_DIR}/g_*")):
                        if g not in pushed:
                            api.upload_file(path_or_fileobj=g, path_in_repo=f"hifigan-ft/{os.path.basename(g)}",
                                            repo_id=HF_REPO_ID, repo_type="model")
                            pushed.add(g)
                            print(f"[hifi-ft] pushed {os.path.basename(g)} to HF")
            except Exception as e:  # pylint: disable=broad-except
                print(f"[hifi-ft] bg error: {e}")

    threading.Thread(target=bg, daemon=True).start()

    cmd = [
        "python", "-m", "matcha.hifigan.train",
        "--input-wavs-dir", WAVS_DIR, "--input-mels-dir", MELS_DIR,
        "--train-filelist", train_fl, "--val-filelist", val_fl,
        "--checkpoint-path", HIFIGAN_FT_DIR, "--init-generator", init_g,
        "--total-steps", str(total_steps), "--checkpoint-interval", str(checkpoint_interval),
    ]
    try:
        subprocess.run(cmd, cwd="/root/Matcha-TTS", env={**os.environ}, check=True)
    finally:
        stop.set()
        volume.commit()
    print("[hifi-ft] training finished")


# ------------------------------------------------------------------- local entrypoint
@app.local_entrypoint()
def main(max_epochs: int = 500):
    """Run the full pipeline end to end."""
    prep_data.remote()
    build_filelists.remote()
    compute_stats.remote()
    train.remote(max_epochs=max_epochs)
