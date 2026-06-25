# Finetuning Matcha-TTS for a new language (GhanaNLP recipe)

This fork adds an **end-to-end, reproducible pipeline** to finetune Matcha-TTS from the English
LJSpeech checkpoint to a new language, export to **sherpa-onnx**, and ship an offline model. It was
built for **Asante Twi** (→ [ghananlpcommunity/nano-twi](https://huggingface.co/ghananlpcommunity/nano-twi),
demo: [HF Space](https://huggingface.co/spaces/ghananlpcommunity/nano-twi),
usage repo: [michsethowusu/nano-twi](https://github.com/michsethowusu/nano-twi)).

Everything here is designed to be **re-run for other languages** by changing a few config values.

---

## 1. What this fork changes vs upstream

| File | Change | Why |
|------|--------|-----|
| `matcha/text/cleaners.py` | add `twi_cleaners` (espeak **`lfn`** voice → IPA) + `twi_phonemes` (pass-through) | phonemize a language espeak has no native voice for; `twi_phonemes` is used at train time on precomputed phonemes |
| `matcha/train.py` | add `finetune_ckpt` (load **weights only**, fresh optimizer/epoch) + force `torch.load(weights_only=False)` | warm-start from English on new data; PyTorch ≥2.6 changed the `torch.load` default and breaks Lightning ckpts |
| `configs/train.yaml` | add `finetune_ckpt: null` | expose the finetune knob |
| `configs/data/twi.yaml` | data module config (filelists, cleaners, mel params) | per-language data config |
| `configs/experiment/twi.yaml` | experiment config; sets `out_size: 172`, callbacks, epochs | per-language training recipe |
| `configs/callbacks/twi.yaml` | swaps in `HFModelCheckpoint`, every-epoch save, top-k pruning | push checkpoints to HF as training runs |
| `matcha/utils/hf_checkpoint.py` | `HFModelCheckpoint` — uploads each saved ckpt to HF (+ mirrors pruning) | offsite, resumable checkpoints |
| `matcha/data/text_mel_datamodule.py` | load audio with **soundfile**; load **precomputed mels** if present | torchaudio routes to absent torchcodec; precomputed mels make training GPU-bound |
| `matcha/utils/utils.py` | `save_figure_to_numpy` uses `buffer_rgba()` | matplotlib ≥3.8 removed `tostring_rgb()` |
| `matcha/cli.py` | force `torch.load(weights_only=False)` | needed by ONNX export / inference / CLI |
| `matcha/hifigan/meldataset.py` | `librosa_mel_fn(sr=…, n_fft=…, …)` keyword args | modern librosa made these keyword-only |
| `matcha/hifigan/train.py` | **new** — single-GPU HiFi-GAN finetuning loop | optional vocoder finetuning on the target speaker |
| `modal/twi_pipeline.py` | **new** — the whole pipeline on [Modal](https://modal.com) | data prep → mels → stats → train → onnx export → vocoder finetune |
| `modal/README.md` | Modal run instructions | — |

> The compiled `matcha/utils/monotonic_align/core*.so` is **git-ignored**. Build it once locally with
> `cythonize -i matcha/utils/monotonic_align/core.pyx` or, if `numpy.distutils` is broken on your
> Python, compile directly:
> `gcc -shared -fPIC -O2 -I"$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')" -I"$(python3 -c 'import numpy;print(numpy.get_include())')" matcha/utils/monotonic_align/core.c -o matcha/utils/monotonic_align/core.$(python3 -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX")[1:])')`

---

## 2. End-to-end workflow (the exact order we ran)

All stages live in `modal/twi_pipeline.py` and are individually runnable + **resumable** (state on a
Modal Volume). See `modal/README.md` for the commands.

1. **`prep_data`** — stream the HF dataset subset → filter (1–15 s, non-empty text) → resample to
   22.05 kHz mono → phonemize with the espeak `lfn` voice → write `wavs/` + `manifest.jsonl`.
   (`--max-clips N` grabs a subset, e.g. for vocoder finetuning.)
2. **`build_filelists`** — `manifest.jsonl` → `train.txt` / `val.txt` as `wav_path|ipa_phonemes`.
3. **`precompute_mels`** — compute raw mels once → `mels/*.npy` (training then loads these; big speedup).
4. **`compute_stats`** — mel mean/std → `twi.json` (resumable, online accumulators).
5. **`train`** — finetune from the English ckpt (`finetune_ckpt`), `out_size=172`, push every epoch
   to HF. Resumes from `last.ckpt` if present.
6. **`export_onnx`** (or run locally, see §4) — Matcha → acoustic ONNX + sherpa metadata; generate
   `tokens.txt`; fetch a vocoder; bundle a slim `espeak-ng-data/`.
7. **Vocoder** — ship **Vocos** (`vocos-22khz-univ.onnx` from sherpa-onnx) by default, **or** finetune
   HiFi-GAN on the target speaker with **`finetune_vocoder`** (`matcha/hifigan/train.py`).

---

## 3. Configs that mattered

- **Mel (must be identical everywhere — model, mels, vocoder):** `sample_rate 22050`, `n_fft 1024`,
  `hop 256`, `win 1024`, `n_mels 80`, `fmin 0`, `fmax 8000`.
- **`n_vocab = 178`** — unchanged from upstream `symbols.py`. The `lfn` IPA output uses only these
  symbols, so the **English checkpoint loads weight-for-weight** (this is what makes warm-start work).
- `add_blank: True`, `n_spks: 1`.
- **`out_size: 172`** (≈2 s crop for the decoder loss) — the single biggest training speedup.
- `batch_size`: 64 (A100) / 32 (smaller GPUs); `every_n_epochs: 1`, `save_top_k: 5`.
- **ODE steps at export:** `--n-timesteps 4` (default, best quality) and `2` (fast, lower quality).

---

## 4. ONNX export quirks (read before exporting)

- Torch ≥2.9's **dynamo** ONNX exporter can't trace Matcha's ODE-solver loop. Use the **legacy
  TorchScript exporter**: `torch.onnx.export(..., dynamo=False, opset_version=15)`. Requires
  `pip install onnxscript` and the built `monotonic_align` extension.
- **sherpa-onnx metadata** the acoustic ONNX must carry: `model_type=matcha-tts`, `voice=<espeak voice>`,
  `language`, `sample_rate=22050`, `n_speakers=1`, `pad_id=0`, `use_icefall=0`, `use_eos_bos=0`,
  `num_ode_steps=<steps>`, `has_espeak=1`.
- **`tokens.txt` MUST include `^` and `$`** (BOS/EOS) — sherpa's piper-phonemizer wraps every
  utterance in `^…$`; without them sherpa throws `_Map_base::at` on *every* input. Build tokens from
  `matcha.text._symbol_to_id`; the duplicate `'` in `symbols.py` needs a single-char placeholder.
- **sherpa-onnx ≥1.13 removed the TTS command-line tool** — use the **Python API** (`OfflineTts`).
- **espeak-ng-data**: ship the bundle's own copy (built with **eSpeak NG 1.52.0**), pruned to the
  voice essentials (`phondata`, `phonindex`, `phontab`, `phondata-manifest`, `intonations`,
  `<voice>_dict`, `lang/.../<voice>`). Use **sherpa-onnx ≥1.12**.

---

## 5. Adapting to a NEW language

1. **Pick an espeak-ng voice** that best covers the language and check its IPA is in-vocabulary:
   ```bash
   espeak-ng -v <voice> --ipa "some sentence in the language"
   ```
   Confirm every output character is already in `matcha/text/symbols.py` (178 symbols). If there are
   out-of-vocab phonemes, either map them to existing symbols, or extend `symbols.py` — **but note
   that changes `n_vocab`, so you can no longer warm-start cleanly from the English checkpoint**
   (you'd train from scratch or resize the embedding).
2. **Add a cleaner** in `matcha/text/cleaners.py`: copy `twi_cleaners`, change `language="<voice>"`.
3. **Copy configs:** `configs/data/twi.yaml`, `configs/experiment/twi.yaml`, `configs/callbacks/twi.yaml`
   → `<lang>.yaml`, updating filelist paths, `cleaners`, and the HF repo id.
4. **Edit `modal/twi_pipeline.py` constants:** `DATASET`, `SUBSET`, `HF_REPO_ID`, the metadata `voice`
   and `language` in `export_onnx`, and the espeak voice in the cleaner.
5. **Run the pipeline** (§2). Warm-starting from English means you usually have an intelligible voice
   within ~10–20 epochs.

---

## 6. Notes

- Matcha is **single-speaker, non-autoregressive** — no voice cloning, but very stable on long text.
- The phonemizer (espeak-ng) is **embedded in sherpa-onnx** on every platform (desktop/Android/iOS/WASM);
  you only ship the small `espeak-ng-data/` folder as an asset.
- Upstream: <https://github.com/shivammehta25/Matcha-TTS>. This fork keeps upstream training/synthesis
  intact and only **adds** the files in §1.
