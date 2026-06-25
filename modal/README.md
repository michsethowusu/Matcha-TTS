# Finetuning Matcha-TTS on Asante Twi (Modal)

Finetunes Matcha-TTS on the **Asante Twi** subset of
[`michsethowusu/ghana-speech`](https://huggingface.co/datasets/michsethowusu/ghana-speech)
(`Asante_Twi_twi`, ~143k clips), starting from the English LJSpeech checkpoint.

## Why this works

- **Phonemizer:** espeak-ng has no Akan/Twi voice, so we use the **`lfn`** (Lingua Franca Nova)
  voice, which maps Twi orthography (incl. `ɛ`/`ɔ`) to clean IPA. See `twi_cleaners` in
  `matcha/text/cleaners.py`.
- **Finetuning (not from scratch):** the IPA that `lfn` emits is drawn entirely from the
  existing 178-symbol set in `matcha/text/symbols.py`, which equals the model's `n_vocab`. So the
  English LJSpeech checkpoint loads **weight-for-weight** and only needs to adapt to Twi.
- **No voice cloning.** Matcha is single/multi-speaker via a learned speaker-ID embedding; it
  cannot clone an unseen voice from reference audio. This produces one Twi voice.

## One-time setup

```bash
pip install modal && modal token new          # if not already authenticated
# HF token needs read (dataset) + write (push checkpoints) scope:
modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
```

Checkpoints are pushed to `$HF_REPO_ID` (default `michsethowusu/matcha-twi`). Override per run:
add `.env({"HF_REPO_ID": "you/your-repo"})` is already baked into the image; to change it, edit
`HF_REPO_ID` at the top of `twi_pipeline.py`.

## Run

Whole pipeline:

```bash
modal run modal/twi_pipeline.py
```

Or stage by stage (each is **resumable** — just re-run after any crash/timeout):

```bash
modal run modal/twi_pipeline.py::prep_data        # stream + filter (1–15s) + resample 22.05kHz + phonemize
modal run modal/twi_pipeline.py::build_filelists  # manifest.jsonl -> train.txt / val.txt
modal run modal/twi_pipeline.py::compute_stats    # mel mean/std -> twi.json
modal run modal/twi_pipeline.py::train            # finetune; pushes every checkpoint to HF
```

## Resumability — how state survives a crash

Everything lives on the Modal Volume **`matcha-twi`** (mounted at `/data`):

| Path | Stage | Purpose |
|------|-------|---------|
| `/data/twi/wavs/*.wav` | prep | 22.05 kHz mono clips |
| `/data/twi/manifest.jsonl` | prep | one line per kept clip (`id`, `wav`, `phon`, `dur`) |
| `/data/twi/state.json` | prep | `{seen, kept}` — drives `dataset.skip()` on resume |
| `/data/twi/train.txt`, `val.txt` | build_filelists | `wav_path\|ipa_phonemes` |
| `/data/twi/twi.json` | compute_stats | `mel_mean` / `mel_std` |
| `/data/pretrained/matcha_ljspeech.ckpt` | train | cached English init |
| `/data/twi/runs/twi/checkpoints/last.ckpt` | train | resume point |

- **prep** commits every 500 kept clips; re-running skips already-consumed source rows.
- **train** commits the volume every 3 min and resumes from `last.ckpt` automatically when
  present (full optimizer/epoch state). On a fresh run it loads the English weights instead
  (fresh optimizer, epoch 0). Every saved checkpoint is uploaded to the HF Hub by
  `HFModelCheckpoint`, so progress is safe even if the volume is lost.

## Tunables

- GPU: `gpu="A10G"` in the `train` function (24 GB is plenty for batch 32).
- Duration filter: `MIN_SECONDS` / `MAX_SECONDS` at the top of `twi_pipeline.py`.
- Checkpoint frequency / retention: `configs/callbacks/twi.yaml` (`every_n_epochs`, `save_top_k`).
- Epochs: `modal run modal/twi_pipeline.py::train --max-epochs 800`.

## Known caveats

- Source verse numbers (e.g. leading `1` in "1 BERƐSOSƐM 1.") get phonemized as `lfn`
  number-words. If they aren't spoken in the audio, strip them in `prep_data` before
  `twi_cleaners` (add a regex in the filter block).
- After training, synthesis needs a vocoder. The default HiFi-GAN (`hifigan_univ_v1`) works but
  is English-trained; for best Twi quality, finetune HiFi-GAN on the same audio later.
```
