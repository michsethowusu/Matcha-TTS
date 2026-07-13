#!/usr/bin/env python3
"""Finetune a pretrained Vocos mel-to-waveform vocoder on a target speaker/dataset.

This script is designed for the Twi Matcha-TTS pipeline. It loads a pretrained Vocos
model (e.g. BSC-LT/vocos-mel-22khz, whose mel params match Matcha's), and finetunes it
on precomputed Matcha mel-spectrograms so the vocoder inverts the acoustic model's
output at deployment time.

Inputs:
  --input-mels-dir: directory with precomputed, normalised Matcha mel .npy files
  --input-wavs-dir: directory with corresponding 22.05 kHz mono .wav files
  --stats-json:     mel_mean / mel_std used to denormalise the input mels
  --checkpoint-path: output directory for checkpoints
  --pretrained:     Hugging Face repo_id of the pretrained Vocos (default BSC-LT/vocos-mel-22khz)

Training uses the standard Vocos generator + multi-period/multi-resolution
discriminators. Early stopping is applied with patience=5 on validation mel loss.

Example:
  python -m matcha.vocos.train \
    --input-wavs-dir /mnt/volume_d2wey28/data/twi_new/wavs \
    --input-mels-dir /mnt/volume_d2wey28/data/twi_new/mels \
    --stats-json /mnt/volume_d2wey28/data/twi_new/twi_new.json \
    --checkpoint-path /mnt/volume_d2wey28/projects/matcha-twi/outputs/vocos_twi_new \
    --pretrained BSC-LT/vocos-mel-22khz
"""
import argparse
import glob
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss

from matcha.utils.audio import mel_spectrogram


def scan_latest(cp_dir, prefix):
    files = glob.glob(os.path.join(cp_dir, prefix + "*"))
    if not files:
        return None
    return max(files, key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))


def load_filelist(path, wavs_dir):
    with open(path, encoding="utf-8") as f:
        return [os.path.join(wavs_dir, ln.strip().split("|")[0] + ".wav") for ln in f if ln.strip()]


class VocosFinetuneDataset(Dataset):
    """Dataset that yields (log_denorm_mel, audio) pairs from precomputed Matcha mels."""

    def __init__(self, ids, wavs_dir, mels_dir, stats, segment_size=None):
        self.ids = ids
        self.wavs_dir = Path(wavs_dir)
        self.mels_dir = Path(mels_dir)
        self.mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)
        self.mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)
        self.segment_size = segment_size  # frames; if set, random crop each epoch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        clip_id = self.ids[i]
        mel_path = self.mels_dir / f"{clip_id}.npy"
        wav_path = self.wavs_dir / f"{clip_id}.wav"

        mel = torch.from_numpy(np.load(mel_path).astype(np.float32))  # (n_mels, T)
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == 22050
        audio = torch.from_numpy(audio)

        # Denormalise to raw mel, then log.
        mel = mel * self.mel_std.unsqueeze(-1) + self.mel_mean.unsqueeze(-1)
        mel = torch.log(mel.clamp(min=1e-5))

        if self.segment_size is not None and mel.shape[-1] > self.segment_size:
            max_start = mel.shape[-1] - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item()
            mel = mel[:, start : start + self.segment_size]
            audio_start = start * 256
            audio_len = self.segment_size * 256
            audio = audio[audio_start : audio_start + audio_len]

        return {"mel": mel, "audio": audio, "id": clip_id}


def collate_fn(batch):
    max_mel_len = max(b["mel"].shape[-1] for b in batch)
    max_audio_len = max(b["audio"].shape[0] for b in batch)
    n_mels = batch[0]["mel"].shape[0]
    mels = torch.zeros(len(batch), n_mels, max_mel_len, dtype=torch.float32)
    audios = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
    lengths = []
    for i, b in enumerate(batch):
        mels[i, :, : b["mel"].shape[-1]] = b["mel"]
        audios[i, : b["audio"].shape[0]] = b["audio"]
        lengths.append(b["audio"].shape[0])
    return {"mel": mels, "audio": audios, "lengths": torch.tensor(lengths, dtype=torch.long)}


def matcha_mel_loss(y_hat, y, stats):
    """L1 between Matcha-style (center=False) log mels of predicted and target audio."""
    mel_hat = mel_spectrogram(
        y_hat,
        n_fft=1024,
        num_mels=80,
        sampling_rate=22050,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
        center=False,
    )
    mel = mel_spectrogram(
        y,
        n_fft=1024,
        num_mels=80,
        sampling_rate=22050,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
        center=False,
    )
    return F.l1_loss(mel_hat, mel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-wavs-dir", required=True)
    p.add_argument("--input-mels-dir", required=True)
    p.add_argument("--stats-json", required=True)
    p.add_argument("--train-filelist", required=True)
    p.add_argument("--val-filelist", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--pretrained", default="BSC-LT/vocos-mel-22khz")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--segment-size", type=int, default=172, help="mel frames per training sample")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience")
    p.add_argument("--pretrain-mel-epochs", type=int, default=1, help="train generator with mel loss only for N epochs")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--stdout-interval", type=int, default=100)
    p.add_argument("--mel-loss-coeff", type=float, default=45.0)
    p.add_argument("--mrd-loss-coeff", type=float, default=1.0)
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --checkpoint-path")
    a = p.parse_args()

    device = torch.device("cuda")
    os.makedirs(a.checkpoint_path, exist_ok=True)

    with open(a.stats_json, encoding="utf-8") as f:
        stats = json.load(f)

    # Load pretrained Vocos backbone + head manually. Some community checkpoints (e.g.
    # BSC-LT/vocos-mel-22khz) store a MelSpectrogramFeatures config with extra keys
    # that the installed vocos package does not accept; we only need the backbone and
    # head weights because the input mel is computed offline by Matcha.
    print(f"[vocos-ft] loading pretrained Vocos from {a.pretrained}")
    from huggingface_hub import hf_hub_download
    from vocos.pretrained import instantiate_class

    config_path = hf_hub_download(repo_id=a.pretrained, filename="config.yaml")
    model_path = hf_hub_download(repo_id=a.pretrained, filename="pytorch_model.bin")
    with open(config_path, encoding="utf-8") as f:
        vocos_cfg = yaml.safe_load(f)

    backbone = instantiate_class((), vocos_cfg["backbone"]).to(device)
    head = instantiate_class((), vocos_cfg["head"]).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)

    def filter_prefix(prefix):
        return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

    backbone.load_state_dict(filter_prefix("backbone."))
    head.load_state_dict(filter_prefix("head."))

    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    gen_params = list(backbone.parameters()) + list(head.parameters())
    optim_g = torch.optim.AdamW(gen_params, a.learning_rate, betas=(0.8, 0.9))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), mrd.parameters()), a.learning_rate, betas=(0.8, 0.9)
    )

    disc_loss = DiscriminatorLoss()
    gen_loss = GeneratorLoss()
    feat_loss = FeatureMatchingLoss()
    mel_recon_loss = MelSpecReconstructionLoss(sample_rate=22050)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if a.resume:
        g_ckpt = scan_latest(a.checkpoint_path, "vocos_g_")
        d_ckpt = scan_latest(a.checkpoint_path, "vocos_do_")
        if g_ckpt and d_ckpt:
            print(f"[vocos-ft] resuming from {g_ckpt} / {d_ckpt}")
            sg = torch.load(g_ckpt, map_location=device, weights_only=False)
            backbone.load_state_dict(sg["backbone"])
            head.load_state_dict(sg["head"])
            sd = torch.load(d_ckpt, map_location=device, weights_only=False)
            mpd.load_state_dict(sd["mpd"])
            mrd.load_state_dict(sd["mrd"])
            optim_g.load_state_dict(sg["optim_g"])
            optim_d.load_state_dict(sd["optim_d"])
            start_epoch = sg.get("epoch", 0) + 1
            best_val_loss = sg.get("best_val_loss", float("inf"))
            epochs_without_improvement = sg.get("epochs_without_improvement", 0)

    def load_ids(path):
        with open(path, encoding="utf-8") as f:
            return [os.path.splitext(os.path.basename(ln.strip().split("|")[0]))[0] for ln in f if ln.strip()]

    train_ids = load_ids(a.train_filelist)
    val_ids = load_ids(a.val_filelist)

    trainset = VocosFinetuneDataset(
        train_ids, a.input_wavs_dir, a.input_mels_dir, stats, segment_size=a.segment_size
    )
    valset = VocosFinetuneDataset(
        val_ids, a.input_wavs_dir, a.input_mels_dir, stats, segment_size=None
    )
    train_loader = DataLoader(
        trainset,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    def save_checkpoint(epoch, val_loss, is_best=False):
        g_path = os.path.join(a.checkpoint_path, f"vocos_g_{epoch:04d}")
        d_path = os.path.join(a.checkpoint_path, f"vocos_do_{epoch:04d}")
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optim_g": optim_g.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            },
            g_path,
        )
        torch.save(
            {
                "mpd": mpd.state_dict(),
                "mrd": mrd.state_dict(),
                "optim_d": optim_d.state_dict(),
                "epoch": epoch,
            },
            d_path,
        )
        if is_best:
            best_g = os.path.join(a.checkpoint_path, "vocos_g_best")
            best_d = os.path.join(a.checkpoint_path, "vocos_do_best")
            torch.save(torch.load(g_path, map_location="cpu", weights_only=False), best_g)
            torch.save(torch.load(d_path, map_location="cpu", weights_only=False), best_d)
            print(f"[vocos-ft] new best val_loss={val_loss:.4f}; saved best checkpoint")
        print(f"[vocos-ft] saved {os.path.basename(g_path)} / {os.path.basename(d_path)}")

    @torch.no_grad()
    def validate():
        backbone.eval()
        head.eval()
        total_loss = 0.0
        total_mel = 0.0
        n = 0
        for batch in val_loader:
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["audio"].to(device, non_blocking=True)
            x = backbone(mel)
            y_g_hat = head(x)

            # Crop to target length.
            target_len = y.shape[-1]
            y_g_hat = y_g_hat[..., :target_len]

            mel_l1 = matcha_mel_loss(y_g_hat, y, stats)
            total_mel += mel_l1.item()
            total_loss += mel_l1.item()
            n += 1
        backbone.train()
        head.train()
        return total_loss / max(n, 1), total_mel / max(n, 1)

    global_step = 0
    done = False
    for epoch in range(start_epoch, a.max_epochs):
        if done:
            break

        backbone.train()
        head.train()
        mpd.train()
        mrd.train()

        train_discriminator = epoch >= a.pretrain_mel_epochs

        for batch_idx, batch in enumerate(train_loader):
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["audio"].to(device, non_blocking=True)

            x = backbone(mel)
            y_g_hat = head(x)

            # Crop predicted audio to target length.
            target_len = y.shape[-1]
            y_g_hat = y_g_hat[..., :target_len]

            # --- discriminator ---
            if train_discriminator:
                optim_d.zero_grad()
                real_score_mp, gen_score_mp, _, _ = mpd(y=y, y_hat=y_g_hat.detach())
                real_score_mrd, gen_score_mrd, _, _ = mrd(y=y, y_hat=y_g_hat.detach())
                loss_mp, _, _ = disc_loss(real_score_mp, gen_score_mp)
                loss_mrd, _, _ = disc_loss(real_score_mrd, gen_score_mrd)
                loss_mp /= len(real_score_mp)
                loss_mrd /= len(real_score_mrd)
                loss_d = loss_mp + a.mrd_loss_coeff * loss_mrd
                loss_d.backward()
                optim_d.step()
            else:
                loss_d = torch.tensor(0.0, device=device)

            # --- generator ---
            optim_g.zero_grad()
            mel_l1 = matcha_mel_loss(y_g_hat, y, stats)
            loss_g = a.mel_loss_coeff * mel_l1

            if train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = mpd(y=y, y_hat=y_g_hat)
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(y=y, y_hat=y_g_hat)
                loss_gen_mp, _ = gen_loss(gen_score_mp)
                loss_gen_mrd, _ = gen_loss(gen_score_mrd)
                loss_gen_mp /= len(gen_score_mp)
                loss_gen_mrd /= len(gen_score_mrd)
                loss_fm_mp = feat_loss(fmap_rs_mp, fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = feat_loss(fmap_rs_mrd, fmap_gs_mrd) / len(fmap_rs_mrd)
                loss_g = loss_g + loss_gen_mp + a.mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + a.mrd_loss_coeff * loss_fm_mrd

            loss_g.backward()
            optim_g.step()

            if global_step % a.stdout_interval == 0:
                print(
                    f"[vocos-ft] epoch {epoch} step {global_step} | g {loss_g.item():.3f} | "
                    f"mel_l1 {mel_l1.item():.4f} | d {loss_d.item():.3f}",
                    flush=True,
                )
            global_step += 1

        # Validation + early stopping.
        val_loss, val_mel = validate()
        print(f"[vocos-ft] epoch {epoch} validation | loss {val_loss:.4f} | mel_l1 {val_mel:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(epoch, val_loss, is_best=is_best)

        if epochs_without_improvement >= a.patience:
            print(
                f"[vocos-ft] early stopping triggered after {epochs_without_improvement} epochs "
                f"without improvement (patience={a.patience}). Best val_loss={best_val_loss:.4f}"
            )
            done = True

    print("[vocos-ft] DONE")


if __name__ == "__main__":
    main()
