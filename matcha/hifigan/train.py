"""Single-GPU HiFi-GAN finetuning for Matcha-TTS Twi.

Finetunes the universal HiFi-GAN (hifigan_univ_v1) on the Twi (wav, mel) pairs so the vocoder
matches the target speaker's timbre. Uses fine_tuning mode: the generator input is the precomputed
matcha mel (.npy in --input-mels-dir) and the target is the wav (--input-wavs-dir). Inits the
generator from --init-generator with FRESH discriminators (the universal release only ships the
generator); resumes from g_*/do_* checkpoints in --checkpoint-path if present.
"""
import argparse
import glob
import itertools
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matcha.hifigan.config import v1
from matcha.hifigan.env import AttrDict
from matcha.hifigan.meldataset import MelDataset, mel_spectrogram
from matcha.hifigan.models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)


def scan_latest(cp_dir, prefix):
    files = glob.glob(os.path.join(cp_dir, prefix + "*"))
    if not files:
        return None
    return max(files, key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))


def load_filelist(path, wavs_dir):
    with open(path, encoding="utf-8") as f:
        return [os.path.join(wavs_dir, ln.strip().split("|")[0] + ".wav") for ln in f if ln.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-wavs-dir", required=True)
    p.add_argument("--input-mels-dir", required=True)
    p.add_argument("--train-filelist", required=True)
    p.add_argument("--val-filelist", required=True)
    p.add_argument("--checkpoint-path", required=True, help="output dir for g_*/do_* checkpoints")
    p.add_argument("--init-generator", default=None, help="generator ckpt to finetune from (g_02500000)")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    p.add_argument("--stdout-interval", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.0002)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()

    h = AttrDict(v1)
    device = torch.device("cuda")
    os.makedirs(a.checkpoint_path, exist_ok=True)

    gen = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    steps, last_epoch = 0, -1
    g_ckpt, do_ckpt = scan_latest(a.checkpoint_path, "g_"), scan_latest(a.checkpoint_path, "do_")
    if g_ckpt and do_ckpt:  # resume
        print(f"[hifi-ft] resuming from {g_ckpt} / {do_ckpt}")
        sg = torch.load(g_ckpt, map_location=device, weights_only=False)
        gen.load_state_dict(sg["generator"])
        sd = torch.load(do_ckpt, map_location=device, weights_only=False)
        mpd.load_state_dict(sd["mpd"]); msd.load_state_dict(sd["msd"])
        steps, last_epoch = sd["steps"] + 1, sd["epoch"]
    elif a.init_generator:  # finetune init: generator only, fresh discriminators
        print(f"[hifi-ft] init generator from {a.init_generator} (fresh discriminators)")
        sg = torch.load(a.init_generator, map_location=device, weights_only=False)
        gen.load_state_dict(sg["generator"])

    optim_g = torch.optim.AdamW(gen.parameters(), a.learning_rate, betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                a.learning_rate, betas=(h.adam_b1, h.adam_b2))
    if g_ckpt and do_ckpt:
        optim_g.load_state_dict(sd["optim_g"]); optim_d.load_state_dict(sd["optim_d"])
    sch_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    train_files = load_filelist(a.train_filelist, a.input_wavs_dir)
    trainset = MelDataset(
        train_files, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate,
        h.fmin, h.fmax, n_cache_reuse=0, shuffle=True, fmax_loss=h.fmax_loss, device=device,
        fine_tuning=True, base_mels_path=a.input_mels_dir,
    )
    loader = DataLoader(trainset, num_workers=6, shuffle=True, batch_size=a.batch_size,
                        pin_memory=True, drop_last=True)

    def save():
        gp = os.path.join(a.checkpoint_path, f"g_{steps:08d}")
        dp = os.path.join(a.checkpoint_path, f"do_{steps:08d}")
        torch.save({"generator": gen.state_dict()}, gp)
        torch.save({"mpd": mpd.state_dict(), "msd": msd.state_dict(), "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(), "steps": steps, "epoch": epoch}, dp)
        print(f"[hifi-ft] saved {os.path.basename(gp)} / {os.path.basename(dp)}")
        return gp

    gen.train(); mpd.train(); msd.train()
    epoch = max(0, last_epoch)
    done = False
    while not done:
        for x, y, _, y_mel in loader:
            x = x.to(device, non_blocking=True)            # input mel (matcha)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            y_mel = y_mel.to(device, non_blocking=True)     # GT mel for loss

            y_g_hat = gen(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                          h.hop_size, h.win_size, h.fmin, h.fmax_loss, center=False)

            # discriminator
            optim_d.zero_grad()
            yr, yg, _, _ = mpd(y, y_g_hat.detach())
            loss_mpd, _, _ = discriminator_loss(yr, yg)
            yr, yg, _, _ = msd(y, y_g_hat.detach())
            loss_msd, _, _ = discriminator_loss(yr, yg)
            (loss_mpd + loss_msd).backward()
            optim_d.step()

            # generator
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            yr_mpd, yg_mpd, fr_mpd, fg_mpd = mpd(y, y_g_hat)
            yr_msd, yg_msd, fr_msd, fg_msd = msd(y, y_g_hat)
            loss_fm = feature_loss(fr_mpd, fg_mpd) + feature_loss(fr_msd, fg_msd)
            loss_adv, _ = generator_loss(yg_mpd)
            loss_adv2, _ = generator_loss(yg_msd)
            loss_g = loss_adv + loss_adv2 + loss_fm + loss_mel
            loss_g.backward()
            optim_g.step()

            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    merr = F.l1_loss(y_mel, y_g_hat_mel).item()
                print(f"[hifi-ft] step {steps} | g {loss_g.item():.3f} | mel_l1 {merr:.4f} | "
                      f"d {(loss_mpd+loss_msd).item():.3f}", flush=True)

            if steps and steps % a.checkpoint_interval == 0:
                save()

            steps += 1
            if steps >= a.total_steps:
                done = True
                break
        sch_g.step(); sch_d.step()
        epoch += 1
    save()
    print("[hifi-ft] DONE")


if __name__ == "__main__":
    main()
