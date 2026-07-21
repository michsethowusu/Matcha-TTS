#!/usr/bin/env python3
"""Finetune the pretrained Vocos mel->waveform vocoder (BSC-LT/vocos-mel-22khz) on the
Ghana-speech data so it renders these voices more faithfully.

Correct by construction (the old matcha/vocos/train.py was broken):
  - Input mel is the STORED raw log-mel (exactly what Matcha's synthesise() feeds the
    vocoder). No spurious denorm/re-log.
  - Only 16 kHz audio exists on disk; it is resampled to 22.05 kHz on the fly to match the
    mel's frame grid (mel fmax=8000 => the 16 kHz source already holds all mel-relevant
    content, so this is consistent; output is 22.05 kHz band-limited to 8 kHz).

Generator = Vocos backbone+head (warm-started from pretrained). Discriminators = MPD+MRD.
Losses = mel-recon (L1) + adversarial + feature-matching. Early-stops on val mel-recon L1.
Pushes {backbone,head} checkpoints to HF each improved epoch.
"""
import argparse, os, yaml
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download, HfApi
from vocos.pretrained import instantiate_class
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HOP = 256
SR = 22050


def read_stems(filelist, mels_dir, wavs_dir):
    stems = []
    with open(filelist, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            stem = Path(ln.split("|")[0]).stem
            if (mels_dir / f"{stem}.npy").exists() and (wavs_dir / f"{stem}.wav").exists():
                stems.append(stem)
    return stems


class VocosDS(Dataset):
    def __init__(self, stems, wavs_dir, mels_dir, segment_frames):
        self.stems, self.wavs_dir, self.mels_dir, self.seg = stems, Path(wavs_dir), Path(mels_dir), segment_frames

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        mel = torch.from_numpy(np.load(self.mels_dir / f"{stem}.npy").astype("float32"))  # (80,T) raw log-mel
        a16, sr = sf.read(self.wavs_dir / f"{stem}.wav", dtype="float32")
        audio = torch.from_numpy(a16)
        if sr != SR:
            audio = torchaudio.functional.resample(audio, sr, SR)
        T = mel.shape[-1]
        seg = self.seg
        if T > seg:
            start = int(torch.randint(0, T - seg + 1, (1,)).item())
            mel = mel[:, start:start + seg]
            audio = audio[start * HOP: start * HOP + seg * HOP]
        # pad audio to exactly seg*HOP (resample rounding / tail)
        need = seg * HOP
        if audio.shape[0] < need:
            audio = torch.nn.functional.pad(audio, (0, need - audio.shape[0]))
        else:
            audio = audio[:need]
        return mel, audio


def collate(batch):
    mels = torch.stack([b[0] for b in batch])   # (B,80,seg)
    audios = torch.stack([b[1] for b in batch])  # (B,seg*HOP)
    return mels, audios


def load_generator(pretrained, device):
    cfg = yaml.safe_load(open(hf_hub_download(pretrained, "config.yaml"), encoding="utf-8"))
    state = torch.load(hf_hub_download(pretrained, "pytorch_model.bin"), map_location=device, weights_only=False)
    backbone = instantiate_class((), cfg["backbone"]).to(device)
    head = instantiate_class((), cfg["head"]).to(device)
    backbone.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("backbone.")})
    head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})
    return backbone, head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/volume_d2wey28/data/ghana_speech")
    p.add_argument("--out-dir", default="/mnt/volume_d2wey28/data/vocos_ghana_ft")
    p.add_argument("--pretrained", default="BSC-LT/vocos-mel-22khz")
    p.add_argument("--segment-frames", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--mel-coeff", type=float, default=45.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--smoke", type=int, default=0, help="if >0, run only N train steps + tiny val and exit")
    a = p.parse_args()

    device = torch.device("cuda:0")
    data = Path(a.data_dir); mels_dir = data / "mels"; wavs_dir = data / "wavs_16k"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_stems = read_stems(data / "train_filtered.txt", mels_dir, wavs_dir)
    val_stems = read_stems(data / "val_filtered.txt", mels_dir, wavs_dir)
    print(f"[vocos-ft] train={len(train_stems)} val={len(val_stems)} clips", flush=True)

    tr = DataLoader(VocosDS(train_stems, wavs_dir, mels_dir, a.segment_frames), batch_size=a.batch_size,
                    shuffle=True, num_workers=a.num_workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.num_workers > 0, pin_memory=True)
    va = DataLoader(VocosDS(val_stems, wavs_dir, mels_dir, a.segment_frames), batch_size=a.batch_size,
                    shuffle=False, num_workers=4, collate_fn=collate, persistent_workers=True)

    backbone, head = load_generator(a.pretrained, device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    disc_loss, gen_loss, fm_loss = DiscriminatorLoss(), GeneratorLoss(), FeatureMatchingLoss()
    mel_loss = MelSpecReconstructionLoss(sample_rate=SR).to(device)

    gp = list(backbone.parameters()) + list(head.parameters())
    dp = list(mpd.parameters()) + list(mrd.parameters())
    opt_g = torch.optim.AdamW(gp, a.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dp, a.lr, betas=(0.8, 0.99))

    def gen(mel):
        return head(backbone(mel))

    api = None
    repo = os.environ.get("HF_REPO_ID")
    if os.environ.get("HF_TOKEN") and repo:
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        print(f"[vocos-ft] pushing checkpoints to {repo}", flush=True)

    @torch.inference_mode()
    def validate():
        backbone.eval(); head.eval()
        tot, n = 0.0, 0
        for mel, audio in va:
            mel, audio = mel.to(device), audio.to(device)
            yh = gen(mel).squeeze(1)
            tot += mel_loss(yh, audio).item(); n += 1
            if a.smoke and n >= 2:
                break
        backbone.train(); head.train()
        return tot / max(n, 1)

    best, wait, step = float("inf"), 0, 0
    for epoch in range(a.max_epochs):
        backbone.train(); head.train(); mpd.train(); mrd.train()
        for mel, audio in tr:
            mel, audio = mel.to(device), audio.to(device)
            yh = gen(mel).squeeze(1)  # (B, seg*HOP)

            # ---- discriminator ----
            opt_d.zero_grad(set_to_none=True)
            r_mpd, g_mpd, _, _ = mpd(audio, yh.detach())
            r_mrd, g_mrd, _, _ = mrd(audio, yh.detach())
            l_d = disc_loss(r_mpd, g_mpd)[0] + disc_loss(r_mrd, g_mrd)[0]
            l_d.backward(); opt_d.step()

            # ---- generator ----
            opt_g.zero_grad(set_to_none=True)
            r_mpd, g_mpd, fr_mpd, fg_mpd = mpd(audio, yh)
            r_mrd, g_mrd, fr_mrd, fg_mrd = mrd(audio, yh)
            l_adv = gen_loss(g_mpd)[0] + gen_loss(g_mrd)[0]
            l_fm = fm_loss(fr_mpd, fg_mpd) + fm_loss(fr_mrd, fg_mrd)
            l_mel = mel_loss(yh, audio)
            l_g = l_adv + l_fm + a.mel_coeff * l_mel
            l_g.backward(); opt_g.step()

            step += 1
            if step % 50 == 0:
                print(f"[vocos-ft] e{epoch} step {step}: mel={l_mel.item():.3f} adv={l_adv.item():.3f} "
                      f"fm={l_fm.item():.3f} d={l_d.item():.3f}", flush=True)
            if a.smoke and step >= a.smoke:
                v = validate()
                print(f"[vocos-ft] SMOKE ok: {step} steps, val mel-L1={v:.3f}", flush=True)
                return

        v = validate()
        improved = v < best - 1e-4
        print(f"[vocos-ft] epoch {epoch} done: val mel-L1={v:.4f} (best {best:.4f}) {'IMPROVED' if improved else ''}", flush=True)
        if improved:
            best, wait = v, 0
            ck = out / f"vocos_epoch{epoch:03d}.pt"
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "pretrained": a.pretrained, "epoch": epoch, "val_mel_l1": v}, ck)
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "pretrained": a.pretrained, "epoch": epoch, "val_mel_l1": v}, out / "last.pt")
            if api:
                for f in (ck, out / "last.pt"):
                    try:
                        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo,
                                        repo_type="model", commit_message=f"vocos ft epoch {epoch} val {v:.4f}")
                    except Exception as e:
                        print(f"[vocos-ft] HF upload failed ({e}); continuing", flush=True)
        else:
            wait += 1
            if wait >= a.patience:
                print(f"[vocos-ft] early stop at epoch {epoch} (best val mel-L1={best:.4f})", flush=True)
                break
    print("[vocos-ft] done", flush=True)


if __name__ == "__main__":
    main()
