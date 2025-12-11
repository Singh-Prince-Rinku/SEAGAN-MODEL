#!/usr/bin/env python3
"""
SEAGAN-style Speech Enhancement (Noise Removal) Training Script

- Generator: U-Net on log-magnitude spectrograms
- Discriminator: PatchGAN-style conditional (noisy + clean/enhanced)
- Loss: L1 (reconstruction) + adversarial (LSGAN)

Requirements:
    pip install torch torchaudio numpy
"""

import os
import glob
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio

# ==========================
#       CONFIG
# ==========================

class Config:
    # Paths (CHANGE THESE TO YOUR FOLDERS)
    noisy_dir = r"E:\Minor-Project-For-Amity-Patna\Models\Audio Data\Noisy Data"   # noisy wavs
    clean_dir = r"E:\Minor-Project-For-Amity-Patna\Models\Audio Data\Noiseless Data"   # clean wavs
    save_dir = r"E:\Minor-Project-For-Amity-Patna\Model SEGAN\checkpoints_seagan"

    # Audio
    sample_rate = 16000
    segment_seconds = 1.0       # train on 1-second chunks
    mono = True

    # STFT / Spectrogram
    n_fft = 512
    hop_length = 128
    win_length = 512

    # Training
    batch_size = 8
    num_workers = 2
    num_epochs = 50
    lr_g = 2e-4
    lr_d = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    lambda_l1 = 100.0  # weight for L1 loss vs GAN loss (like pix2pix)
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()


# ==========================
#       DATASET
# ==========================

def list_wav_pairs(noisy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    pairs = []
    for nf in noisy_files:
        name = os.path.basename(nf)
        cf = os.path.join(clean_dir, name)
        if os.path.exists(cf):
            pairs.append((nf, cf))
    return pairs


class SEAGANDataset(Dataset):
    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)

        self.pairs = list_wav_pairs(noisy_dir, clean_dir)
        if len(self.pairs) == 0:
            raise RuntimeError("No paired .wav files found! Check your folders & names.")

        self.resampler_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _get_resampler(self, orig_sr: int):
        if orig_sr == self.sample_rate:
            return None
        if orig_sr not in self.resampler_cache:
            self.resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self.resampler_cache[orig_sr]

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # shape: (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        return wav  # (1, samples)

    def _aligned_random_crop(self, noisy: torch.Tensor, clean: torch.Tensor):
        """
        Crop noisy and clean with the same start index for alignment.
        noisy, clean: (1, T)
        """
        T = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:, :T]
        clean = clean[:, :T]

        if T <= self.segment_samples:
            pad = self.segment_samples - T
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))
            return noisy, clean
        else:
            start = random.randint(0, T - self.segment_samples)
            end = start + self.segment_samples
            return noisy[:, start:end], clean[:, start:end]

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]

        noisy = self._load_audio(noisy_path)
        clean = self._load_audio(clean_path)

        noisy, clean = self._aligned_random_crop(noisy, clean)

        return noisy, clean


# ==========================
#   SPECTROGRAM HELPERS
# ==========================

class STFTMagTransform(nn.Module):
    """
    Convert waveform -> log-magnitude spectrogram
    """

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # register window so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T)
        return: (B, 1, F, T_spec)
        """
        B, C, T = wav.shape

        wav = wav.view(B * C, T)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(spec)  # (B*C, F, T_spec)
        log_mag = torch.log1p(mag)  # log(1 + mag)
        log_mag = log_mag.view(B, C, log_mag.shape[1], log_mag.shape[2])
        return log_mag


# ==========================
#   SIZE MATCH HELPER
# ==========================

def match_size(a: torch.Tensor, b: torch.Tensor):
    """
    Crop a and b to have the same (H, W). Keeps the top-left region.
    a, b: (..., H, W)
    returns: (a_crop, b_crop)
    """
    Ha, Wa = a.shape[-2], a.shape[-1]
    Hb, Wb = b.shape[-2], b.shape[-1]
    H = min(Ha, Hb)
    W = min(Wa, Wb)
    a_c = a[..., :H, :W]
    b_c = b[..., :H, :W]
    return a_c, b_c


# ==========================
#      GENERATOR (U-NET)
# ==========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]

        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net operating on (B, 1, F, T) log-magnitude spectrograms
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_ch, base_ch, down=True, use_bn=False)  # (64)
        self.down2 = ConvBlock(base_ch, base_ch * 2)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.down4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.down5 = ConvBlock(base_ch * 8, base_ch * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up1 = ConvBlock(base_ch * 8, base_ch * 8, down=False)
        self.up2 = ConvBlock(base_ch * 8 * 2, base_ch * 8, down=False)
        self.up3 = ConvBlock(base_ch * 8 * 2, base_ch * 4, down=False)
        self.up4 = ConvBlock(base_ch * 4 * 2, base_ch * 2, down=False)
        self.up5 = ConvBlock(base_ch * 2 * 2, base_ch, down=False)

        self.final = nn.ConvTranspose2d(
            base_ch * 2, out_ch, kernel_size=4, stride=2, padding=1
        )
        # Output non-negative log-magnitude
        self.out_act = nn.ReLU()

    def _crop_to(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Center-crop src to have the same H, W as tgt.
        src: (B, C, Hs, Ws)
        tgt: (B, C, Ht, Wt) (only Ht, Wt are used)
        """
        _, _, Hs, Ws = src.shape
        _, _, Ht, Wt = tgt.shape

        if Hs == Ht and Ws == Wt:
            return src

        start_h = max((Hs - Ht) // 2, 0)
        start_w = max((Ws - Wt) // 2, 0)
        end_h = start_h + Ht
        end_w = start_w + Wt

        return src[:, :, start_h:end_h, start_w:end_w]

    def forward(self, x):
        # encoder
        d1 = self.down1(x)  # B,64
        d2 = self.down2(d1) # B,128
        d3 = self.down3(d2) # B,256
        d4 = self.down4(d3) # B,512
        d5 = self.down5(d4) # B,512

        bott = self.bottleneck(d5)

        # decoder with crops + skips
        u1 = self.up1(bott)
        d5_c = self._crop_to(d5, u1)
        u1 = torch.cat([u1, d5_c], dim=1)

        u2 = self.up2(u1)
        d4_c = self._crop_to(d4, u2)
        u2 = torch.cat([u2, d4_c], dim=1)

        u3 = self.up3(u2)
        d3_c = self._crop_to(d3, u3)
        u3 = torch.cat([u3, d3_c], dim=1)

        u4 = self.up4(u3)
        d2_c = self._crop_to(d2, u4)
        u4 = torch.cat([u4, d2_c], dim=1)

        u5 = self.up5(u4)
        d1_c = self._crop_to(d1, u5)
        u5 = torch.cat([u5, d1_c], dim=1)

        out = self.final(u5)
        out = self.out_act(out)  # non-negative log-magnitude
        return out


# ==========================
#   DISCRIMINATOR (PatchGAN)
# ==========================

class PatchDiscriminator(nn.Module):
    """
    Conditional discriminator: input = concat(noisy_spec, clean_or_fake_spec)
    """
    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        # no batchnorm in first layer
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1),
            # no activation -> LSGAN
        )

    def forward(self, x):
        return self.model(x)  # (B, 1, H', W')


# ==========================
#         TRAINING
# ==========================

def save_checkpoint(epoch, G, D, opt_g, opt_d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def train():
    device = cfg.device
    print(f"Using device: {device}")

    dataset = SEAGANDataset(
        cfg.noisy_dir, cfg.clean_dir, cfg.sample_rate, cfg.segment_seconds
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    stft_transform = STFTMagTransform(
        cfg.n_fft, cfg.hop_length, cfg.win_length
    ).to(device)

    G = UNetGenerator(in_ch=1, out_ch=1).to(device)
    D = PatchDiscriminator(in_ch=2).to(device)

    # LSGAN loss
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    opt_g = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    for epoch in range(1, cfg.num_epochs + 1):
        G.train()
        D.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for i, (noisy_wav, clean_wav) in enumerate(loader):
            noisy_wav = noisy_wav.to(device)  # (B,1,T)
            clean_wav = clean_wav.to(device)  # (B,1,T)

            # -------------------------
            #   Waveform -> Spectrogram
            # -------------------------
            noisy_spec = stft_transform(noisy_wav)  # (B,1,F,T_spec)
            clean_spec = stft_transform(clean_wav)  # (B,1,F,T_spec)

            # Ensure same size for real pair
            noisy_spec, clean_spec = match_size(noisy_spec, clean_spec)

            # =========================
            #   Train Discriminator
            # =========================
            opt_d.zero_grad()

            # Real pair: (noisy, clean)
            real_input = torch.cat([noisy_spec, clean_spec], dim=1)
            pred_real = D(real_input)
            target_real = torch.ones_like(pred_real)
            loss_d_real = criterion_gan(pred_real, target_real)

            # Fake pair: (noisy, enhanced)
            with torch.no_grad():
                fake_spec = G(noisy_spec)
            # match noisy and fake sizes
            noisy_for_fake_d, fake_spec_d = match_size(noisy_spec, fake_spec)
            fake_input = torch.cat([noisy_for_fake_d, fake_spec_d], dim=1)
            pred_fake = D(fake_input)
            target_fake = torch.zeros_like(pred_fake)
            loss_d_fake = criterion_gan(pred_fake, target_fake)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # =========================
            #     Train Generator
            # =========================
            opt_g.zero_grad()

            fake_spec = G(noisy_spec)

            # GAN loss (want D(noisy, fake) = 1)
            noisy_for_fake_g, fake_spec_g = match_size(noisy_spec, fake_spec)
            fake_input_g = torch.cat([noisy_for_fake_g, fake_spec_g], dim=1)
            pred_fake_for_g = D(fake_input_g)
            target_real_for_g = torch.ones_like(pred_fake_for_g)
            loss_g_gan = criterion_gan(pred_fake_for_g, target_real_for_g)

            # L1 reconstruction loss (match fake & clean sizes)
            fake_l1, clean_l1 = match_size(fake_spec, clean_spec)
            loss_g_l1 = criterion_l1(fake_l1, clean_l1) * cfg.lambda_l1

            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()

            running_d_loss += loss_d.item()
            running_g_loss += loss_g.item()

            if (i + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch}/{cfg.num_epochs}] "
                    f"Step [{i+1}/{len(loader)}] "
                    f"D Loss: {loss_d.item():.4f}  "
                    f"G Loss: {loss_g.item():.4f}  "
                    f"(GAN: {loss_g_gan.item():.4f}, L1: {loss_g_l1.item():.4f})"
                )

        avg_d = running_d_loss / len(loader)
        avg_g = running_g_loss / len(loader)
        print(
            f"==> Epoch {epoch} finished | "
            f"Avg D Loss: {avg_d:.4f} | Avg G Loss: {avg_g:.4f}"
        )

        # save checkpoint every few epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"seagan_epoch_{epoch}.pt")
            save_checkpoint(epoch, G, D, opt_g, opt_d, ckpt_path)

    # final save
    ckpt_path = os.path.join(cfg.save_dir, f"seagan_final.pt")
    save_checkpoint(cfg.num_epochs, G, D, opt_g, opt_d, ckpt_path)


if __name__ == "__main__":
    train()
