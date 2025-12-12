"""
Lightweight pipeline module for Hugging Face Spaces.
- denoise_chunked_final: denoise an input WAV using your SEGAN UNet generator checkpoint (local checkpoint path expected)
- save_audio_as_png_lossless / load_audio_from_png_lossless: lossless packing of int16 PCM into a PNG
- write_wav_from_tensor: helper to write restored wav

This file expects SEGAN.py to be present in the repo root and define:
- Config (with attributes sample_rate, n_fft, hop_length, win_length)
- STFTMagTransform
- UNetGenerator

Adjust CHUNK_SECONDS / device as desired. The checkpoint file must be at checkpoint/seagan_final.pt (tracked with git-lfs).
"""

import os
import math
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image

# try to import SEGAN model code from repo root
try:
    from SEGAN import Config, STFTMagTransform, UNetGenerator
except Exception as e:
    raise ImportError("Failed to import SEGAN module. Make sure SEGAN.py is present and defines Config, STFTMagTransform, UNetGenerator.")

# ------------------ InferConfig ------------------
class InferConfig(Config):
    def __init__(self, ckpt_path="checkpoint/seagan_final.pt"):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ audio I/O helpers ------------------

def load_mono_resampled(path: str, target_sr: int):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0)


def robust_save(path: str, wav_tensor: torch.Tensor, sr: int):
    x = wav_tensor.detach().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    while x.dim() > 2 and x.size(0) == 1:
        x = x.squeeze(0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.float()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torchaudio.save(path, x, sr)

# ------------------ PNG packing helpers ------------------

def save_audio_as_png_lossless(tensor: torch.Tensor, png_path: str, width: int = 2048):
    """
    Packs a 1D int16 PCM tensor into a PNG image as uint16 pixels (grayscale).
    width: target image width in pixels. Height computed ceil(len/width).
    Returns saved png_path.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype == np.int32:
        arr = arr.astype(np.int16)
    elif arr.dtype == np.int16:
        pass
    else:
        arr = arr.astype(np.int16)

    length = arr.shape[0]
    height = math.ceil(length / width)
    pad = height * width - length
    if pad > 0:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.int16)])
    img = arr.reshape((height, width))
    img_u16 = (img.astype(np.int32) + 32768).astype(np.uint16)
    im = Image.fromarray(img_u16, mode='I;16')
    os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
    im.save(png_path, format='PNG')
    return png_path


def load_audio_from_png_lossless(png_path: str, original_length: int = None):
    im = Image.open(png_path)
    arr = np.array(im, dtype=np.uint16)
    arr = arr.astype(np.int32) - 32768
    flat = arr.reshape(-1)
    if original_length is not None:
        flat = flat[:original_length]
    return torch.from_numpy(flat.astype(np.int16)).float() / 32767.0


def write_wav_from_tensor(tensor: torch.Tensor, out_wav_path: str, sr: int):
    if tensor.dim() == 1:
        wav = tensor.unsqueeze(0)
    else:
        wav = tensor
    os.makedirs(os.path.dirname(out_wav_path) or '.', exist_ok=True)
    torchaudio.save(out_wav_path, wav, sr)
    return out_wav_path

# ------------------ simplified denoiser ------------------

def denoise_chunked_final(input_path: str, output_path: str, cfg: InferConfig,
                          chunk_seconds: float = 50.0, overlap: float = 0.5,
                          use_spectral_gate: bool = True, noise_frac: float = 0.1, subtract_strength: float = 1.0,
                          pack_png: bool = False, png_width: int = 2048):
    device = cfg.device
    print("Device:", device)
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    G = UNetGenerator(in_ch=1, out_ch=1).to(device)
    if 'G_state' in ckpt:
        G.load_state_dict(ckpt['G_state'])
    elif 'model_state' in ckpt:
        G.load_state_dict(ckpt['model_state'])
    else:
        G.load_state_dict(ckpt)
    G.eval()

    stft = STFTMagTransform(cfg.n_fft, cfg.hop_length, cfg.win_length).to(device)
    window = stft.window.to(device)

    wav = load_mono_resampled(input_path, cfg.sample_rate)
    T = wav.shape[0]
    sr = cfg.sample_rate
    chunk_samples = max(1, int(chunk_seconds * sr))
    hop = max(1, int(chunk_samples * (1.0 - overlap)))

    out_len = T + chunk_samples
    out_buffer = torch.zeros(out_len, dtype=torch.float32)
    weight_buffer = torch.zeros(out_len, dtype=torch.float32)
    synth_win = torch.hann_window(chunk_samples, periodic=True, dtype=torch.float32)

    idx = 0
    while idx < T:
        start = idx
        end = min(idx + chunk_samples, T)
        chunk = wav[start:end].unsqueeze(0).unsqueeze(0).to(device)
        L = chunk.shape[-1]
        if L < chunk_samples:
            need = chunk_samples - L
            frag = chunk[..., -min(L, need):].flip(-1)
            chunk = torch.cat([chunk, frag], dim=-1)
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
        with torch.no_grad():
            spec = stft(chunk)
            fake = G(spec)
            mag = torch.expm1(fake.clamp_min(0.0)).squeeze(1)
        chunk_1d = chunk.view(1, -1)
        complex_noisy = torch.stft(chunk_1d, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                   win_length=cfg.win_length, window=window, return_complex=True)
        phase = torch.angle(complex_noisy)
        n_frames = min(mag.shape[-1], phase.shape[-1])
        mag = mag[..., :n_frames]
        phase = phase[..., :n_frames]
        expected_F = cfg.n_fft // 2 + 1
        if mag.shape[1] < expected_F:
            mag = F.pad(mag, (0,0,0, expected_F - mag.shape[1]))
        elif mag.shape[1] > expected_F:
            mag = mag[:, :expected_F, :]
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        complex_spec = torch.complex(real, imag).squeeze(0)
        wav_rec = torch.istft(complex_spec.unsqueeze(0).to(device), n_fft=cfg.n_fft,
                              hop_length=cfg.hop_length, win_length=cfg.win_length,
                              window=window, length=chunk_samples).squeeze(0).cpu()
        if wav_rec.shape[-1] < chunk_samples:
            wav_rec = F.pad(wav_rec, (0, chunk_samples - wav_rec.shape[-1]))
        elif wav_rec.shape[-1] > chunk_samples:
            wav_rec = wav_rec[:chunk_samples]
        win = synth_win.clone().cpu()
        wav_rec_win = wav_rec * win
        write_start = start
        write_end = start + chunk_samples
        out_buffer[write_start:write_end] += wav_rec_win
        weight_buffer[write_start:write_end] += win
        idx += hop

    nonzero = weight_buffer > 1e-8
    out_buffer[nonzero] = out_buffer[nonzero] / weight_buffer[nonzero]
    denoised = out_buffer[:T].contiguous()

    denoised = torch.clamp(denoised, -0.999, 0.999)
    robust_save(output_path, denoised, sr)

    if pack_png:
        pcm = (denoised.numpy() * 32767.0).astype(np.int16)
        png_path = os.path.splitext(output_path)[0] + "_packed.png"
        save_audio_as_png_lossless(pcm, png_path, width=png_width)
        return output_path, png_path

    return output_path
