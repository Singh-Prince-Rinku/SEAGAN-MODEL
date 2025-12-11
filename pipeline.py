#!/usr/bin/env python3
"""
pipeline.py

Contains:
 - InferConfig (wraps your SEGAN.Config)
 - denoise_chunked_final(...) -> denoised WAV path, packed PNG path, reconstructed WAV path
 - save_audio_as_png_lossless / load_audio_from_png_lossless / write_wav_from_tensor
 - helper utilities used by the denoiser (robust_save, mirror-pad, spectral gating)

Usage: import the functions in your FastAPI `app.py` or run this file directly for a local test.

Note: this module expects your SEGAN.py (containing Config, STFTMagTransform, UNetGenerator)
to be available in the same directory or in PYTHONPATH. Adjust imports if needed.
"""

import os
import math
import wave
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image

# Try to import SEGAN components - user must have SEGAN.py in same folder or package
try:
    from SEGAN import Config, STFTMagTransform, UNetGenerator
except Exception as e:
    # If import fails, raise a clear error when functions are used; keep module importable for tools that
    # just want pack/unpack functions.
    Config = None
    STFTMagTransform = None
    UNetGenerator = None
    _import_error = e


# ----------------- Configuration (defaults) -----------------
DEFAULT_CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "./checkpoints/seagan_final.pt")

# ----------------- Infer config wrapper ---------------------
class InferConfig:
    """Simple wrapper for your SEGAN.Config. If SEGAN.Config is available we use it; else provide defaults.
    Attributes expected by the pipeline: ckpt_path, device, n_fft, hop_length, win_length, sample_rate
    """
    def __init__(self,
                 ckpt_path: str = DEFAULT_CHECKPOINT,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 sample_rate: int = 16000):
        # If real SEGAN.Config exists, instantiate it and override ckpt_path + device
        if Config is not None:
            try:
                cfg = Config()
                cfg.ckpt_path = ckpt_path
                cfg.device = device
                # keep other fields from Config if present
                self.__dict__.update(cfg.__dict__)
                return
            except Exception:
                # fall through to default fields
                pass
        # fallback defaults
        self.ckpt_path = ckpt_path
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate


# ---------------- utilities -------------------

def load_mono_resampled(path: str, target_sr: int):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0)  # (T,)


def robust_save(path: str, wav_tensor: torch.Tensor, sr: int):
    x = wav_tensor.detach().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    while x.dim() > 2 and x.size(0) == 1:
        x = x.squeeze(0)
    if x.dim() > 2:
        x = torch.squeeze(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.float()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, x, sr)
    print(f"Saved WAV: {path} (shape={tuple(x.shape)})")


def pad_or_crop_freq(mag: torch.Tensor, target_F: int):
    F_mag = mag.shape[1]
    if F_mag == target_F:
        return mag
    if F_mag < target_F:
        pad = target_F - F_mag
        return F.pad(mag, (0, 0, 0, pad))
    else:
        return mag[:, :target_F, :]


def mirror_pad_last_chunk(chunk: torch.Tensor, target_len: int):
    L = chunk.shape[-1]
    if L >= target_len:
        return chunk[:, :, :target_len]
    need = target_len - L
    frag = chunk[..., -min(L, need):].flip(-1)
    out = torch.cat([chunk, frag], dim=-1)
    if out.shape[-1] < target_len:
        out = F.pad(out, (0, target_len - out.shape[-1]))
    return out[:, :, :target_len]


# ---------------- spectral gating (final cleanup) ----------------

def spectral_subtract_and_reconstruct(waveform: torch.Tensor, stft_mod, cfg: InferConfig,
                                      noise_frac=0.1, subtract_strength=1.0, device='cpu'):
    if waveform.dim() == 1:
        wav = waveform.unsqueeze(0)  # (1, T)
    else:
        wav = waveform
    wav = wav.to(device)

    n_fft = cfg.n_fft
    hop = cfg.hop_length
    win = stft_mod.window.to(device) if stft_mod is not None else torch.hann_window(cfg.win_length).to(device)

    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=cfg.win_length, window=win, return_complex=True)
    mag = torch.abs(spec)     # (1, F, T)
    phase = torch.angle(spec) # (1, F, T)

    frame_energy = mag.pow(2).sum(dim=1).squeeze(0)  # (T,)
    n_frames = frame_energy.shape[-1]
    if n_frames <= 0:
        return wav.squeeze(0).cpu()

    k = max(1, int(n_frames * noise_frac))
    idxs = torch.argsort(frame_energy)[:k]
    noise_floor = mag[:, :, idxs].median(dim=-1).values  # (1, F)
    noise_floor_exp = noise_floor.unsqueeze(-1).repeat(1, 1, mag.shape[-1])

    alpha = subtract_strength
    mag_sub = mag - alpha * noise_floor_exp
    mag_sub = torch.clamp(mag_sub, min=0.0)

    real = mag_sub * torch.cos(phase)
    imag = mag_sub * torch.sin(phase)
    complex_sub = torch.complex(real, imag)

    recon = torch.istft(complex_sub, n_fft=n_fft, hop_length=hop, win_length=cfg.win_length, window=win, length=wav.shape[-1])
    return recon.squeeze(0).cpu()


# ---------------- core chunked denoiser (improved) ----------------

def denoise_chunked_final(input_path: str, output_path: str, cfg: InferConfig,
                          chunk_seconds=3.0, overlap=0.5,
                          use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0,
                          pack_png=True, png_width=2048):
    """
    Runs the chunked denoiser using the SEGAN generator.
    Returns tuple: (out_wav_path, packed_png_path_or_None, recon_wav_path_or_None)
    """
    device = cfg.device
    print("Device:", device)

    # Check SEGAN availability
    if UNetGenerator is None or STFTMagTransform is None or Config is None:
        raise RuntimeError(f"SEGAN components not available. Original import error: {_import_error}")

    # load model + stft
    print("Loading checkpoint:", cfg.ckpt_path)
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    G = UNetGenerator(in_ch=1, out_ch=1).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    stft = STFTMagTransform(cfg.n_fft, cfg.hop_length, cfg.win_length).to(device)
    window = stft.window.to(device)

    # load audio
    wav = load_mono_resampled(input_path, cfg.sample_rate)  # (T,)
    T = wav.shape[0]
    sr = cfg.sample_rate
    print(f"Input: {T} samples ({T/sr:.2f} s) SR={sr}")

    chunk_samples = max(1, int(chunk_seconds * sr))
    hop = max(1, int(chunk_samples * (1.0 - overlap)))
    print(f"Chunk {chunk_samples} samples, hop {hop} samples")

    out_len = T + chunk_samples
    out_buffer = torch.zeros(out_len, dtype=torch.float32)
    weight_buffer = torch.zeros(out_len, dtype=torch.float32)

    synth_win = torch.hann_window(chunk_samples, periodic=True, dtype=torch.float32)

    idx = 0
    while idx < T:
        start = idx
        end = min(idx + chunk_samples, T)
        chunk = wav[start:end].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
        orig_len = chunk.shape[-1]
        if orig_len < chunk_samples:
            chunk = mirror_pad_last_chunk(chunk, chunk_samples).to(device)

        with torch.no_grad():
            spec = stft(chunk)             # (1,1,F_spec,Frames)
            fake = G(spec)                 # (1,1,F_fake,Frames)
            mag = torch.expm1(fake.clamp_min(0.0)).squeeze(1)  # (1,F_fake,Frames)

        chunk_1d = chunk.view(1, -1)
        complex_noisy = torch.stft(chunk_1d, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                   win_length=cfg.win_length, window=window, return_complex=True)
        phase = torch.angle(complex_noisy)  # (1,F_phase,Frames_phase)

        n_frames_mag = mag.shape[-1]
        n_frames_phase = phase.shape[-1]
        min_frames = min(n_frames_mag, n_frames_phase)
        mag = mag[..., :min_frames]
        phase = phase[..., :min_frames]

        expected_F = cfg.n_fft // 2 + 1
        mag = pad_or_crop_freq(mag, expected_F)

        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        complex_spec = torch.complex(real, imag).squeeze(0)  # (F, frames)

        wav_rec = torch.istft(complex_spec.unsqueeze(0).to(device),
                              n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                              win_length=cfg.win_length, window=window,
                              length=chunk_samples).squeeze(0).cpu()

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

    if use_spectral_gate:
        print("Applying final spectral gating...")
        denoised = spectral_subtract_and_reconstruct(denoised.unsqueeze(0), stft, cfg,
                                                     noise_frac=noise_frac, subtract_strength=subtract_strength,
                                                     device=cfg.device)

    denoised = torch.clamp(denoised, -0.999, 0.999)

    # save denoised wav
    robust_save(output_path, denoised, sr)

    packed_png = None
    recon_wav = None
    if pack_png:
        packed_png = os.path.splitext(output_path)[0] + "_packed.png"
        save_audio_as_png_lossless(denoised, packed_png, width=png_width)
        print("Packed denoised audio into PNG:", packed_png)
        # optional: reconstruct to verify
        recon_wav = os.path.splitext(output_path)[0] + "_reconstructed_from_png.wav"
        restored = load_audio_from_png_lossless(packed_png, original_length=denoised.shape[-1])
        write_wav_from_tensor(restored, recon_wav, sr)
        print("Reconstructed WAV from PNG:", recon_wav)

    return output_path, packed_png, recon_wav


# === Lossless audio <-> PNG packing (bit-perfect) ===

def audio_tensor_to_int16_array(wav_tensor: torch.Tensor):
    if isinstance(wav_tensor, torch.Tensor):
        x = wav_tensor.detach().cpu().numpy()
    else:
        x = np.asarray(wav_tensor)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    x = np.clip(x, -1.0, 1.0)
    int16 = (x * 32767.0).astype(np.int16)
    return int16


def int16_array_to_audio_tensor(int16_arr: np.ndarray):
    arr = np.asarray(int16_arr, dtype=np.int16)
    float32 = (arr.astype(np.float32) / 32767.0)
    return torch.from_numpy(float32)


def save_audio_as_png_lossless(wav_tensor: torch.Tensor, png_path: str, width: int = 2048):
    samples = audio_tensor_to_int16_array(wav_tensor)
    N = samples.shape[0]
    height = math.ceil(N / width)
    total = width * height
    pad = total - N
    padded = np.pad(samples, (0, pad), mode='constant', constant_values=0).astype(np.int16)

    arr = padded.reshape((height, width))
    uint16_view = arr.view(np.uint16)

    im = Image.fromarray(uint16_view, mode='I;16')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    im.save(png_path, format='PNG')
    print(f"Saved lossless audio PNG: {png_path} (samples={N}, width={width}, height={height})")
    return png_path


def load_audio_from_png_lossless(png_path: str, original_length: int = None):
    im = Image.open(png_path)
    arr_uint16 = np.array(im, dtype=np.uint16)
    int16_arr = arr_uint16.view(np.int16).reshape(-1)
    if original_length is not None:
        int16_arr = int16_arr[:original_length]
    float_tensor = int16_array_to_audio_tensor(int16_arr)
    return float_tensor  # 1D torch tensor


def write_wav_from_tensor(tensor: torch.Tensor, out_wav_path: str, sr: int):
    x = tensor.detach().cpu().numpy()
    int16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    with wave.open(out_wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    print(f"WAV written (lossless restore): {out_wav_path} (samples={int16.size}, sr={sr})")
    return out_wav_path


# ----------------- CLI for quick local test -----------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Denoise WAV and pack into PNG (pipeline module)')
    parser.add_argument('--input', '-i', required=True, help='Input WAV file path')
    parser.add_argument('--output', '-o', required=False, help='Output denoised WAV path (default: input_den.wav)')
    parser.add_argument('--checkpoint', '-c', required=False, help='Checkpoint path')
    parser.add_argument('--png-width', type=int, default=2048)
    parser.add_argument('--no-pack', dest='pack', action='store_false')
    parser.set_defaults(pack=True)

    args = parser.parse_args()

    inp = args.input
    out = args.output or os.path.splitext(inp)[0] + '_denoised.wav'
    ckpt = args.checkpoint or DEFAULT_CHECKPOINT
    cfg = InferConfig(ckpt_path=ckpt)

    print('Running pipeline...')
    try:
        out_wav, packed_png, recon = denoise_chunked_final(inp, out, cfg, chunk_seconds=50.0, overlap=0.5,
                                                           use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0,
                                                           pack_png=args.pack, png_width=args.png_width)
        print('Done.\n', out_wav, packed_png, recon)
    except Exception as e:
        print('Pipeline error:', e)
        raise
