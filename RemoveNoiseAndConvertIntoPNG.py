#!/usr/bin/env python3
"""
ChunkedDenoise-SEGAN-to-PNG.py

- Runs your chunked SEGAN denoiser (mirror-pad last chunk, hann overlap-add, final spectral gating)
- Saves denoised audio (WAV) and packs denoised PCM into a lossless 16-bit PNG (mono).
- Optionally reconstructs WAV from PNG to demonstrate bit-perfect roundtrip.

Edit INPUT_AUDIO / OUTPUT_AUDIO / CHECKPOINT below.
Requires: torch, torchaudio, numpy, Pillow (PIL)
"""

import os
import math
import wave
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image

# Import SEGAN components - keep your module with these classes available
from SEGAN import Config, STFTMagTransform, UNetGenerator

# ---------------- USER CONFIG ----------------
INPUT_AUDIO  = r"E:\Test Audio Data\AudioSong\romantic-song-tera-roothna-by-ashir-hindi-top-trending-viral-song-231771.wav"
OUTPUT_AUDIO = r"E:\Test Audio Data\Denoised Results\romantic-song-tera-roothna-by-ashir-hindi-top-trending-viral-song-231771.wav"
CHECKPOINT   = r"E:\Minor-Project-For-Amity-Patna\Model SEGAN\checkpoints_seagan\seagan_final.pt"

CHUNK_SECONDS      = 50.0         # chunk length (seconds)
CHUNK_OVERLAP      = 0.5          # fraction overlap (0.5 = 50%)
USE_SPECTRAL_GATE  = True         # final spectral gating to suppress residual noise
NOISE_FRAC         = 0.10        # fraction of lowest-energy frames to estimate noise floor
SUBTRACT_STRENGTH  = 1.0         # how strongly to subtract noise floor (1.0 = full)

# PNG packing options
PNG_WIDTH = 2048  # pixels per row for the packed PNG (adjust for memory/shape)
# ---------------------------------------------

class InferConfig(Config):
    ckpt_path = CHECKPOINT
    device = "cuda" if torch.cuda.is_available() else "cpu"

icfg = InferConfig()

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
    # remove leading unit batch dims
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
    # mag: (1, F_mag, T)
    F_mag = mag.shape[1]
    if F_mag == target_F:
        return mag
    if F_mag < target_F:
        pad = target_F - F_mag
        return F.pad(mag, (0, 0, 0, pad))
    else:
        return mag[:, :target_F, :]

def mirror_pad_last_chunk(chunk: torch.Tensor, target_len: int):
    # chunk: (1,1,L)
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

def spectral_subtract_and_reconstruct(waveform: torch.Tensor, stft_mod: STFTMagTransform, cfg: InferConfig,
                                      noise_frac=0.1, subtract_strength=1.0, device='cpu'):
    """
    waveform: (T,) or (1, T) tensor (CPU)
    Estimate noise floor from lowest-energy frames (fraction noise_frac),
    subtract (soft) from magnitude spectrogram, reconstruct with original phase via torch.istft.
    Returns denoised waveform (1, T_recon) on CPU.
    """
    if waveform.dim() == 1:
        wav = waveform.unsqueeze(0)  # (1, T)
    else:
        wav = waveform
    wav = wav.to(device)

    n_fft = cfg.n_fft
    hop = cfg.hop_length
    win = stft_mod.window.to(device)

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
                          use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0):
    device = cfg.device
    print("Device:", device)

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

    robust_save(output_path, denoised, sr)

    # After saving WAV, pack denoised audio into lossless PNG
    png_path = os.path.splitext(output_path)[0] + "_packed.png"
    save_audio_as_png_lossless(denoised, png_path, width=PNG_WIDTH)
    print("Packed denoised audio into PNG:", png_path)

    # Optionally: reconstruct WAV from PNG to verify roundtrip correctness
    recon_wav = os.path.splitext(output_path)[0] + "_reconstructed_from_png.wav"
    restored = load_audio_from_png_lossless(png_path, original_length=denoised.shape[-1])
    write_wav_from_tensor(restored, recon_wav, sr)
    print("Reconstructed WAV from PNG:", recon_wav)

    return output_path, png_path, recon_wav

# === Lossless audio <-> PNG packing (bit-perfect) ===

def audio_tensor_to_int16_array(wav_tensor: torch.Tensor):
    """wav_tensor: 1D float tensor (-1..1) or (1, T). Returns numpy int16 1D array."""
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
    """Return torch.FloatTensor mono in range -1..1"""
    arr = np.asarray(int16_arr, dtype=np.int16)
    float32 = (arr.astype(np.float32) / 32767.0)
    return torch.from_numpy(float32)

def save_audio_as_png_lossless(wav_tensor: torch.Tensor, png_path: str, width: int = 2048):
    """
    Pack int16 PCM samples into a single-channel 16-bit PNG.
    width = number of pixels per row. height computed automatically.
    """
    samples = audio_tensor_to_int16_array(wav_tensor)
    N = samples.shape[0]
    height = math.ceil(N / width)
    total = width * height
    pad = total - N
    padded = np.pad(samples, (0, pad), mode='constant', constant_values=0).astype(np.int16)

    arr = padded.reshape((height, width))

    # reinterpret signed int16 bits as uint16 so PIL can save without changing bits
    uint16_view = arr.view(np.uint16)

    im = Image.fromarray(uint16_view, mode='I;16')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    im.save(png_path, format='PNG')
    print(f"Saved lossless audio PNG: {png_path} (samples={N}, width={width}, height={height})")
    return png_path

def load_audio_from_png_lossless(png_path: str, original_length: int = None):
    """
    Read 16-bit PNG saved by save_audio_as_png_lossless and return torch.FloatTensor mono (-1..1).
    If original_length provided, crop the extra padding.
    """
    im = Image.open(png_path)
    arr_uint16 = np.array(im, dtype=np.uint16)
    int16_arr = arr_uint16.view(np.int16).reshape(-1)
    if original_length is not None:
        int16_arr = int16_arr[:original_length]
    float_tensor = int16_array_to_audio_tensor(int16_arr)
    return float_tensor  # 1D torch tensor

def write_wav_from_tensor(tensor: torch.Tensor, out_wav_path: str, sr: int):
    # tensor: 1D float -1..1
    x = tensor.detach().cpu().numpy()
    int16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    with wave.open(out_wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    print(f"WAV written (lossless restore): {out_wav_path} (samples={int16.size}, sr={sr})")
    return out_wav_path

# ---------------- run ----------------

if __name__ == "__main__":
    print("Running final chunked denoiser -> PNG (lossless) ...")
    out_wav, out_png, out_recon = denoise_chunked_final(INPUT_AUDIO, OUTPUT_AUDIO, icfg,
                                                        chunk_seconds=CHUNK_SECONDS,
                                                        overlap=CHUNK_OVERLAP,
                                                        use_spectral_gate=USE_SPECTRAL_GATE,
                                                        noise_frac=NOISE_FRAC,
                                                        subtract_strength=SUBTRACT_STRENGTH)
    print("Done.")
    print("Denoised WAV:", out_wav)
    print("Packed PNG:", out_png)
    print("Reconstructed WAV from PNG (verification):", out_recon)
