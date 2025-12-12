<<<<<<< HEAD
SEAGAN Speech Enhancement & API
===============================

A minimal speech-denoising project built around a SEGAN-style U-Net generator. It includes:
- Training script to learn on paired noisy/clean audio.
- Inference pipeline that denoises long clips in chunks and can pack output audio losslessly into PNG.
- FastAPI service to expose denoise + PNG pack/restore endpoints.

Repo Contents
-------------
- `SEGAN.py` – training components: config, dataset, U-Net generator, PatchGAN discriminator, training loop.
- `pipeline.py` – inference utilities: chunked denoiser, spectral gating cleanup, PNG pack/restore helpers.
- `app.py` – FastAPI app wiring the pipeline for HTTP use.
- `seagan_final.pt` – example checkpoint (place your own if different).
- `requirements.txt` – Python dependencies.

Prerequisites
-------------
- Python 3.9+ (tested with PyTorch CPU/GPU builds).
- For GPU inference/training, install the matching CUDA-enabled `torch`/`torchaudio`.
- FFmpeg is not required; `torchaudio` handles WAV I/O.

Install
-------
```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows PowerShell: .\.venv\Scripts\activate
pip install -r requirements.txt
```
If you need a specific CUDA wheel, install torch/torchaudio first, then run `pip install -r requirements.txt` with `--no-deps`.

Quick Inference (CLI)
---------------------
Use the chunked denoiser directly:
```bash
python pipeline.py --input path/to/noisy.wav --output path/to/denoised.wav --checkpoint seagan_final.pt
```
Notes:
- `--png-width` controls width when packing to PNG; omit `--no-pack` to also write `*_packed.png` and a reconstructed WAV check.
- The denoiser mirrors/overlaps chunks to reduce seams and optionally runs a spectral subtraction cleanup.

FastAPI Service
---------------
Environment variables:
- `CHECKPOINT_PATH` (default `/app/checkpoints/seagan_final.pt`)
- `CHECKPOINT_URL` (optional download at startup)
- `SAMPLE_RATE` (default `16000`)
- `PNG_WIDTH` (default `2048`)

Run locally:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /denoise-and-pack` – form-data key `file` with WAV. Returns packed PNG of denoised audio.
- `POST /restore-from-png` – form-data key `file` with packed PNG. Returns restored WAV.
- `GET /health` – health check.

Model Training
--------------
`SEGAN.py` trains on paired noisy/clean WAVs. Update `Config.noisy_dir`, `Config.clean_dir`, and `Config.save_dir` to your paths, then run:
```bash
python SEGAN.py
```
Checkpoints are written every 5 epochs and as `seagan_final.pt` at the end. The inference pipeline expects a `G_state` entry inside the checkpoint.

PNG Packing/Restoration Utilities
---------------------------------
`pipeline.py` exposes:
- `save_audio_as_png_lossless(tensor, png_path, width)` – stores int16 PCM in a lossless PNG.
- `load_audio_from_png_lossless(png_path, original_length)` – restores the tensor.
- `write_wav_from_tensor(tensor, out_wav_path, sr)` – writes mono WAV.

Tips
----
- Keep input WAVs mono or they will be averaged to mono.
- Large files are chunked; adjust `chunk_seconds` and `overlap` in `denoise_chunked_final`.
- Ensure the checkpoint matches the model architecture in `SEGAN.py`.
=======
---
title: SEGAN
emoji: 🏢
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Remove BackgroundNoise and Generate Image from the Audio
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> b15accf770b0a139b21a8b09501ce8fd93a23c44
