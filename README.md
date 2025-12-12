---
title: SEGAN
emoji: 🏢
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.1.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Remove BackgroundNoise and Generate Image from the Audio
---

# SEAGAN Speech Enhancement & API

A minimal speech-denoising project built around a SEGAN-style U-Net generator. It includes:

- Training script to learn on paired noisy/clean audio.
- Inference pipeline that denoises long clips in chunks and can pack output audio losslessly into PNG.
- FastAPI service to expose denoise + PNG pack/restore endpoints.
- Gradio demo for Hugging Face Spaces (app.py).

---

## Repo Contents

- `SEGAN.py` – training components: config, dataset, U-Net generator, PatchGAN discriminator, training loop.  
- `pipeline.py` – inference utilities: chunked denoiser, spectral gating cleanup, PNG pack/restore helpers.  
- `app.py` – Gradio / FastAPI app wiring the pipeline for UI/API use.  
- `checkpoint/seagan_final.pt` – example checkpoint (place your own if different) — tracked with git-lfs.  
- `requirements.txt` – Python dependencies.

---

## Prerequisites

- Python 3.9+ (tested with PyTorch CPU/GPU builds).  
- For GPU inference/training, install the matching CUDA-enabled `torch`/`torchaudio`.  
- FFmpeg is not required; `torchaudio` handles WAV I/O.

---

## Install

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# or cmd:
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
