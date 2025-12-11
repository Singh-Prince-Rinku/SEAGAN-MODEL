# app.py
import os
import io
import uvicorn
import torch
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

# --- Import your denoiser functions (adjust import if SEGAN.py is in subfolder) ---
# from SEGAN import Config, STFTMagTransform, UNetGenerator
# from your_denoiser_module import denoise_chunked_final, save_audio_as_png_lossless, load_audio_from_png_lossless, write_wav_from_tensor
# For clarity, this file assumes denoise_chunked_final and packing functions are available in the `pipeline` module.
from pipeline import InferConfig, denoise_chunked_final, save_audio_as_png_lossless, load_audio_from_png_lossless, write_wav_from_tensor

# --- Config from env ---
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/seagan_final.pt")
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL")  # optional: download at startup
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
PNG_WIDTH = int(os.environ.get("PNG_WIDTH", "2048"))

# Create directories
os.makedirs("/app/data", exist_ok=True)
os.makedirs("/app/checkpoints", exist_ok=True)
os.makedirs("/tmp", exist_ok=True)

app = FastAPI(title="SEGAN Denoise + PNG packer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download checkpoint if provided via URL and not present
def ensure_checkpoint():
    if os.path.isfile(CHECKPOINT):
        print("Checkpoint exists:", CHECKPOINT)
        return CHECKPOINT
    if CHECKPOINT_URL:
        import requests
        print("Downloading checkpoint from URL...")
        r = requests.get(CHECKPOINT_URL, stream=True, timeout=60)
        if r.status_code != 200:
            raise RuntimeError("Failed to download checkpoint; status=" + str(r.status_code))
        outp = CHECKPOINT
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Downloaded checkpoint to", outp)
        return outp
    raise FileNotFoundError("No checkpoint found; set CHECKPOINT_PATH or CHECKPOINT_URL environment variable.")

# Initialize model config object (pipeline expects an InferConfig from your SEGAN code)
icfg = InferConfig()  # make sure this respects ckpt path in env inside your class
icfg.ckpt_path = CHECKPOINT

@app.on_event("startup")
def startup_event():
    # ensure checkpoint present
    try:
        cp = ensure_checkpoint()
    except Exception as e:
        print("Warning: checkpoint not found at startup:", e)
    print("Startup complete.")

@app.post("/denoise-and-pack")
async def denoise_and_pack(file: UploadFile = File(...)):
    """
    Accepts a WAV file upload. Returns a packed PNG containing lossless int16 PCM of denoised audio.
    Form-data key: 'file'
    """
    # Accept only audio/wav or octet-stream
    if file.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        # still accept many clients — but warn
        print("Warning: uploaded content_type:", file.content_type)

    # Save upload to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in.flush()
        tmp_in_path = tmp_in.name

    # Prepare output paths
    base = os.path.splitext(os.path.basename(tmp_in_path))[0]
    out_wav_path = f"/app/data/{base}_denoised.wav"
    out_png_path = f"/app/data/{base}_packed.png"
    # Run denoiser & packer (this function should save WAV and pack PNG; returns paths)
    try:
        print("Running denoiser for:", tmp_in_path)
        # Denoser might be heavy — run on CPU if no GPU
        out = denoise_chunked_final(tmp_in_path, out_wav_path, icfg,
                                    chunk_seconds=50.0, overlap=0.5,
                                    use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0)
        # out may be (wav_path, png_path, recon_wav) depending on your pipeline
    except Exception as e:
        print("Denoiser error:", e)
        raise HTTPException(status_code=500, detail="Denoiser failed: " + str(e))

    # If your denoiser already wrote packed PNG, send that; else pack
    if os.path.exists(out_png_path):
        png_to_send = out_png_path
    else:
        # load denoised tensor (you may adapt this to how denoiser returns data)
        # The pipeline.save_audio_as_png_lossless takes a tensor; if you only have file, use torchaudio.load
        import torchaudio
        wav, sr = torchaudio.load(out_wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav1d = wav.squeeze(0)
        save_audio_as_png_lossless(wav1d, out_png_path, width=PNG_WIDTH)
        png_to_send = out_png_path

    return FileResponse(png_to_send, media_type="image/png", filename=os.path.basename(png_to_send))

@app.post("/restore-from-png")
async def restore_from_png(file: UploadFile = File(...)):
    """
    Accept a packed PNG upload and return restored WAV (mono int16) using SAMPLE_RATE env var.
    """
    if file.content_type not in ("image/png", "application/octet-stream"):
        print("Warning: uploaded content_type:", file.content_type)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        tmp_png.write(await file.read())
        tmp_png.flush()
        tmp_png_path = tmp_png.name

    try:
        restored_tensor = load_audio_from_png_lossless(tmp_png_path, original_length=None)
        out_wav = f"/app/data/restored_{os.path.basename(tmp_png_path)}.wav"
        write_wav_from_tensor(restored_tensor, out_wav, SAMPLE_RATE)
    except Exception as e:
        print("Restore error:", e)
        raise HTTPException(status_code=500, detail="Restore failed: " + str(e))

    return FileResponse(out_wav, media_type="audio/wav", filename=os.path.basename(out_wav))

# Optional simple healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# Run when invoked directly (development)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
