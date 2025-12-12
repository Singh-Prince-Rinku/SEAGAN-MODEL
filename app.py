# app.py
import os
import tempfile
import traceback
import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

# Import pipeline that contains denoiser and pack/unpack functions
# Ensure pipeline.py exists in repository root
from pipeline import InferConfig, denoise_chunked_final, save_audio_as_png_lossless, load_audio_from_png_lossless, write_wav_from_tensor

# ---------------- Config from env ----------------
S3_BUCKET = os.environ.get("S3_BUCKET")            # e.g. my-segan-bucket
MODEL_KEY = os.environ.get("MODEL_KEY")            # e.g. models/seagan_model.pt
LOCAL_CKPT = os.environ.get("LOCAL_CKPT", "/app/checkpoints/seagan_model.pt")
API_KEY = os.environ.get("API_KEY")                # required header value
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
PNG_WIDTH = int(os.environ.get("PNG_WIDTH", "2048"))

# Safety checks
if S3_BUCKET is None or MODEL_KEY is None:
    print("WARNING: S3_BUCKET or MODEL_KEY env var not set. If you don't set them, the app won't be able to download the model at startup.")

app = FastAPI(title="SEGAN Denoise + PNG packer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def download_model_from_s3(bucket_name, key, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                      config=BotoConfig(signature_version='s3v4'))
    print(f"Downloading model from s3://{bucket_name}/{key} ...")
    s3.download_file(bucket_name, key, dest_path)
    print("Downloaded model ->", dest_path)
    return dest_path

@app.on_event("startup")
def startup_event():
    try:
        # Attempt to download model if not present
        if not os.path.exists(LOCAL_CKPT):
            if S3_BUCKET and MODEL_KEY:
                download_model_from_s3(S3_BUCKET, MODEL_KEY, LOCAL_CKPT)
            else:
                print("No S3 model info provided and checkpoint not found at LOCAL_CKPT.")
        else:
            print("Checkpoint already present at", LOCAL_CKPT)
        # create InferConfig pointing to local checkpoint
        global icfg
        icfg = InferConfig(ckpt_path=LOCAL_CKPT)
        print("InferConfig prepared. Device:", icfg.device)
    except Exception as e:
        print("Startup error:", e)
        traceback.print_exc()

def check_api_key(x_api_key: str = Header(None)):
    if API_KEY:
        if x_api_key is None or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/denoise-and-pack")
async def denoise_and_pack(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    # Save incoming file to a temp file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        data = await file.read()
        tmp_in.write(data)
        tmp_in.flush()
        tmp_in_path = tmp_in.name

    base = os.path.splitext(os.path.basename(tmp_in_path))[0]
    out_wav_path = f"/app/data/{base}_denoised.wav"
    out_png_path = f"/app/data/{base}_packed.png"

    try:
        # Run denoiser (this will save out_wav_path and packed png)
        print("Running denoiser for:", tmp_in_path)
        denoise_chunked_final(tmp_in_path, out_wav_path, icfg,
                              chunk_seconds=50.0, overlap=0.5,
                              use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0,
                              pack_png=True, png_width=PNG_WIDTH)
    except Exception as e:
        print("Denoiser failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Denoiser failed: " + str(e))

    if not os.path.exists(out_png_path):
        # if pipeline didn't write PNG, pack here
        import torchaudio
        wav, sr = torchaudio.load(out_wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav1d = wav.squeeze(0)
        save_audio_as_png_lossless(wav1d, out_png_path, width=PNG_WIDTH)

    return FileResponse(out_png_path, media_type="image/png", filename=os.path.basename(out_png_path))

@app.post("/restore-from-png")
async def restore_from_png(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        tmp_png.write(await file.read())
        tmp_png.flush()
        tmp_png_path = tmp_png.name

    try:
        restored_tensor = load_audio_from_png_lossless(tmp_png_path, original_length=None)
        out_wav = f"/app/data/restored_{os.path.basename(tmp_png_path)}.wav"
        write_wav_from_tensor(restored_tensor, out_wav, SAMPLE_RATE)
    except Exception as e:
        print("Restore failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Restore failed: " + str(e))

    return FileResponse(out_wav, media_type="audio/wav", filename=os.path.basename(out_wav))

@app.get("/health")
def health():
    return {"status": "ok"}
