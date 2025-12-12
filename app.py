# app/app.py
"""
Gradio app that exposes two buttons:
- Denoise & pack to PNG
- Restore PNG to WAV

This app expects 'checkpoint/seagan_final.pt' to exist in the repo (git-lfs).
"""

import gradio as gr
import os
from pipeline.pipeline import InferConfig, denoise_chunked_final, save_audio_as_png_lossless, load_audio_from_png_lossless, write_wav_from_tensor

# prepare config pointing to local checkpoint
CFG = InferConfig(ckpt_path="checkpoints/seagan_final.pt")
PNG_WIDTH = 2048


def denoise_and_pack_gr(input_file):
    if input_file is None:
        return None
    src = input_file
    base = os.path.splitext(os.path.basename(src))[0]
    out_wav = f"/tmp/{base}_denoised.wav"
    out_png = f"/tmp/{base}_packed.png"
    try:
        result = denoise_chunked_final(src, out_wav, CFG,
                                       chunk_seconds=30.0, overlap=0.5,
                                       use_spectral_gate=True, noise_frac=0.1, subtract_strength=1.0,
                                       pack_png=True, png_width=PNG_WIDTH)
    except Exception as e:
        return f"ERROR: {e}"
    if isinstance(result, tuple):
        return result[1]
    return None


def restore_png_gr(png_file):
    if png_file is None:
        return None
    try:
        restored = load_audio_from_png_lossless(png_file, original_length=None)
        out_wav = f"/tmp/restored_{os.path.basename(png_file)}.wav"
        write_wav_from_tensor(restored, out_wav, CFG.sample_rate)
        return out_wav
    except Exception as e:
        return f"ERROR: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# SEGAN Denoiser — Denoise ➜ Pack PNG ➜ Restore WAV")
    with gr.Row():
        with gr.Column():
            wav_in = gr.Audio(label="Upload WAV (mono)", type="filepath")
            btn = gr.Button("Denoise & Pack to PNG")
            out_png = gr.File(label="Packed PNG (download)")
        with gr.Column():
            png_in = gr.File(label="Upload Packed PNG", type="filepath")
            btn2 = gr.Button("Restore PNG to WAV")
            out_wav = gr.Audio(label="Restored WAV", type="filepath")

    btn.click(denoise_and_pack_gr, inputs=wav_in, outputs=out_png)
    btn2.click(restore_png_gr, inputs=png_in, outputs=out_wav)

if __name__ == "__main__":
    demo.launch()
