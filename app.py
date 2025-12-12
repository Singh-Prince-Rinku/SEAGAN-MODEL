# app.py  (place at repo root)
import os
import gradio as gr
from pipeline.pipeline import InferConfig, denoise_chunked_final, load_audio_from_png_lossless, write_wav_from_tensor

# ensure SSR disabled early (best-effort)
# Note: gradio uses the ssr_mode parameter in launch; we still set env var to be safe
os.environ.setdefault("GRADIO_SSR_MODE", "false")

# Config: checkpoint path in repo (checkpoint/seagan_final.pt)
CFG = InferConfig(ckpt_path="checkpoint/seagan_final.pt")
PNG_WIDTH = 2048

def denoise_and_pack_gr(input_file):
    if input_file is None:
        return None
    src = input_file
    base = os.path.splitext(os.path.basename(src))[0]
    out_wav = f"/tmp/{base}_denoised.wav"
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

# queue to avoid blocking the server for long-running jobs
demo.queue()

# Launch with explicit ssr_mode=False and prevent_thread_lock to reduce teardown races
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0",
                server_port=port,
                share=False,
                ssr_mode=False,
                prevent_thread_lock=True)
