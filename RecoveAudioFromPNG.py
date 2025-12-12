#!/usr/bin/env python3
"""
Restore_Audio_From_Image.py

This script takes the PNG image that contains lossless PCM audio (packed earlier)
and converts it back into the original WAV audio.

Your paths are already inserted below.
"""

import os
from PIL import Image
import numpy as np
import wave

# ------------------- USER PATHS ----------------------

# Path of the PNG created after denoising
PACKED_PNG_PATH = r"E:\Test Audio Data\Denoised Results\romantic-song-tera-roothna-by-ashir-hindi-top-trending-viral-song-231771_packed.png"

# Output WAV after restoring from PNG
OUTPUT_WAV_PATH = r"E:\Test Audio Data\Denoised Results\romantic-song-tera-roothna-by-ashir-hindi-top-trending-viral-song-231771_restored.wav"

# SAMPLE RATE used during denoising (same as your SEGAN config)
SAMPLE_RATE = 16000    # Change if your model uses a different SR

# Original number of samples (optional — crop padding)
# If you know the exact audio length in samples, put it here
# Otherwise set to None
ORIGINAL_SAMPLE_COUNT = None
# ------------------------------------------------------


def load_int16_from_png(png_path: str):
    """Load PNG (I;16) and return a 1-D numpy int16 array."""
    if not os.path.isfile(png_path):
        raise FileNotFoundError(f"PNG not found: {png_path}")

    im = Image.open(png_path)
    arr_uint16 = np.array(im, dtype=np.uint16)

    # reinterpret bit pattern as signed int16
    int16_arr = arr_uint16.view(np.int16).reshape(-1)
    return int16_arr


def write_wav_int16(int16_arr: np.ndarray, out_wav_path: str, sample_rate: int):
    """Write int16 PCM array to WAV file."""
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)

    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_arr.tobytes())

    print(f"\nWAV successfully restored → {out_wav_path}")
    print(f"Total Samples: {len(int16_arr)}  SampleRate: {sample_rate}\n")


def restore_audio():
    print("\nReading packed PNG...")
    int16_arr = load_int16_from_png(PACKED_PNG_PATH)
    print(f"Loaded {len(int16_arr)} samples from image.")

    # Crop padding if original sample count is known
    if ORIGINAL_SAMPLE_COUNT is not None:
        int16_arr = int16_arr[:ORIGINAL_SAMPLE_COUNT]
        print(f"Cropped to original length: {ORIGINAL_SAMPLE_COUNT} samples")

    print("Writing WAV file...")
    write_wav_int16(int16_arr, OUTPUT_WAV_PATH, SAMPLE_RATE)


if __name__ == "__main__":
    restore_audio()
