"""
Transcribe Arabic .wav files using the locally cached openai/whisper-large-v3
model (loaded via HuggingFace transformers — no internet required).

Results are printed to stdout.

Usage:
    # Transcribe first 5 files in wav_files/
    python3 transcribe.py

    # Transcribe a specific number of files
    python3 transcribe.py --count 10

    # Transcribe every file
    python3 transcribe.py --count 0

    # Point at a different wav directory
    python3 transcribe.py --wav-dir /path/to/wavs --count 3
"""

import os
import sys

# ── Self-reexec with clean env vars set BEFORE any C++ runtime loads ─────────
# TF's CUDA/protobuf messages are emitted by shared libraries at import time,
# so os.environ changes in Python are too late. Re-launching the process with
# the correct env vars already in place prevents TF from initialising at all.
_SENTINEL = "__TF_SILENCED__"
if _SENTINEL not in os.environ and "streamlit" not in sys.modules:
    env = {
        **os.environ,
        _SENTINEL:                                 "1",
        "USE_TF":                                  "0",   # don't load TF backend
        "USE_TORCH":                               "1",
        "TF_CPP_MIN_LOG_LEVEL":                    "3",
        "TF_ENABLE_ONEDNN_OPTS":                   "0",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION":  "python",
        "GRPC_VERBOSITY":                          "ERROR",
        "ABSL_MIN_LOG_LEVEL":                      "3",
    }
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Whisper language token -> human-readable name
# (subset of Whisper's supported languages)
LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",  "en": "English", "fr": "French",  "de": "German",
    "es": "Spanish", "it": "Italian", "ja": "Japanese","ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "pt": "Portuguese", "nl": "Dutch",
    "tr": "Turkish", "pl": "Polish",  "sv": "Swedish",  "da": "Danish",
    "fi": "Finnish", "nb": "Norwegian","hi": "Hindi",  "ur": "Urdu",
    "fa": "Persian", "he": "Hebrew",  "id": "Indonesian","ms": "Malay",
}


# ── Model configuration ───────────────────────────────────────────────────────

HF_CACHE = Path.home() / ".cache/huggingface/hub"
MODEL_SNAPSHOT = (
    HF_CACHE
    / "models--openai--whisper-large-v3"
    / "snapshots"
    / "06f233fe06e710322aca913c1bc4249a0d71fce1"
)

DEFAULT_WAV_DIR = Path(__file__).parent.parent / "wav_files"
DEFAULT_COUNT   = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_pipeline(model_path: Path):
    """Load whisper-large-v3 from a local snapshot, or fallback to HF Hub.

    Returns
    -------
    asr          : HuggingFace ASR pipeline
    model        : raw WhisperForConditionalGeneration (for language detection)
    processor    : WhisperProcessor
    device       : torch device string
    torch_dtype  : dtype used by the model
    """
    device      = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Fallback logic for environments without the pre-cached snapshot
    model_id = str(model_path)
    local_only = True
    if not model_path.exists():
        print(f"Local model snapshot not found at {model_path}. Falling back to HF Hub.")
        model_id = "openai/whisper-large-v3"
        local_only = False

    print(f"Loading model: {model_id}")
    print(f"Device: {device}  |  dtype: {torch_dtype}\n")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype, # Keep torch_dtype here
        use_safetensors=True,
        local_files_only=local_only,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_only,
    )

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,       # <-- CHANGED from torch_dtype to dtype
        device=device,
    )
    return asr, model, processor, device, torch_dtype

def detect_language_for_file(wav_path: Path, model, processor, device, torch_dtype) -> tuple[str, str]:
    """Detect the spoken language in *wav_path* using Whisper's language-id head.

    Returns (lang_code, lang_name) e.g. ('ar', 'Arabic').
    Only the first 30 s of audio is used (sufficient for detection).
    """
    TARGET_SR = 16_000

    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:                       # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    # Trim to first 30 s to keep detection fast
    max_samples = TARGET_SR * 30
    audio_np = waveform[0, :max_samples].numpy()

    input_features = (
        processor(audio_np, sampling_rate=TARGET_SR, return_tensors="pt")
        .input_features
        .to(device)
        .to(torch_dtype)
    )

    # detect_language returns a (batch,) LongTensor of language token IDs
    with torch.no_grad():
        lang_token_ids = model.detect_language(input_features)   # shape: (1,)

    lang_tokens = processor.tokenizer.convert_ids_to_tokens(lang_token_ids.tolist())
    # lang_tokens[0] looks like "<|ar|>"
    lang_code = lang_tokens[0].strip("<|>")   # e.g. "ar"
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
    return lang_code, lang_name


def transcribe_files(asr, model, processor, device, torch_dtype, wav_paths: list[Path]) -> None:
    """Auto-detect language, transcribe each file, and print results to stdout."""
    total = len(wav_paths)
    for i, wav in enumerate(wav_paths, start=1):
        print(f"{'─'*60}")
        print(f"[{i}/{total}]  {wav.name}")
        print(f"{'─'*60}")

        # ── Language detection ───────────────────────────────────────────
        lang_code, lang_name = detect_language_for_file(
            wav, model, processor, device, torch_dtype
        )
        print(f"  Detected language: {lang_name} ({lang_code})\n")

        # ── Transcription ────────────────────────────────────────────────
        t0 = time.time()
        result = asr(
            str(wav),
            return_timestamps=True,
            generate_kwargs={
                "language": lang_code,
                "task": "transcribe",
            },
        )
        elapsed = time.time() - t0

        transcript = result.get("text", "").strip()
        print(transcript)
        print(f"\n  ⏱  {elapsed:.1f}s\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Arabic .wav files with whisper-large-v3 (local HF model)."
    )
    parser.add_argument(
        "--wav-dir",
        default=str(DEFAULT_WAV_DIR),
        help=f"Directory containing .wav files (default: {DEFAULT_WAV_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of files to transcribe. 0 = all files (default: 5)",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_SNAPSHOT),
        help="Path to the local whisper model snapshot directory.",
    )
    args = parser.parse_args()

    wav_dir    = Path(args.wav_dir)
    model_path = Path(args.model)

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not wav_dir.exists():
        sys.exit(f"wav directory not found: {wav_dir}\n"
                 "Run download_wav_files.py first.")

    if not model_path.exists():
        sys.exit(f"Model snapshot not found: {model_path}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        sys.exit(f"No .wav files found in {wav_dir}")

    if args.count > 0:
        wav_files = wav_files[: args.count]

    print(f"Transcribing {len(wav_files)} file(s) from {wav_dir}\n")

    # ── Load model & transcribe ──────────────────────────────────────────────
    asr, model, processor, device, torch_dtype = build_pipeline(model_path)
    transcribe_files(asr, model, processor, device, torch_dtype, wav_files)

    print(f"{'═'*60}")
    print("All done.")


if __name__ == "__main__":
    main()
