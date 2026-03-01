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

import argparse
import sys
import time
from pathlib import Path

# Suppress torch FutureWarning about pynvml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except ImportError:
    sys.exit("Required packages missing. Install with:\n  pip install transformers torch")


# ── Model configuration ───────────────────────────────────────────────────────

HF_CACHE = Path.home() / ".cache/huggingface/hub"
MODEL_SNAPSHOT = (
    HF_CACHE
    / "models--openai--whisper-large-v3"
    / "snapshots"
    / "06f233fe06e710322aca913c1bc4249a0d71fce1"
)

DEFAULT_WAV_DIR = Path(__file__).parent / "wav_files"
DEFAULT_COUNT   = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_pipeline(model_path: Path):
    """Load whisper-large-v3 from a local snapshot and return an ASR pipeline."""
    device      = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from: {model_path}")
    print(f"Device: {device}  |  dtype: {torch_dtype}\n")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return asr


def transcribe_files(asr, wav_paths: list[Path]) -> None:
    """Transcribe each file and print results to stdout."""
    total = len(wav_paths)
    for i, wav in enumerate(wav_paths, start=1):
        print(f"{'─'*60}")
        print(f"[{i}/{total}]  {wav.name}")
        print(f"{'─'*60}")

        t0 = time.time()
        result = asr(
            str(wav),
            return_timestamps=True,
            generate_kwargs={
                "language": "arabic",
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
    asr = build_pipeline(model_path)
    transcribe_files(asr, wav_files)

    print(f"{'═'*60}")
    print("All done.")


if __name__ == "__main__":
    main()
