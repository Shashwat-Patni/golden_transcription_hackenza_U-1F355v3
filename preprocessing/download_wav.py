"""
Download all .wav files listed in the transcription CSV into a local
`wav_files/` directory.

Usage:
    python download_wav_files.py
    python download_wav_files.py --csv "Transcription Assessment Arabic_SA Dataset(Arabic_SA).csv"
    python download_wav_files.py --output wav_files --workers 4
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


DEFAULT_CSV = "Transcription Assessment Arabic_SA Dataset(Arabic_SA).csv"
DEFAULT_OUTPUT_DIR = "../wav_files"
DEFAULT_WORKERS = 4


def load_audio_urls(csv_path: str) -> list[dict]:
    """Return a list of dicts with keys 'audio_id' and 'url'."""
    entries = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get("audio", "").strip()
            audio_id = row.get("audio_id", "").strip()
            if url.lower().endswith(".wav"):
                entries.append({"audio_id": audio_id, "url": url})
            else:
                print(f"  [SKIP] Row {audio_id!r} – not a .wav URL: {url!r}")
    return entries


def download_file(entry: dict, output_dir: Path, session: requests.Session, retries: int = 3) -> tuple[str, bool, str]:
    """Download a single file. Returns (filename, success, message)."""
    url = entry["url"]
    filename = Path(urlparse(url).path).name  # e.g. 2_hours_Segment_001.wav
    dest = output_dir / filename

    if dest.exists():
        return filename, True, "already exists, skipped"

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
                tmp.rename(dest)
            return filename, True, "downloaded"
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
            else:
                return filename, False, str(exc)

    return filename, False, "unknown error"


def main():
    parser = argparse.ArgumentParser(description="Download .wav files from the transcription CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to the transcription CSV file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Destination directory for .wav files.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel download threads.")
    args = parser.parse_args()

    # Resolve paths relative to the script's directory
    script_dir = Path(__file__).parent.parent
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else script_dir / args.csv
    output_dir = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output

    if not csv_path.exists():
        sys.exit(f"CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {output_dir}")
    print(f"CSV              : {csv_path}")

    entries = load_audio_urls(str(csv_path))
    if not entries:
        sys.exit("No .wav URLs found in the CSV.")

    print(f"Found {len(entries)} .wav URLs. Downloading with {args.workers} worker(s)...\n")

    ok_count = skip_count = fail_count = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "wav-downloader/1.0"})

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_entry = {
                executor.submit(download_file, entry, output_dir, session): entry
                for entry in entries
            }
            for i, future in enumerate(as_completed(future_to_entry), start=1):
                entry = future_to_entry[future]
                filename, success, msg = future.result()
                status = "OK  " if success else "FAIL"
                if success and "skipped" in msg:
                    skip_count += 1
                elif success:
                    ok_count += 1
                else:
                    fail_count += 1
                print(f"  [{status}] ({i:>3}/{len(entries)}) {filename}  —  {msg}")

    print(f"\nDone. Downloaded: {ok_count}  |  Skipped (exists): {skip_count}  |  Failed: {fail_count}")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
