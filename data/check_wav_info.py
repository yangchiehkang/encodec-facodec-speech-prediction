# data/check_wav_info.py

import argparse
from pathlib import Path
import torchaudio


def read_filelist(path, max_items=None):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_items is not None:
        lines = lines[:max_items]

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--max_items", type=int, default=10)
    args = parser.parse_args()

    wav_paths = read_filelist(args.filelist, args.max_items)

    print(f"Checking {len(wav_paths)} wav files from {args.filelist}")
    print("-" * 80)

    for wav_path in wav_paths:
        wav_path = Path(wav_path)

        if not wav_path.exists():
            print(f"[Missing] {wav_path}")
            continue

        wav, sr = torchaudio.load(str(wav_path))
        duration = wav.shape[-1] / sr

        print(f"Path: {wav_path}")
        print(f"  Shape: {tuple(wav.shape)}")
        print(f"  Sample rate: {sr}")
        print(f"  Duration: {duration:.2f} sec")
        print()


if __name__ == "__main__":
    main()
