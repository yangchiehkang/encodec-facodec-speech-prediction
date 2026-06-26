import argparse
import hashlib
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codec.encodec_wrapper import EnCodecWrapper


def read_filelist(filelist_path):
    filelist_path = Path(filelist_path)

    if not filelist_path.exists():
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")

    wav_paths = []

    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                wav_paths.append(line)

    return wav_paths


def make_token_filename(wav_path, index):
    wav_path = str(wav_path)
    stem = Path(wav_path).stem

    h = hashlib.md5(wav_path.encode("utf-8")).hexdigest()[:10]

    return f"{index:08d}_{stem}_{h}.pt"


def extract_one(wrapper, wav_path, out_path):
    result = wrapper.encode_file(wav_path)

    tokens = result["codes"].squeeze(0).cpu().long()

    if tokens.dim() != 2:
        raise RuntimeError(f"Unexpected token shape: {tokens.shape}")

    num_codebooks = tokens.shape[0]
    num_frames = tokens.shape[1]

    item = {
        "tokens": tokens,
        "sample_rate": wrapper.sample_rate,
        "wav_path": str(wav_path),
        "num_codebooks": num_codebooks,
        "num_frames": num_frames,
    }

    torch.save(item, out_path)

    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    filelist_path = Path(args.filelist)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = read_filelist(filelist_path)

    if args.limit is not None:
        wav_paths = wav_paths[:args.limit]

    print(f"Filelist      : {filelist_path}")
    print(f"Output dir    : {out_dir}")
    print(f"Num wav files : {len(wav_paths)}")
    print(f"Bandwidth     : {args.bandwidth} kbps")

    wrapper = EnCodecWrapper(bandwidth=args.bandwidth)

    print(f"Device        : {wrapper.device}")
    print(f"Sample rate   : {wrapper.sample_rate}")
    print(f"Channels      : {wrapper.channels}")

    success = 0
    skipped = 0
    failed = 0

    for idx, wav_path in enumerate(tqdm(wav_paths, desc="Extracting tokens")):
        wav_path = Path(wav_path)
        token_name = make_token_filename(wav_path, idx)
        out_path = out_dir / token_name

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            extract_one(wrapper, wav_path, out_path)
            success += 1
        except Exception as e:
            failed += 1
            print(f"\nFailed: {wav_path}")
            print(f"Reason: {e}")

    print("\nDone.")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
