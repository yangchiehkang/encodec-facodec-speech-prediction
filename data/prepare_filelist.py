import argparse
import random
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def get_duration(wav_path):
    with sf.SoundFile(str(wav_path)) as f:
        duration = len(f) / f.samplerate
    return duration


def collect_wavs(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.wav"))


def filter_by_duration(wav_paths, min_duration):
    kept = []
    skipped_short = 0
    skipped_error = 0

    for wav_path in tqdm(wav_paths, desc="Checking duration"):
        try:
            duration = get_duration(wav_path)
            if duration >= min_duration:
                kept.append(wav_path)
            else:
                skipped_short += 1
        except Exception:
            skipped_error += 1

    return kept, skipped_short, skipped_error


def write_filelist(paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()).replace("\\", "/") + "\n")


def random_split(all_wavs, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(all_wavs)

    n = len(all_wavs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = all_wavs[:n_train]
    valid = all_wavs[n_train:n_train + n_valid]
    test = all_wavs[n_train + n_valid:]

    return train, valid, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/filelists")
    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    train_dir = data_root / "si_tr_s"
    valid_dir = data_root / "si_dt_05"
    test_dir = data_root / "si_et_05"

    if train_dir.exists() and valid_dir.exists() and test_dir.exists():
        print("Detected WSJ0 split folders.")
        train_wavs = collect_wavs(train_dir)
        valid_wavs = collect_wavs(valid_dir)
        test_wavs = collect_wavs(test_dir)
    else:
        print("WSJ0 split folders not found. Using random split.")
        all_wavs = collect_wavs(data_root)
        train_wavs, valid_wavs, test_wavs = random_split(
            all_wavs,
            train_ratio=0.8,
            valid_ratio=0.1,
            seed=args.seed
        )

    print(f"Original train wavs: {len(train_wavs)}")
    print(f"Original valid wavs: {len(valid_wavs)}")
    print(f"Original test wavs : {len(test_wavs)}")

    print(f"\nFiltering wavs shorter than {args.min_duration} seconds...\n")

    train_wavs, train_short, train_err = filter_by_duration(train_wavs, args.min_duration)
    valid_wavs, valid_short, valid_err = filter_by_duration(valid_wavs, args.min_duration)
    test_wavs, test_short, test_err = filter_by_duration(test_wavs, args.min_duration)

    write_filelist(train_wavs, out_dir / "train.txt")
    write_filelist(valid_wavs, out_dir / "valid.txt")
    write_filelist(test_wavs, out_dir / "test.txt")

    print("\nDone.")
    print(f"Saved train filelist to: {out_dir / 'train.txt'}")
    print(f"Saved valid filelist to: {out_dir / 'valid.txt'}")
    print(f"Saved test filelist to : {out_dir / 'test.txt'}")

    print("\nFinal counts:")
    print(f"train: {len(train_wavs)} kept, {train_short} short skipped, {train_err} error skipped")
    print(f"valid: {len(valid_wavs)} kept, {valid_short} short skipped, {valid_err} error skipped")
    print(f"test : {len(test_wavs)} kept, {test_short} short skipped, {test_err} error skipped")


if __name__ == "__main__":
    main()
