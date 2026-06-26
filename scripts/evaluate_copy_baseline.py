import argparse
import csv
import json
import math
import os
import warnings
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from pystoi import stoi
from pesq import pesq


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_filelist(filelist_path):
    items = []

    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # 支持常见格式：
            # 1) path.wav
            # 2) path.wav|text
            # 3) path.wav,text
            # 4) path.wav text
            candidates = []

            if "|" in line:
                candidates.append(line.split("|")[0].strip())

            if "," in line:
                candidates.append(line.split(",")[0].strip())

            candidates.append(line.split()[0].strip())

            selected = None
            for c in candidates:
                if os.path.exists(c):
                    selected = c
                    break

                # 尝试相对当前工作目录
                c2 = os.path.normpath(c)
                if os.path.exists(c2):
                    selected = c2
                    break

            if selected is None:
                # 如果暂时找不到，也保留第一列，后面报 warning
                selected = candidates[-1]

            items.append(selected)

    return items


def to_mono(audio):
    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1)


def normalize_float(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0:
        audio = audio / max_abs

    return audio


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    frac = Fraction(target_sr, orig_sr).limit_denominator()
    resampled = resample_poly(audio, frac.numerator, frac.denominator)

    return resampled.astype(np.float32)


def safe_crop(audio, start_sample, end_sample):
    n = len(audio)

    start_sample = max(0, int(start_sample))
    end_sample = max(start_sample, int(end_sample))

    if start_sample >= n:
        return np.zeros(end_sample - start_sample, dtype=np.float32)

    segment = audio[start_sample:min(end_sample, n)]

    target_len = end_sample - start_sample
    if len(segment) < target_len:
        pad_len = target_len - len(segment)
        segment = np.pad(segment, (0, pad_len), mode="constant")

    return segment.astype(np.float32)


def match_length(a, b):
    n = min(len(a), len(b))
    if n <= 0:
        return a[:0], b[:0]

    return a[:n], b[:n]


def compute_metrics(pred_audio, ref_audio, sr):
    pred_audio, ref_audio = match_length(pred_audio, ref_audio)

    result = {
        "stoi": float("nan"),
        "pesq": float("nan"),
    }

    if len(pred_audio) == 0 or len(ref_audio) == 0:
        return result

    pred_audio = np.asarray(pred_audio, dtype=np.float32)
    ref_audio = np.asarray(ref_audio, dtype=np.float32)

    # 避免完全静音导致部分指标异常
    if np.max(np.abs(pred_audio)) == 0 or np.max(np.abs(ref_audio)) == 0:
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result["stoi"] = float(stoi(ref_audio, pred_audio, sr, extended=False))
    except Exception:
        result["stoi"] = float("nan")

    try:
        # pesq package: mode="wb" for 16 kHz, mode="nb" for 8 kHz
        mode = "wb" if sr == 16000 else "nb"
        result["pesq"] = float(pesq(sr, ref_audio, pred_audio, mode))
    except Exception:
        result["pesq"] = float("nan")

    return result


def json_dump_with_nan(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def mean_ignore_nan(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate copy baseline for future speech prediction."
    )

    parser.add_argument("--test_filelist", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)

    parser.add_argument("--context_sec", type=float, default=2.0)
    parser.add_argument("--pred_sec", type=float, default=1.0)

    # 兼容报告里的 --eval_sample_rate，也兼容之前 evaluate.py 的 --eval_sr
    parser.add_argument("--eval_sample_rate", "--eval_sr", dest="eval_sr", type=int, default=16000)

    # 当前只保留 DNSMOS 占位列
    parser.add_argument("--compute_dnsmos", action="store_true")

    args = parser.parse_args()

    pred_dir = os.path.join(args.output_dir, "pred")
    ref_dir = os.path.join(args.output_dir, "ref")
    full_dir = os.path.join(args.output_dir, "full")
    context_dir = os.path.join(args.output_dir, "context")

    ensure_dir(args.output_dir)
    ensure_dir(pred_dir)
    ensure_dir(ref_dir)
    ensure_dir(full_dir)
    ensure_dir(context_dir)
    ensure_dir(os.path.dirname(args.out_csv) or ".")

    if args.out_json is None:
        base, _ = os.path.splitext(args.out_csv)
        args.out_json = base + "_summary.json"

    filelist = read_filelist(args.test_filelist)

    rows = []

    for idx, wav_path in enumerate(tqdm(filelist, desc="Copy baseline")):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        prefix = f"{idx:04d}_{utt_id}"

        row = {
            "index": idx,
            "utt_id": utt_id,
            "wav_path": wav_path,
            "pred_path": "",
            "ref_path": "",
            "full_path": "",
            "context_path": "",
            "stoi": float("nan"),
            "pesq": float("nan"),
            "dnsmos_sig": float("nan"),
            "dnsmos_bak": float("nan"),
            "dnsmos_ovrl": float("nan"),
            "status": "FAILED",
            "error": "",
        }

        try:
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Audio file not found: {wav_path}")

            audio, sr = sf.read(wav_path)
            audio = to_mono(audio)
            audio = normalize_float(audio)

            context_end = int(round(args.context_sec * sr))
            pred_len = int(round(args.pred_sec * sr))

            # Copy baseline:
            # context    : 0s ~ 2s
            # prediction : 1s ~ 2s, copied as future prediction
            # reference  : 2s ~ 3s
            context_start = 0
            context_end_sample = context_end

            pred_start = int(round((args.context_sec - args.pred_sec) * sr))
            pred_end = context_end

            ref_start = context_end
            ref_end = context_end + pred_len

            context_audio = safe_crop(audio, context_start, context_end_sample)
            pred_audio = safe_crop(audio, pred_start, pred_end)
            ref_audio = safe_crop(audio, ref_start, ref_end)

            # full: 0~2s context + copied 1~2s prediction => pseudo 0~3s output
            full_audio = np.concatenate([context_audio, pred_audio], axis=0)

            context_path = os.path.join(context_dir, f"{prefix}_context_0_2s.wav")
            pred_path = os.path.join(pred_dir, f"{prefix}_copy_pred_future_2_3s.wav")
            ref_path = os.path.join(ref_dir, f"{prefix}_reference_future_2_3s.wav")
            full_path = os.path.join(full_dir, f"{prefix}_copy_combined_context_pred_0_3s.wav")

            sf.write(context_path, context_audio, sr)
            sf.write(pred_path, pred_audio, sr)
            sf.write(ref_path, ref_audio, sr)
            sf.write(full_path, full_audio, sr)

            pred_eval = resample_audio(pred_audio, sr, args.eval_sr)
            ref_eval = resample_audio(ref_audio, sr, args.eval_sr)

            metrics = compute_metrics(pred_eval, ref_eval, args.eval_sr)

            row["context_path"] = context_path
            row["pred_path"] = pred_path
            row["ref_path"] = ref_path
            row["full_path"] = full_path
            row["stoi"] = metrics["stoi"]
            row["pesq"] = metrics["pesq"]
            row["status"] = "OK"

        except Exception as e:
            row["error"] = repr(e)

        rows.append(row)

    fieldnames = [
        "index",
        "utt_id",
        "wav_path",
        "pred_path",
        "ref_path",
        "full_path",
        "context_path",
        "stoi",
        "pesq",
        "dnsmos_sig",
        "dnsmos_bak",
        "dnsmos_ovrl",
        "status",
        "error",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total = len(rows)
    valid_rows = [r for r in rows if r["status"] == "OK"]
    num_valid = len(valid_rows)

    stoi_mean = mean_ignore_nan([r["stoi"] for r in valid_rows])
    pesq_mean = mean_ignore_nan([r["pesq"] for r in valid_rows])

    summary = {
        "test_filelist": args.test_filelist,
        "output_dir": args.output_dir,
        "pred_dir": pred_dir,
        "ref_dir": ref_dir,
        "eval_sr": args.eval_sr,
        "context_sec": args.context_sec,
        "pred_sec": args.pred_sec,
        "num_total": total,
        "num_valid": num_valid,
        "stoi_mean": stoi_mean,
        "pesq_mean": pesq_mean,
        "dnsmos_sig_mean": float("nan"),
        "dnsmos_bak_mean": float("nan"),
        "dnsmos_ovrl_mean": float("nan"),
        "note": (
            "Copy baseline uses audio from context_sec - pred_sec to context_sec "
            "as prediction, and audio from context_sec to context_sec + pred_sec "
            "as reference. STOI and PESQ are computed after resampling prediction "
            "and reference to eval_sr. DNSMOS columns are reserved as NaN unless "
            "an external DNSMOS model is integrated."
        ),
    }

    json_dump_with_nan(summary, args.out_json)

    print("=" * 80)
    print("Copy baseline evaluation finished.")
    print(f"Total files : {total}")
    print(f"Valid files : {num_valid}")
    print(f"STOI mean   : {stoi_mean}")
    print(f"PESQ mean   : {pesq_mean}")
    print(f"CSV saved   : {args.out_csv}")
    print(f"JSON saved  : {args.out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
