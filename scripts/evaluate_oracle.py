import argparse
import csv
import json
import math
import os
import warnings
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch
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

            if not line or line.startswith("#"):
                continue

            candidates = []

            if "|" in line:
                candidates.append(line.split("|")[0].strip())

            if "," in line:
                candidates.append(line.split(",")[0].strip())

            candidates.append(line.split()[0].strip())

            selected = None
            for c in candidates:
                c = os.path.normpath(c)
                if os.path.exists(c):
                    selected = c
                    break

            if selected is None:
                selected = os.path.normpath(candidates[-1])

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

    return audio.astype(np.float32)


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    frac = Fraction(target_sr, orig_sr).limit_denominator()
    y = resample_poly(audio, frac.numerator, frac.denominator)

    return y.astype(np.float32)


def safe_crop(audio, start_sample, end_sample):
    n = len(audio)

    start_sample = max(0, int(start_sample))
    end_sample = max(start_sample, int(end_sample))

    target_len = end_sample - start_sample

    if start_sample >= n:
        return np.zeros(target_len, dtype=np.float32)

    segment = audio[start_sample:min(end_sample, n)]

    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)), mode="constant")

    return segment.astype(np.float32)


def match_length(a, b):
    n = min(len(a), len(b))
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

    if np.max(np.abs(pred_audio)) == 0 or np.max(np.abs(ref_audio)) == 0:
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result["stoi"] = float(stoi(ref_audio, pred_audio, sr, extended=False))
    except Exception:
        result["stoi"] = float("nan")

    try:
        mode = "wb" if sr == 16000 else "nb"
        result["pesq"] = float(pesq(sr, ref_audio, pred_audio, mode))
    except Exception:
        result["pesq"] = float("nan")

    return result


def mean_ignore_nan(values):
    valid = []

    for v in values:
        if v is None:
            continue
        try:
            if not math.isnan(float(v)):
                valid.append(float(v))
        except Exception:
            continue

    if len(valid) == 0:
        return float("nan")

    return float(np.mean(valid))


def load_encodec_model(device, bandwidth):
    try:
        from encodec import EncodecModel
    except ImportError as e:
        raise ImportError(
            "The package 'encodec' is not installed. "
            "Please install it with: pip install encodec"
        ) from e

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(float(bandwidth))
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def encodec_reconstruct(model, audio_24k, device):
    """
    audio_24k: numpy mono audio, shape [T]
    return: numpy mono reconstructed audio, shape [T_recon]
    """

    wav = torch.from_numpy(audio_24k).float().to(device)

    # EnCodec expects [B, C, T]
    wav = wav.unsqueeze(0).unsqueeze(0)

    encoded_frames = model.encode(wav)
    reconstructed = model.decode(encoded_frames)

    reconstructed = reconstructed.squeeze(0).squeeze(0)
    reconstructed = reconstructed.detach().cpu().numpy().astype(np.float32)

    return reconstructed


def json_dump_with_nan(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EnCodec oracle reconstruction upper-bound."
    )

    parser.add_argument("--codec", type=str, default="encodec", choices=["encodec"])
    parser.add_argument("--test_filelist", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)

    parser.add_argument("--context_sec", type=float, default=2.0)
    parser.add_argument("--pred_sec", type=float, default=1.0)

    parser.add_argument("--eval_sample_rate", "--eval_sr", dest="eval_sr", type=int, default=16000)
    parser.add_argument("--codec_sample_rate", type=int, default=24000)
    parser.add_argument("--bandwidth", type=float, default=6.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute_dnsmos", action="store_true")

    args = parser.parse_args()

    if args.out_json is None:
        base, _ = os.path.splitext(args.out_csv)
        args.out_json = base + "_summary.json"

    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.out_csv) or ".")

    pred_dir = os.path.join(args.output_dir, "pred")
    ref_dir = os.path.join(args.output_dir, "ref")
    full_dir = os.path.join(args.output_dir, "full")
    context_dir = os.path.join(args.output_dir, "context")

    ensure_dir(pred_dir)
    ensure_dir(ref_dir)
    ensure_dir(full_dir)
    ensure_dir(context_dir)

    print("=" * 80)
    print("Loading EnCodec oracle model...")
    print(f"Device          : {args.device}")
    print(f"Codec SR        : {args.codec_sample_rate}")
    print(f"Bandwidth       : {args.bandwidth}")
    print("=" * 80)

    device = torch.device(args.device)
    model = load_encodec_model(device, args.bandwidth)

    filelist = read_filelist(args.test_filelist)

    rows = []

    total_sec = args.context_sec + args.pred_sec

    for idx, wav_path in enumerate(tqdm(filelist, desc="Oracle reconstruction")):
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

            # Convert original audio to EnCodec 24 kHz.
            audio_24k = resample_audio(audio, sr, args.codec_sample_rate)

            codec_sr = args.codec_sample_rate

            total_len = int(round(total_sec * codec_sr))
            context_len = int(round(args.context_sec * codec_sr))
            pred_len = int(round(args.pred_sec * codec_sr))

            # Use true 0s-3s audio for oracle reconstruction.
            original_0_3 = safe_crop(audio_24k, 0, total_len)

            # EnCodec encode -> decode.
            recon_0_3 = encodec_reconstruct(model, original_0_3, device)

            # Match reconstructed length to expected 0s-3s.
            if len(recon_0_3) < total_len:
                recon_0_3 = np.pad(recon_0_3, (0, total_len - len(recon_0_3)), mode="constant")
            else:
                recon_0_3 = recon_0_3[:total_len]

            # Original reference future: 2s-3s.
            ref_future = safe_crop(original_0_3, context_len, context_len + pred_len)

            # Oracle prediction: reconstructed 2s-3s.
            pred_future = safe_crop(recon_0_3, context_len, context_len + pred_len)

            # Context is original 0s-2s.
            context_audio = safe_crop(original_0_3, 0, context_len)

            context_path = os.path.join(context_dir, f"{prefix}_context_0_2s.wav")
            pred_path = os.path.join(pred_dir, f"{prefix}_oracle_pred_future_2_3s.wav")
            ref_path = os.path.join(ref_dir, f"{prefix}_reference_future_2_3s.wav")
            full_path = os.path.join(full_dir, f"{prefix}_oracle_reconstructed_0_3s.wav")

            sf.write(context_path, context_audio, codec_sr)
            sf.write(pred_path, pred_future, codec_sr)
            sf.write(ref_path, ref_future, codec_sr)
            sf.write(full_path, recon_0_3, codec_sr)

            pred_eval = resample_audio(pred_future, codec_sr, args.eval_sr)
            ref_eval = resample_audio(ref_future, codec_sr, args.eval_sr)

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

    valid_rows = [r for r in rows if r["status"] == "OK"]

    summary = {
        "codec": args.codec,
        "test_filelist": args.test_filelist,
        "output_dir": args.output_dir,
        "pred_dir": pred_dir,
        "ref_dir": ref_dir,
        "eval_sr": args.eval_sr,
        "codec_sample_rate": args.codec_sample_rate,
        "bandwidth": args.bandwidth,
        "context_sec": args.context_sec,
        "pred_sec": args.pred_sec,
        "num_total": len(rows),
        "num_valid": len(valid_rows),
        "stoi_mean": mean_ignore_nan([r["stoi"] for r in valid_rows]),
        "pesq_mean": mean_ignore_nan([r["pesq"] for r in valid_rows]),
        "dnsmos_sig_mean": float("nan"),
        "dnsmos_bak_mean": float("nan"),
        "dnsmos_ovrl_mean": float("nan"),
        "note": (
            "Oracle reconstruction encodes and decodes the ground-truth 0-3s audio "
            "with EnCodec, then compares the reconstructed 2-3s segment against "
            "the original 2-3s reference. STOI and PESQ are computed after "
            "resampling prediction and reference to eval_sr. DNSMOS columns are "
            "reserved as NaN unless an external DNSMOS model is integrated."
        ),
    }

    json_dump_with_nan(summary, args.out_json)

    print("=" * 80)
    print("Oracle reconstruction evaluation finished.")
    print(f"Total files : {summary['num_total']}")
    print(f"Valid files : {summary['num_valid']}")
    print(f"STOI mean   : {summary['stoi_mean']}")
    print(f"PESQ mean   : {summary['pesq_mean']}")
    print(f"CSV saved   : {args.out_csv}")
    print(f"JSON saved  : {args.out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
