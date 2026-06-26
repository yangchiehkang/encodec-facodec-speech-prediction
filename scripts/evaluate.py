import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm


def load_audio_mono(path):
    wav, sr = sf.read(str(path), always_2d=False)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    return wav, sr


def resample_audio(wav, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wav.astype(np.float32)

    wav = librosa.resample(
        wav,
        orig_sr=orig_sr,
        target_sr=target_sr,
    )

    return wav.astype(np.float32)


def trim_or_pad_to_same_length(pred, ref):
    target_len = min(len(pred), len(ref))

    pred = pred[:target_len]
    ref = ref[:target_len]

    return pred, ref


def safe_stoi(ref, pred, sr):
    try:
        return float(stoi(ref, pred, sr, extended=False))
    except Exception:
        return float("nan")


def safe_pesq(ref, pred, sr):
    try:
        return float(pesq(sr, ref, pred, "wb"))
    except Exception:
        return float("nan")


def compute_dnsmos_placeholder(wav, sr):
    """
    DNSMOS requires an external DNSMOS ONNX model.
    This placeholder keeps the result table compatible.
    """
    return {
        "dnsmos_sig": float("nan"),
        "dnsmos_bak": float("nan"),
        "dnsmos_ovrl": float("nan"),
    }


def find_reference_file(pred_file, ref_dir):
    pred_name = pred_file.name

    ref_name = pred_name.replace(
        "_pred_future_2_3s.wav",
        "_reference_future_2_3s.wav",
    )

    ref_path = ref_dir / ref_name

    if ref_path.exists():
        return ref_path

    prefix = pred_name.split("_pred_future_2_3s.wav")[0]
    candidates = list(ref_dir.glob(f"{prefix}*reference*.wav"))

    if len(candidates) > 0:
        return candidates[0]

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--eval_sr", type=int, default=16000)

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    pred_files = sorted(pred_dir.glob("*.wav"))

    if len(pred_files) == 0:
        raise RuntimeError(f"No wav files found in pred_dir: {pred_dir}")

    rows = []

    for pred_path in tqdm(pred_files, desc="Evaluating", dynamic_ncols=True):
        ref_path = find_reference_file(pred_path, ref_dir)

        if ref_path is None:
            rows.append(
                {
                    "file": pred_path.name,
                    "pred_path": str(pred_path),
                    "ref_path": "",
                    "stoi": float("nan"),
                    "pesq": float("nan"),
                    "dnsmos_sig": float("nan"),
                    "dnsmos_bak": float("nan"),
                    "dnsmos_ovrl": float("nan"),
                    "status": "missing_reference",
                }
            )
            continue

        pred_wav, pred_sr = load_audio_mono(pred_path)
        ref_wav, ref_sr = load_audio_mono(ref_path)

        pred_16k = resample_audio(pred_wav, pred_sr, args.eval_sr)
        ref_16k = resample_audio(ref_wav, ref_sr, args.eval_sr)

        pred_16k, ref_16k = trim_or_pad_to_same_length(pred_16k, ref_16k)

        stoi_score = safe_stoi(ref_16k, pred_16k, args.eval_sr)
        pesq_score = safe_pesq(ref_16k, pred_16k, args.eval_sr)

        dnsmos_scores = compute_dnsmos_placeholder(pred_16k, args.eval_sr)

        rows.append(
            {
                "file": pred_path.name,
                "pred_path": str(pred_path),
                "ref_path": str(ref_path),
                "stoi": stoi_score,
                "pesq": pesq_score,
                "dnsmos_sig": dnsmos_scores["dnsmos_sig"],
                "dnsmos_bak": dnsmos_scores["dnsmos_bak"],
                "dnsmos_ovrl": dnsmos_scores["dnsmos_ovrl"],
                "status": "ok",
            }
        )

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    valid_df = df[df["status"] == "ok"]

    summary = {
        "pred_dir": str(pred_dir),
        "ref_dir": str(ref_dir),
        "eval_sr": args.eval_sr,
        "num_total": int(len(df)),
        "num_valid": int(len(valid_df)),
        "stoi_mean": float(valid_df["stoi"].mean()),
        "pesq_mean": float(valid_df["pesq"].mean()),
        "dnsmos_sig_mean": float(valid_df["dnsmos_sig"].mean()),
        "dnsmos_bak_mean": float(valid_df["dnsmos_bak"].mean()),
        "dnsmos_ovrl_mean": float(valid_df["dnsmos_ovrl"].mean()),
        "note": (
            "STOI and PESQ are computed after resampling prediction and reference "
            "to 16 kHz. DNSMOS columns are reserved as NaN unless an external "
            "DNSMOS model is integrated."
        ),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Evaluation finished.")
    print(f"Total files : {summary['num_total']}")
    print(f"Valid files : {summary['num_valid']}")
    print(f"STOI mean   : {summary['stoi_mean']}")
    print(f"PESQ mean   : {summary['pesq_mean']}")
    print(f"CSV saved   : {out_csv}")
    print(f"JSON saved  : {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
