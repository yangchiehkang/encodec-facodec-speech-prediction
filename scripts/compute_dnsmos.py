import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm


# ============================================================
# DNSMOS import
# ============================================================

try:
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
except Exception:
    from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(r"E:\Working\deep-learning\speech_prediction_project")

GENERATED_ROOT = PROJECT_ROOT / "generated"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# System wav folders
# Edit these names if your folder names are different.
# Each path should point to the folder containing predicted wav files.
# ============================================================

SYSTEM_WAV_DIRS = {
    "copy_baseline": GENERATED_ROOT / "copy_baseline" / "pred",
    "encodec_oracle": GENERATED_ROOT / "encodec_oracle" / "pred",
    "encodec_small": GENERATED_ROOT / "encodec_small" / "pred",
    "encodec_topk20": GENERATED_ROOT / "encodec_topk20" / "pred",
}


# ============================================================
# DNSMOS settings
# ============================================================

TARGET_SR = 16000

# DNSMOS is normally designed for longer clips.
# Since this project evaluates 1-second future speech,
# we repeat short clips to satisfy a stable inference window.
MIN_DNSMOS_SECONDS = 9.01


def collect_wavs_from_dir(wav_dir: Path):
    """
    Recursively collect wav files from a directory.
    """
    wav_dir = Path(wav_dir)

    if not wav_dir.exists():
        return []

    wav_paths = sorted(wav_dir.rglob("*.wav"))
    return wav_paths


def load_audio_mono_16k(path: Path):
    """
    Load waveform, convert to mono, and resample to 16 kHz.
    """
    wav, sr = sf.read(str(path), dtype="float32")

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if wav.size == 0:
        raise ValueError("Empty audio file")

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    return wav


def repeat_to_min_duration(wav: np.ndarray, sr: int, min_seconds: float):
    """
    Repeat short clips to satisfy DNSMOS input duration.
    """
    min_len = int(sr * min_seconds)

    if len(wav) >= min_len:
        return wav

    repeat_times = int(np.ceil(min_len / len(wav)))
    wav = np.tile(wav, repeat_times)[:min_len]

    return wav


def parse_dnsmos_output(score):
    """
    Parse TorchMetrics DNSMOS output.

    TorchMetrics DNSMOS usually returns four scores:
        [p808_mos, mos_sig, mos_bak, mos_ovr]

    The main DNSMOS value commonly reported is mos_ovr.
    """
    if isinstance(score, dict):
        return {
            "p808_mos": float(score.get("p808_mos", np.nan)),
            "mos_sig": float(score.get("mos_sig", np.nan)),
            "mos_bak": float(score.get("mos_bak", np.nan)),
            "mos_ovr": float(score.get("mos_ovr", np.nan)),
        }

    if isinstance(score, (list, tuple)):
        score = score[0]

    if torch.is_tensor(score):
        arr = score.detach().cpu().numpy()
    else:
        arr = np.asarray(score)

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return {
            "p808_mos": np.nan,
            "mos_sig": np.nan,
            "mos_bak": np.nan,
            "mos_ovr": float(arr),
        }

    arr = arr.flatten()

    if len(arr) >= 4:
        return {
            "p808_mos": float(arr[0]),
            "mos_sig": float(arr[1]),
            "mos_bak": float(arr[2]),
            "mos_ovr": float(arr[3]),
        }

    return {
        "p808_mos": np.nan,
        "mos_sig": np.nan,
        "mos_bak": np.nan,
        "mos_ovr": float(arr[-1]),
    }


def init_dnsmos_metric(device: str):
    """
    Initialize TorchMetrics DNSMOS.

    Some TorchMetrics versions use different constructor arguments,
    so this function tries the common versions.
    """
    try:
        metric = DeepNoiseSuppressionMeanOpinionScore(
            fs=TARGET_SR,
            personalized=False
        )
    except TypeError:
        try:
            metric = DeepNoiseSuppressionMeanOpinionScore(
                sample_rate=TARGET_SR,
                personalized=False
            )
        except TypeError:
            metric = DeepNoiseSuppressionMeanOpinionScore(TARGET_SR)

    metric = metric.to(device)
    metric.eval()

    return metric


def compute_system_dnsmos(system_name: str, wav_dir: Path, metric, device: str):
    """
    Compute DNSMOS for one system.
    """
    wav_paths = collect_wavs_from_dir(wav_dir)

    print("\n" + "=" * 80)
    print(f"[SYSTEM] {system_name}")
    print(f"[DIR]    {wav_dir}")
    print(f"[FILES]  {len(wav_paths)} wav files found")
    print("=" * 80)

    if len(wav_paths) == 0:
        print(f"[WARN] No wav files found for system: {system_name}. Skipping.")
        return None, None

    rows = []

    for wav_path in tqdm(wav_paths, desc=f"DNSMOS {system_name}"):
        wav_path = Path(wav_path)

        try:
            wav = load_audio_mono_16k(wav_path)
            original_duration = len(wav) / TARGET_SR

            wav_for_metric = repeat_to_min_duration(
                wav,
                sr=TARGET_SR,
                min_seconds=MIN_DNSMOS_SECONDS,
            )

            wav_tensor = torch.from_numpy(wav_for_metric).float().unsqueeze(0).to(device)

            with torch.no_grad():
                score = metric(wav_tensor)

            parsed = parse_dnsmos_output(score)

            rows.append({
                "system": system_name,
                "wav_path": str(wav_path),
                "duration_sec": original_duration,
                "p808_mos": parsed["p808_mos"],
                "mos_sig": parsed["mos_sig"],
                "mos_bak": parsed["mos_bak"],
                "mos_ovr": parsed["mos_ovr"],
                "error": "",
            })

        except Exception as e:
            rows.append({
                "system": system_name,
                "wav_path": str(wav_path),
                "duration_sec": np.nan,
                "p808_mos": np.nan,
                "mos_sig": np.nan,
                "mos_bak": np.nan,
                "mos_ovr": np.nan,
                "error": repr(e),
            })

    df = pd.DataFrame(rows)

    valid = df[df["error"] == ""].copy()

    summary = {
        "system": system_name,
        "wav_dir": str(wav_dir),
        "num_files_total": len(df),
        "num_files_valid": len(valid),
        "p808_mos_mean": valid["p808_mos"].mean(),
        "p808_mos_std": valid["p808_mos"].std(),
        "mos_sig_mean": valid["mos_sig"].mean(),
        "mos_sig_std": valid["mos_sig"].std(),
        "mos_bak_mean": valid["mos_bak"].mean(),
        "mos_bak_std": valid["mos_bak"].std(),
        "mos_ovr_mean": valid["mos_ovr"].mean(),
        "mos_ovr_std": valid["mos_ovr"].std(),
    }

    summary_df = pd.DataFrame([summary])

    per_file_out = OUTPUT_DIR / f"{system_name}_dnsmos_per_file.csv"
    summary_out = OUTPUT_DIR / f"{system_name}_dnsmos_summary.csv"

    df.to_csv(per_file_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] Per-file DNSMOS: {per_file_out}")
    print(f"[SAVED] Summary DNSMOS:  {summary_out}")

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))

    if len(valid) == 0:
        print(f"\n[WARN] All files failed for system: {system_name}")
        error_counts = df["error"].value_counts().head(5)
        print(error_counts)

    return df, summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        help=(
            "System to evaluate. Use one of: all, "
            + ", ".join(SYSTEM_WAV_DIRS.keys())
        ),
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="Optional custom wav directory. If set, only this directory will be evaluated.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional system name when using --wav_dir.",
    )

    args = parser.parse_args()

    print("[INFO] Project root:", PROJECT_ROOT)
    print("[INFO] Generated root:", GENERATED_ROOT)
    print("[INFO] Output dir:", OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device)

    print("[INFO] Initializing DNSMOS metric...")
    metric = init_dnsmos_metric(device)
    print("[INFO] DNSMOS metric initialized.")

    all_summaries = []

    # ============================================================
    # Custom directory mode
    # ============================================================

    if args.wav_dir is not None:
        wav_dir = Path(args.wav_dir)
        system_name = args.name if args.name is not None else "custom_system"

        _, summary_df = compute_system_dnsmos(
            system_name=system_name,
            wav_dir=wav_dir,
            metric=metric,
            device=device,
        )

        if summary_df is not None:
            all_summaries.append(summary_df)

    # ============================================================
    # Predefined systems mode
    # ============================================================

    else:
        if args.system == "all":
            selected = SYSTEM_WAV_DIRS
        else:
            if args.system not in SYSTEM_WAV_DIRS:
                raise ValueError(
                    f"Unknown system: {args.system}. "
                    f"Available systems: {list(SYSTEM_WAV_DIRS.keys())}"
                )
            selected = {args.system: SYSTEM_WAV_DIRS[args.system]}

        for system_name, wav_dir in selected.items():
            _, summary_df = compute_system_dnsmos(
                system_name=system_name,
                wav_dir=wav_dir,
                metric=metric,
                device=device,
            )

            if summary_df is not None:
                all_summaries.append(summary_df)

    # ============================================================
    # Save combined summary
    # ============================================================

    if len(all_summaries) > 0:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_out = OUTPUT_DIR / "dnsmos_all_systems_summary.csv"
        combined.to_csv(combined_out, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print("[DONE] DNSMOS computation finished.")
        print(f"[SAVED] Combined summary: {combined_out}")
        print("=" * 80)

        print("\n[COMBINED SUMMARY]")
        cols = [
            "system",
            "num_files_valid",
            "p808_mos_mean",
            "mos_sig_mean",
            "mos_bak_mean",
            "mos_ovr_mean",
        ]
        print(combined[cols].to_string(index=False))

    else:
        print("\n[WARN] No DNSMOS summaries were generated.")


if __name__ == "__main__":
    main()
