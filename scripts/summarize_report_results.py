from pathlib import Path
import pandas as pd
import numpy as np

root = Path(r"E:\Working\deep-learning\speech_prediction_project")
outputs = root / "outputs"

result_files = {
    "Copy baseline": outputs / "copy_baseline_results.csv",
    "EnCodec oracle": outputs / "encodec_oracle_results.csv",
    "EnCodec debug": outputs / "encodec_debug_results.csv",
    "EnCodec small": outputs / "encodec_small_results.csv",
    "EnCodec top-k20": outputs / "encodec_topk20_results.csv",
    "EnCodec small eval": outputs / "eval_encodec_small.csv",
}

oracle_small16_files = {
    "EnCodec oracle small16": outputs / "encodec_oracle_small16_results.csv",
    "FACodec oracle small16": outputs / "facodec_oracle_small16_results.csv",
}

metric_cols = [
    "stoi",
    "pesq",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
]

rows = []

print("=" * 80)
print("Full prediction / evaluation result summary")
print("=" * 80)

for name, path in result_files.items():
    print("\n", name, path)
    if not path.exists():
        print("Missing")
        continue

    df = pd.read_csv(path)

    row = {
        "system": name,
        "file": str(path.relative_to(root)),
        "num_rows": len(df),
    }

    if "status" in df.columns:
        status = df["status"].astype(str).str.lower()
        row["num_ok"] = int(status.isin(["ok", "OK".lower()]).sum())
        row["num_error"] = int((~status.isin(["ok", "OK".lower()])).sum())
    else:
        row["num_ok"] = None
        row["num_error"] = None

    for col in metric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std()) if vals.notna().any() else np.nan
            row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"{col}_valid_count"] = int(vals.notna().sum())
        else:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"] = np.nan
            row[f"{col}_median"] = np.nan
            row[f"{col}_valid_count"] = 0

    rows.append(row)

full_summary = pd.DataFrame(rows)
print(full_summary)

full_out = outputs / "report_full_eval_summary.csv"
full_summary.to_csv(full_out, index=False)
print("\nSaved:", full_out)

print("\n" + "=" * 80)
print("Oracle reconstruction small16 summary")
print("=" * 80)

small_rows = []

small_metric_cols = [
    "mse",
    "mae",
    "rmse",
    "snr_db",
    "si_sdr_db",
    "duration_sec",
    "elapsed_sec",
]

for name, path in oracle_small16_files.items():
    print("\n", name, path)
    if not path.exists():
        print("Missing")
        continue

    df = pd.read_csv(path)

    row = {
        "system": name,
        "file": str(path.relative_to(root)),
        "num_rows": len(df),
    }

    if "status" in df.columns:
        status = df["status"].astype(str).str.lower()
        row["num_ok"] = int((status == "ok").sum())
        row["num_error"] = int((status != "ok").sum())

    for col in small_metric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std()) if vals.notna().any() else np.nan
            row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan

    small_rows.append(row)

small_summary = pd.DataFrame(small_rows)
print(small_summary)

small_out = outputs / "report_oracle_small16_summary.csv"
small_summary.to_csv(small_out, index=False)
print("\nSaved:", small_out)

print("\nDone.")
