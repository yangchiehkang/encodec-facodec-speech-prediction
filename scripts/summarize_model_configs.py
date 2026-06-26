from pathlib import Path
import json
import pandas as pd

root = Path(r"E:\Working\deep-learning\speech_prediction_project")

print("=" * 80)
print("Model config summary")
print("=" * 80)

config_files = [
    r"checkpoints\encodec_medium\args.json",
    r"checkpoints\encodec_medium\planned_run_config.json",
    r"checkpoints\facodec_small_cb0\args.json",
    r"checkpoints\facodec_small_cb1\args.json",
    r"checkpoints\facodec_small_cb2\args.json",
    r"checkpoints\facodec_small_cb3\args.json",
    r"checkpoints\facodec_small_cb4\args.json",
    r"checkpoints\facodec_small_cb5\args.json",
]

for rel in config_files:
    p = root / rel
    print("\n" + "-" * 80)
    print(rel)
    print("-" * 80)
    print("Exists:", p.exists())

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            print(json.dumps(obj, indent=2))
        except Exception as e:
            print("Read error:", repr(e))

print("\n" + "=" * 80)
print("Training logs best validation summary")
print("=" * 80)

log_files = [
    r"checkpoints\encodec_medium\logs\train_log.csv",
    r"checkpoints\facodec_small_cb0\logs\train_log.csv",
    r"checkpoints\facodec_small_cb1\logs\train_log.csv",
    r"checkpoints\facodec_small_cb2\logs\train_log.csv",
    r"checkpoints\facodec_small_cb3\logs\train_log.csv",
    r"checkpoints\facodec_small_cb4\logs\train_log.csv",
    r"checkpoints\facodec_small_cb5\logs\train_log.csv",
]

rows = []

for rel in log_files:
    p = root / rel
    if not p.exists():
        print("[MISS]", rel)
        continue

    df = pd.read_csv(p)
    best_idx = df["valid_loss"].idxmin()
    best = df.loc[best_idx]
    final = df.iloc[-1]

    rows.append({
        "log": rel,
        "num_epochs": len(df),
        "best_epoch": int(best["epoch"]),
        "best_valid_loss": float(best["valid_loss"]),
        "best_valid_ppl": float(best["valid_ppl"]),
        "final_train_loss": float(final["train_loss"]),
        "final_valid_loss": float(final["valid_loss"]),
        "final_train_ppl": float(final["train_ppl"]),
        "final_valid_ppl": float(final["valid_ppl"]),
    })

summary = pd.DataFrame(rows)
print(summary)

out = root / "outputs" / "report_model_training_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)
print("\nSaved:", out)
