from pathlib import Path
import json

root = Path(r"E:\Working\deep-learning\speech_prediction_project")

print("=" * 80)
print("Dataset / token directory summary")
print("=" * 80)

candidate_dirs = [
    "data",
    "tokens",
    "tokens/facodec_codebook/train",
    "tokens/facodec_codebook/valid",
    "tokens/encodec/train",
    "tokens/encodec/valid",
]

for rel in candidate_dirs:
    p = root / rel
    print("\n", rel)
    print("-" * 80)
    print("Exists:", p.exists())

    if p.exists() and p.is_dir():
        files = [x for x in p.rglob("*") if x.is_file()]
        print("Num files:", len(files))
        suffix_counts = {}
        for f in files:
            suffix_counts[f.suffix.lower()] = suffix_counts.get(f.suffix.lower(), 0) + 1
        print("Suffix counts:", suffix_counts)

candidate_lists = list(root.rglob("*.txt")) + list(root.rglob("*.json"))
print("\nPossible split/list/config files:")
for p in candidate_lists[:200]:
    rel = p.relative_to(root)
    name = p.name.lower()
    if any(k in name for k in ["train", "valid", "test", "split", "list", "config", "args"]):
        print(" -", rel)
