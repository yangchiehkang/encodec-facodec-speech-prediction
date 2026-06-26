import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EnCodecTokenDataset(Dataset):
    def __init__(
        self,
        token_dir,
        block_size=224,
        codebook_idx=0,
        random_crop=True,
    ):
        self.token_dir = Path(token_dir)
        self.block_size = block_size
        self.codebook_idx = codebook_idx
        self.random_crop = random_crop

        if not self.token_dir.exists():
            raise FileNotFoundError(f"Token dir not found: {self.token_dir}")

        self.pt_files = sorted(self.token_dir.glob("*.pt"))

        if len(self.pt_files) == 0:
            raise RuntimeError(f"No .pt files found in: {self.token_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]

        item = torch.load(pt_path, map_location="cpu")
        tokens = item["tokens"]

        if tokens.dim() != 2:
            raise RuntimeError(f"Expected tokens shape [num_codebooks, num_frames], got {tokens.shape}")

        seq = tokens[self.codebook_idx, :].long()

        needed_len = self.block_size + 1

        if seq.numel() < needed_len:
            pad_len = needed_len - seq.numel()
            seq = torch.cat(
                [
                    seq,
                    torch.zeros(pad_len, dtype=torch.long),
                ],
                dim=0,
            )

        if self.random_crop and seq.numel() > needed_len:
            max_start = seq.numel() - needed_len
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = 0

        chunk = seq[start:start + needed_len]

        x = chunk[:-1]
        y = chunk[1:]

        return {
            "input_ids": x,
            "labels": y,
            "pt_path": str(pt_path),
            "wav_path": item.get("wav_path", ""),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=224)
    parser.add_argument("--codebook_idx", type=int, default=0)
    args = parser.parse_args()

    dataset = EnCodecTokenDataset(
        token_dir=args.token_dir,
        block_size=args.block_size,
        codebook_idx=args.codebook_idx,
        random_crop=False,
    )

    sample = dataset[0]

    print(f"Token dir     : {args.token_dir}")
    print(f"Num samples   : {len(dataset)}")
    print(f"Input shape   : {sample['input_ids'].shape}")
    print(f"Target shape  : {sample['labels'].shape}")
    print(f"Input dtype    : {sample['input_ids'].dtype}")
    print(f"Target dtype   : {sample['labels'].dtype}")
    print(f"First pt file  : {sample['pt_path']}")
    print(f"Original wav   : {sample['wav_path']}")


if __name__ == "__main__":
    main()
