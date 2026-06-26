import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import EnCodecTokenDataset
from models.transformer_lm import TransformerLM, TransformerLMConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(csv_path, row, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def move_batch_to_device(batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    return input_ids, labels


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    grad_clip,
):
    model.train()

    total_loss = 0.0
    total_items = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        dynamic_ncols=True,
    )

    for batch in pbar:
        input_ids, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

        avg_loss = total_loss / max(total_items, 1)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            ppl=f"{ppl:.2f}",
        )

    avg_loss = total_loss / max(total_items, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()

    total_loss = 0.0
    total_items = 0

    pbar = tqdm(
        dataloader,
        desc=f"Valid epoch {epoch}",
        dynamic_ncols=True,
    )

    for batch in pbar:
        input_ids, labels = move_batch_to_device(batch, device)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

        avg_loss = total_loss / max(total_items, 1)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            ppl=f"{ppl:.2f}",
        )

    avg_loss = total_loss / max(total_items, 1)
    return avg_loss


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    valid_loss,
    best_valid_loss,
    config,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "best_valid_loss": best_valid_loss,
        "config": config.__dict__,
        "args": vars(args),
    }

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_token_dir", type=str, required=True)
    parser.add_argument("--valid_token_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--codebook_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    args_path = output_dir / "args.json"
    log_csv_path = log_dir / "train_log.csv"

    save_json(vars(args), args_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Training EnCodec Transformer LM")
    print("=" * 80)
    print(f"Device          : {device}")
    print(f"Train token dir : {args.train_token_dir}")
    print(f"Valid token dir : {args.valid_token_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Vocab size      : {args.vocab_size}")
    print(f"Seq len         : {args.seq_len}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Epochs          : {args.epochs}")
    print(f"d_model         : {args.d_model}")
    print(f"n_heads         : {args.n_heads}")
    print(f"n_layers        : {args.n_layers}")
    print(f"ffn_dim         : {args.ffn_dim}")
    print(f"LR              : {args.lr}")
    print(f"Seed            : {args.seed}")
    print("=" * 80)

    if args.ffn_dim != 4 * args.d_model:
        print(
            "Note: current TransformerLM uses fixed FFN size = 4 * d_model. "
            f"Received ffn_dim={args.ffn_dim}, but model will use {4 * args.d_model}."
        )

    train_dataset = EnCodecTokenDataset(
        token_dir=args.train_token_dir,
        block_size=args.seq_len,
        codebook_idx=args.codebook_idx,
        random_crop=True,
    )

    valid_dataset = EnCodecTokenDataset(
        token_dir=args.valid_token_dir,
        block_size=args.seq_len,
        codebook_idx=args.codebook_idx,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"Train samples   : {len(train_dataset)}")
    print(f"Valid samples   : {len(valid_dataset)}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Valid batches   : {len(valid_loader)}")

    config = TransformerLMConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.d_model,
        dropout=args.dropout,
    )

    model = TransformerLM(config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_params = count_parameters(model)

    print(f"Model parameters: {num_params:,}")
    print("=" * 80)

    best_valid_loss = float("inf")

    fieldnames = [
        "epoch",
        "train_loss",
        "valid_loss",
        "train_ppl",
        "valid_ppl",
        "best_valid_loss",
        "epoch_time_sec",
        "checkpoint",
        "is_best",
    ]

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            epoch=epoch,
        )

        epoch_time = time.time() - start_time

        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
        valid_ppl = math.exp(valid_loss) if valid_loss < 20 else float("inf")

        is_best = valid_loss < best_valid_loss

        if is_best:
            best_valid_loss = valid_loss

        epoch_ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"

        save_checkpoint(
            path=epoch_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            best_valid_loss=best_valid_loss,
            config=config,
            args=args,
        )

        if is_best:
            best_ckpt_path = output_dir / "best.pt"

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                best_valid_loss=best_valid_loss,
                config=config,
                args=args,
            )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "valid_loss": f"{valid_loss:.6f}",
            "train_ppl": f"{train_ppl:.6f}",
            "valid_ppl": f"{valid_ppl:.6f}",
            "best_valid_loss": f"{best_valid_loss:.6f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
            "checkpoint": str(epoch_ckpt_path),
            "is_best": int(is_best),
        }

        append_csv_row(
            csv_path=log_csv_path,
            row=row,
            fieldnames=fieldnames,
        )

        print("-" * 80)
        print(f"Epoch           : {epoch}/{args.epochs}")
        print(f"Train loss      : {train_loss:.6f}")
        print(f"Valid loss      : {valid_loss:.6f}")
        print(f"Train ppl       : {train_ppl:.4f}")
        print(f"Valid ppl       : {valid_ppl:.4f}")
        print(f"Best valid loss : {best_valid_loss:.6f}")
        print(f"Epoch time      : {epoch_time:.2f} sec")
        print(f"Saved checkpoint: {epoch_ckpt_path}")

        if is_best:
            print(f"Saved best      : {output_dir / 'best.pt'}")

        print("-" * 80)

    final_path = output_dir / "last.pt"

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        train_loss=train_loss,
        valid_loss=valid_loss,
        best_valid_loss=best_valid_loss,
        config=config,
        args=args,
    )

    print("=" * 80)
    print("Training finished.")
    print(f"Best valid loss : {best_valid_loss:.6f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Best checkpoint : {output_dir / 'best.pt'}")
    print(f"Log file        : {log_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
