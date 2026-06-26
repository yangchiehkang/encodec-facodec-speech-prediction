import argparse
import json
import math
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from encodec import EncodecModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.transformer_lm import TransformerLM, TransformerLMConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio_mono(path, target_sr):
    wav, sr = sf.read(path, always_2d=False)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if sr != target_sr:
        wav = librosa.resample(
            wav,
            orig_sr=sr,
            target_sr=target_sr,
        )

    return wav, target_sr


def save_wav(path, wav, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    sf.write(str(path), wav, sr)


def trim_or_pad(wav, target_len):
    wav = np.asarray(wav, dtype=np.float32)

    if len(wav) > target_len:
        return wav[:target_len]

    if len(wav) < target_len:
        pad_len = target_len - len(wav)
        return np.pad(wav, (0, pad_len), mode="constant")

    return wav


def load_filelist(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]

    return items


def load_encodec_model(device, bandwidth):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def encode_audio_to_codes(codec_model, wav_np, device):
    """
    Encode waveform into EnCodec discrete codes.

    Args:
        codec_model: EnCodec model
        wav_np: np.ndarray [num_samples]
        device: torch.device

    Returns:
        codes: LongTensor [n_codebooks, T]
    """
    wav = torch.from_numpy(wav_np).float().to(device)
    wav = wav.unsqueeze(0).unsqueeze(0)

    encoded_frames = codec_model.encode(wav)

    code_list = []

    for codes, scale in encoded_frames:
        # codes shape: [B, n_codebooks, T_frame]
        code_list.append(codes)

    codes = torch.cat(code_list, dim=-1)
    codes = codes[0].long()

    return codes


@torch.no_grad()
def decode_codes_to_audio(codec_model, codes, device):
    """
    Decode EnCodec discrete codes back into waveform.

    Args:
        codec_model: EnCodec model
        codes: LongTensor [n_codebooks, T]

    Returns:
        wav_np: np.ndarray [num_samples]
    """
    codes = codes.unsqueeze(0).to(device).long()

    encoded_frames = [(codes, None)]
    wav = codec_model.decode(encoded_frames)

    wav = wav[0, 0].detach().cpu().numpy().astype(np.float32)

    return wav


def build_transformer_from_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain `config`.")

    config_dict = checkpoint["config"]

    config = TransformerLMConfig(
        vocab_size=config_dict["vocab_size"],
        block_size=config_dict["block_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
        dropout=config_dict.get("dropout", 0.0),
    )

    model = TransformerLM(config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config, checkpoint


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    """
    Apply top-k and top-p filtering to logits.

    Args:
        logits: Tensor [vocab_size]
        top_k: int
        top_p: float

    Returns:
        filtered logits
    """
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[-1]
        logits[logits < threshold] = -float("inf")

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        if sorted_indices_to_remove.numel() > 1:
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

    return logits


@torch.no_grad()
def generate_first_codebook_tokens(
    model,
    context_tokens,
    num_future_tokens,
    block_size,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    greedy=False,
):
    """
    Autoregressively generate future tokens for the first codebook only.

    Args:
        model: Transformer LM
        context_tokens: LongTensor [T_context]
        num_future_tokens: int
        block_size: int
        temperature: float
        top_k: int
        top_p: float
        greedy: bool

    Returns:
        generated_future: LongTensor [num_future_tokens]
    """
    device = next(model.parameters()).device

    generated = context_tokens.clone().long().to(device)

    for _ in tqdm(
        range(num_future_tokens),
        desc="Generating first-codebook tokens",
        dynamic_ncols=True,
    ):
        idx_cond = generated[-block_size:].unsqueeze(0)

        outputs = model(idx_cond)
        logits = outputs["logits"]

        next_logits = logits[0, -1, :]

        if temperature <= 0:
            greedy = True
        else:
            next_logits = next_logits / temperature

        if greedy:
            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            filtered_logits = top_k_top_p_filtering(
                next_logits,
                top_k=top_k,
                top_p=top_p,
            )

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        generated = torch.cat([generated, next_token], dim=0)

    generated_future = generated[-num_future_tokens:]

    return generated_future.detach().cpu()


def fill_future_codebooks(
    context_codes,
    future_first_codebook,
    fill_strategy="repeat_last",
    repeat_context_tokens=75,
):
    """
    Fill future tokens for non-first codebooks.

    The current mainline system only predicts future tokens for the first codebook.
    EnCodec decoding, however, requires all codebooks. Therefore, future tokens
    for other codebooks are approximated with a filling strategy.

    Args:
        context_codes:
            LongTensor [n_codebooks, T_context]

        future_first_codebook:
            LongTensor [T_future]

        fill_strategy:
            repeat_last:
                For codebooks 1..N, repeat the last context token of each codebook.

            repeat_context:
                For codebooks 1..N, take the tail part of context tokens and
                cyclically repeat it to fill the future segment.

        repeat_context_tokens:
            Number of tail context tokens used by repeat_context.
            If <= 0, the whole context sequence is used.

    Returns:
        future_codes:
            LongTensor [n_codebooks, T_future]
    """
    context_codes = context_codes.detach().cpu().long()
    future_first_codebook = future_first_codebook.detach().cpu().long()

    n_codebooks, context_len = context_codes.shape
    future_len = future_first_codebook.numel()

    future_codes = torch.zeros(
        n_codebooks,
        future_len,
        dtype=torch.long,
    )

    # The first codebook is generated by Transformer LM.
    future_codes[0] = future_first_codebook

    if n_codebooks == 1:
        return future_codes

    if fill_strategy == "repeat_last":
        for q in range(1, n_codebooks):
            last_token = context_codes[q, -1]
            future_codes[q] = last_token

    elif fill_strategy == "repeat_context":
        if repeat_context_tokens is None or repeat_context_tokens <= 0:
            tail_len = context_len
        else:
            tail_len = min(repeat_context_tokens, context_len)

        for q in range(1, n_codebooks):
            tail_tokens = context_codes[q, -tail_len:]

            repeat_count = math.ceil(future_len / tail_len)
            repeated = tail_tokens.repeat(repeat_count)

            future_codes[q] = repeated[:future_len]

    else:
        raise ValueError(
            f"Unknown fill_strategy: {fill_strategy}. "
            "Supported strategies: repeat_last, repeat_context"
        )

    return future_codes


def process_one_wav(
    wav_path,
    sample_index,
    args,
    codec_model,
    lm_model,
    lm_config,
    device,
):
    wav_path = Path(wav_path)

    sr = 24000
    wav, sr = load_audio_mono(
        wav_path,
        target_sr=sr,
    )

    context_samples = int(args.context_sec * sr)
    pred_samples = int(args.pred_sec * sr)
    total_required_samples = context_samples + pred_samples

    if len(wav) < total_required_samples:
        print(f"Skip short wav: {wav_path}")
        print(f"Length samples: {len(wav)}, required: {total_required_samples}")
        return None

    # 0-2s context
    context_wav = wav[:context_samples]

    # 2-3s ground-truth future
    reference_future_wav = wav[
        context_samples:context_samples + pred_samples
    ]

    # Encode context waveform into EnCodec tokens.
    context_codes = encode_audio_to_codes(
        codec_model=codec_model,
        wav_np=context_wav,
        device=device,
    )

    frame_rate = getattr(codec_model, "frame_rate", 75)
    num_future_tokens = int(round(args.pred_sec * frame_rate))

    first_codebook_context = context_codes[0]

    # Generate future first-codebook tokens.
    future_first_codebook = generate_first_codebook_tokens(
        model=lm_model,
        context_tokens=first_codebook_context,
        num_future_tokens=num_future_tokens,
        block_size=lm_config.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    # Fill other future codebooks with approximation strategy.
    future_codes = fill_future_codebooks(
        context_codes=context_codes,
        future_first_codebook=future_first_codebook,
        fill_strategy=args.fill_strategy,
        repeat_context_tokens=args.repeat_context_tokens,
    )

    # Concatenate context tokens and predicted future tokens.
    combined_codes = torch.cat(
        [
            context_codes.detach().cpu(),
            future_codes,
        ],
        dim=-1,
    )

    # Decode 0-3s combined tokens.
    combined_pred_wav = decode_codes_to_audio(
        codec_model=codec_model,
        codes=combined_codes,
        device=device,
    )

    combined_target_len = context_samples + pred_samples
    combined_pred_wav = trim_or_pad(
        combined_pred_wav,
        combined_target_len,
    )

    # Extract predicted future waveform from 2-3s.
    predicted_future_wav = combined_pred_wav[
        context_samples:context_samples + pred_samples
    ]

    predicted_future_wav = trim_or_pad(
        predicted_future_wav,
        pred_samples,
    )

    context_wav = trim_or_pad(
        context_wav,
        context_samples,
    )

    reference_future_wav = trim_or_pad(
        reference_future_wav,
        pred_samples,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{sample_index:04d}_{wav_path.stem}"

    context_path = out_dir / f"{stem}_context_0_2s.wav"
    pred_path = out_dir / f"{stem}_pred_future_2_3s.wav"
    ref_path = out_dir / f"{stem}_reference_future_2_3s.wav"
    combined_path = out_dir / f"{stem}_combined_context_pred_0_3s.wav"
    token_path = out_dir / f"{stem}_tokens.pt"
    meta_path = out_dir / f"{stem}_meta.json"

    save_wav(context_path, context_wav, sr)
    save_wav(pred_path, predicted_future_wav, sr)
    save_wav(ref_path, reference_future_wav, sr)
    save_wav(combined_path, combined_pred_wav, sr)

    torch.save(
        {
            "wav_path": str(wav_path),
            "context_codes": context_codes.detach().cpu(),
            "future_first_codebook": future_first_codebook.detach().cpu(),
            "future_codes": future_codes.detach().cpu(),
            "combined_codes": combined_codes.detach().cpu(),
            "sample_rate": sr,
            "context_sec": args.context_sec,
            "pred_sec": args.pred_sec,
            "fill_strategy": args.fill_strategy,
            "repeat_context_tokens": args.repeat_context_tokens,
        },
        token_path,
    )

    meta = {
        "wav_path": str(wav_path),
        "checkpoint": str(args.checkpoint),
        "sample_rate": sr,
        "context_sec": args.context_sec,
        "pred_sec": args.pred_sec,
        "context_samples": context_samples,
        "pred_samples": pred_samples,
        "frame_rate": frame_rate,
        "context_token_len": int(context_codes.shape[-1]),
        "future_token_len": int(num_future_tokens),
        "num_codebooks": int(context_codes.shape[0]),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "fill_strategy": args.fill_strategy,
        "repeat_context_tokens": args.repeat_context_tokens,
        "outputs": {
            "context": str(context_path),
            "prediction": str(pred_path),
            "reference": str(ref_path),
            "combined": str(combined_path),
            "tokens": str(token_path),
            "meta": str(meta_path),
        },
        "note": (
            "Only the first codebook future tokens are predicted by Transformer LM. "
            "Other future codebooks are filled by an approximation strategy due to "
            "local GPU memory and modeling constraints."
        ),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("-" * 80)
    print(f"WAV              : {wav_path}")
    print(f"Context saved    : {context_path}")
    print(f"Prediction saved : {pred_path}")
    print(f"Reference saved  : {ref_path}")
    print(f"Combined saved   : {combined_path}")
    print(f"Tokens saved     : {token_path}")
    print(f"Meta saved       : {meta_path}")
    print("-" * 80)

    return meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav_path", type=str, default=None)
    parser.add_argument("--test_filelist", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--context_sec", type=float, default=2.0)
    parser.add_argument("--pred_sec", type=float, default=1.0)

    parser.add_argument("--bandwidth", type=float, default=6.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument(
        "--fill_strategy",
        type=str,
        default="repeat_last",
        choices=["repeat_last", "repeat_context"],
        help=(
            "Strategy for filling future tokens of non-first codebooks. "
            "repeat_last repeats the last context token. "
            "repeat_context cyclically repeats the tail context tokens."
        ),
    )

    parser.add_argument(
        "--repeat_context_tokens",
        type=int,
        default=75,
        help=(
            "Number of tail context tokens used by repeat_context. "
            "For EnCodec 24 kHz, 75 tokens are approximately 1 second. "
            "Set <= 0 to use the whole context."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.wav_path is None and args.test_filelist is None:
        raise ValueError("Please provide either --wav_path or --test_filelist.")

    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Generate future speech with EnCodec Transformer LM")
    print("=" * 80)
    print(f"Device                : {device}")
    print(f"Checkpoint            : {args.checkpoint}")
    print(f"Output dir            : {args.out_dir}")
    print(f"Context sec           : {args.context_sec}")
    print(f"Predict sec           : {args.pred_sec}")
    print(f"Bandwidth             : {args.bandwidth}")
    print(f"Temperature           : {args.temperature}")
    print(f"Top-k                 : {args.top_k}")
    print(f"Top-p                 : {args.top_p}")
    print(f"Greedy                : {args.greedy}")
    print(f"Fill strategy         : {args.fill_strategy}")
    print(f"Repeat context tokens : {args.repeat_context_tokens}")
    print("=" * 80)

    codec_model = load_encodec_model(
        device=device,
        bandwidth=args.bandwidth,
    )

    lm_model, lm_config, checkpoint = build_transformer_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print("Loaded Transformer LM:")
    print(f"Vocab size    : {lm_config.vocab_size}")
    print(f"Block size    : {lm_config.block_size}")
    print(f"Layers        : {lm_config.n_layer}")
    print(f"Heads         : {lm_config.n_head}")
    print(f"Embedding dim : {lm_config.n_embd}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "best_valid_loss" in checkpoint:
        print(f"Best valid loss : {checkpoint['best_valid_loss']}")

    print("=" * 80)

    if args.wav_path is not None:
        wav_paths = [args.wav_path]
    else:
        wav_paths = load_filelist(args.test_filelist)

    wav_paths = wav_paths[:args.num_examples]

    all_meta = []

    for i, wav_path in enumerate(wav_paths):
        meta = process_one_wav(
            wav_path=wav_path,
            sample_index=i,
            args=args,
            codec_model=codec_model,
            lm_model=lm_model,
            lm_config=lm_config,
            device=device,
        )

        if meta is not None:
            all_meta.append(meta)

    summary_path = Path(args.out_dir) / "generation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            all_meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("Generation finished.")
    print(f"Generated examples: {len(all_meta)}")
    print(f"Summary file      : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
