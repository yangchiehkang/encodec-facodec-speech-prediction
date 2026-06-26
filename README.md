# Autoregressive Speech Prediction with EnCodec and FACodec Token Language Models

A lightweight research implementation of autoregressive speech prediction using discrete neural audio codec tokens.

This project predicts a future speech segment from a short speech context. It encodes waveform audio into discrete codec tokens, trains Transformer language models over those tokens, generates future tokens autoregressively, decodes the predicted tokens back into waveform audio, and evaluates the generated speech using objective speech metrics.

---

## Overview

The task is to predict the next **1 second** of speech given the previous **2 seconds** of speech.

The main experimental setting is:

- **Input context:** speech from 0s to 2s
- **Prediction target:** speech from 2s to 3s
- **Main codec:** EnCodec 24 kHz mono model
- **Auxiliary codec analysis:** FACodec
- **Main model:** decoder-only Transformer language model
- **Evaluation metrics:** STOI, PESQ, DNSMOS

The overall pipeline is:

```text
Waveform
  → Neural audio codec encoder
  → Discrete codec tokens
  → Transformer codec language model
  → Predicted future codec tokens
  → Neural audio codec decoder
  → Predicted future waveform
  → STOI / PESQ / DNSMOS evaluation
```

The main EnCodec system predicts the first EnCodec codebook and fills the remaining codebooks using a repeat-last heuristic. This design keeps the system lightweight and reproducible while highlighting the importance of multi-codebook modeling.

---

## Key Features

- Autoregressive future speech prediction
- EnCodec-based waveform tokenization
- FACodec oracle and per-codebook analysis
- Transformer language modeling over codec tokens
- Copy baseline for future speech prediction
- EnCodec oracle reconstruction upper bound
- Greedy decoding and top-k sampling comparison
- STOI, PESQ, and DNSMOS evaluation
- Lightweight reproducible training and evaluation scripts
- Analysis of multi-codebook modeling difficulty

---

## Main Results

The main full-set evaluation was conducted on **644 test utterances**.

| System | # Utterances | STOI | PESQ | DNSMOS |
|---|---:|---:|---:|---:|
| Copy baseline | 644 | 0.137 | 1.053 | 2.480 |
| EnCodec oracle | 644 | 0.910 | 2.506 | 2.071 |
| EnCodec small | 644 | 0.351 | 1.108 | 1.485 |
| EnCodec top-k20 | 644 | 0.204 | 1.086 | 1.618 |

### Interpretation

- **EnCodec small** improves over the copy baseline in reference-based metrics, especially STOI.
- **EnCodec oracle** is much stronger than the learned prediction model, showing that the main bottleneck is future token prediction rather than codec reconstruction alone.
- **Copy baseline** has the highest DNSMOS because it copies real speech from the previous context. DNSMOS is no-reference, so it measures naturalness rather than whether the generated audio matches the true future segment.
- **Top-k sampling** improves no-reference quality slightly but reduces reference-based future prediction accuracy.

---

## Repository Structure

```text
speech_prediction_project/
├─ codec/
│  ├─ encodec_wrapper.py
│  └─ extract_encodec_tokens.py
│
├─ data/
│  ├─ prepare_filelist.py
│  ├─ dataset.py
│  ├─ check_wav_info.py
│  ├─ train.txt
│  ├─ valid.txt
│  ├─ test.txt
│  ├─ train_debug.txt
│  ├─ valid_debug.txt
│  └─ test_debug.txt
│
├─ models/
│  └─ transformer_lm.py
│
├─ scripts/
│  ├─ train.py
│  ├─ generate.py
│  ├─ evaluate.py
│  ├─ evaluate_copy_baseline.py
│  ├─ evaluate_oracle.py
│  ├─ compute_dnsmos.py
│  ├─ summarize_dataset_info.py
│  ├─ summarize_model_configs.py
│  └─ summarize_report_results.py
│
├─ outputs/
│  ├─ report_full_eval_summary.csv
│  ├─ report_model_training_summary.csv
│  ├─ report_oracle_small16_summary.csv
│  └─ facodec_lm_all_codebooks_metrics.csv
│
├─ logs/
│  ├─ experiment_status_20260429.md
│  └─ generation_status_20260430.md
│
├─ report.pdf
├─ report.tex
├─ requirements.txt
└─ README.md
```

Large generated artifacts such as model checkpoints, token caches, generated audio, and raw waveform datasets are intentionally excluded from the repository.

---

## Method

### EnCodec Token Modeling

The main pipeline uses the EnCodec 24 kHz mono model at 6.0 kbps bandwidth.

EnCodec produces multiple discrete codebook streams. To keep training lightweight, the main Transformer language model predicts only the first codebook.

During generation:

1. The first 2 seconds of speech are used as context.
2. The model predicts future first-codebook tokens autoregressively.
3. The remaining codebooks are filled using a repeat-last strategy.
4. EnCodec decodes the complete token tensor back into waveform audio.

This creates a compact baseline for codec-token speech prediction.

---

### Transformer Language Model

The EnCodec small model is a decoder-only causal Transformer trained with next-token prediction.

| Parameter | Value |
|---|---:|
| Vocabulary size | 1024 |
| Sequence length | 512 |
| Batch size | 8 |
| Epochs | 20 |
| Hidden dimension | 256 |
| Transformer layers | 4 |
| Attention heads | 4 |
| FFN dimension | 1024 |
| Learning rate | 0.0003 |
| Parameters | 3.81M |
| Best validation loss | 2.4869 |
| Best validation perplexity | 12.0237 |

The model is trained using cross-entropy loss for autoregressive next-token prediction.

---

### FACodec Analysis

FACodec produces six factorized codebook streams. This project trains one Transformer language model independently for each codebook.

The FACodec per-codebook results are:

| Codebook | Best Epoch | Best Loss | Best PPL | Final Loss |
|---:|---:|---:|---:|---:|
| 0 | 19 | 3.954 | 52.13 | 3.956 |
| 1 | 20 | 2.809 | 16.59 | 2.809 |
| 2 | 16 | 4.948 | 140.88 | 4.948 |
| 3 | 14 | 5.664 | 288.25 | 5.675 |
| 4 | 14 | 5.754 | 315.60 | 5.772 |
| 5 | 13 | 5.567 | 261.53 | 5.589 |

### FACodec Findings

- Codebook 1 is the easiest to model.
- Codebook 0 is also relatively predictable.
- Codebooks 3, 4, and 5 are much harder to predict.
- Later codebooks likely contain higher-entropy residual acoustic information.
- Future systems should consider hierarchical or conditional multi-codebook generation.

---

## Codec Oracle Reconstruction

Oracle reconstruction evaluates codec reconstruction quality rather than future prediction quality.

The codec receives the ground-truth waveform, encodes it into tokens, and decodes it back into waveform.

| Codec | MSE | SNR | SI-SDR | Time |
|---|---:|---:|---:|---:|
| EnCodec | 8.44e-5 | 3.618 | 2.072 | 0.111 s/item |
| FACodec | 1.03e-4 | 2.514 | -0.197 | 0.329 s/item |

On the evaluated subset, EnCodec provides stronger reconstruction quality and faster inference than FACodec.

---

## Installation

### 1. Create Environment

```bash
conda create -n speech_pred python=3.10 -y
conda activate speech_pred
```

### 2. Install PyTorch

For CUDA 11.8:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only environments:

```bash
pip install torch torchaudio torchvision
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- torchaudio
- EnCodec
- librosa
- soundfile
- tqdm
- pandas
- matplotlib
- pystoi
- pesq
- torchmetrics
- onnxruntime

---

## Dataset Preparation

The experiments are designed for WSJ-style speech datasets or other clean speech datasets.

Place waveform files under a local dataset directory. The directory may contain nested speaker folders.

Only utterances with duration of at least 3 seconds are used, because each sample requires:

- 2 seconds of context speech
- 1 second of future reference speech

Generate train / validation / test filelists:

```bash
python data/prepare_filelist.py \
  --wav_dir /path/to/wav_dataset \
  --out_dir data \
  --min_duration 3.0 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

This creates:

```text
data/train.txt
data/valid.txt
data/test.txt
```

---

## Token Extraction

Extract EnCodec tokens for each split:

```bash
python codec/extract_encodec_tokens.py \
  --filelist data/train.txt \
  --output_dir tokens/encodec/train \
  --bandwidth 6.0
```

```bash
python codec/extract_encodec_tokens.py \
  --filelist data/valid.txt \
  --output_dir tokens/encodec/valid \
  --bandwidth 6.0
```

```bash
python codec/extract_encodec_tokens.py \
  --filelist data/test.txt \
  --output_dir tokens/encodec/test \
  --bandwidth 6.0
```

Each token file stores codec token information, sample rate, source path, number of codebooks, and number of frames.

---

## Training

Train the EnCodec small Transformer language model:

```bash
python scripts/train.py \
  --train_token_dir tokens/encodec/train \
  --valid_token_dir tokens/encodec/valid \
  --output_dir checkpoints/encodec_small \
  --vocab_size 1024 \
  --seq_len 512 \
  --batch_size 8 \
  --epochs 20 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_dim 1024 \
  --lr 0.0003 \
  --seed 42
```

The best checkpoint is saved as:

```text
checkpoints/encodec_small/best.pt
```

---

## Generation

Generate future speech using greedy decoding:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/encodec_small/best.pt \
  --test_filelist data/test.txt \
  --output_dir generated/encodec_small \
  --context_sec 2.0 \
  --pred_sec 1.0 \
  --sample_rate 24000 \
  --greedy \
  --fill_strategy repeat_last
```

Generate with top-k sampling:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/encodec_small/best.pt \
  --test_filelist data/test.txt \
  --output_dir generated/encodec_topk20 \
  --context_sec 2.0 \
  --pred_sec 1.0 \
  --sample_rate 24000 \
  --temperature 0.8 \
  --top_k 20 \
  --fill_strategy repeat_last
```

Generated outputs are stored in:

```text
generated/SYSTEM_NAME/pred/
generated/SYSTEM_NAME/ref/
generated/SYSTEM_NAME/full/
```

---

## Baselines and Oracle Systems

### Copy Baseline

The copy baseline copies the 1s–2s context segment as the predicted 2s–3s future segment.

```bash
python scripts/evaluate_copy_baseline.py \
  --test_filelist data/test.txt \
  --output_dir generated/copy_baseline \
  --out_csv outputs/copy_baseline_results.csv \
  --context_sec 2.0 \
  --pred_sec 1.0 \
  --eval_sample_rate 16000
```

### EnCodec Oracle

The EnCodec oracle encodes and decodes the ground-truth waveform to estimate codec reconstruction upper-bound quality.

```bash
python scripts/evaluate_oracle.py \
  --codec encodec \
  --test_filelist data/test.txt \
  --output_dir generated/encodec_oracle \
  --out_csv outputs/encodec_oracle_results.csv \
  --context_sec 2.0 \
  --pred_sec 1.0 \
  --eval_sample_rate 16000
```

---

## Evaluation

### STOI and PESQ

Evaluate generated future speech against the reference future segment:

```bash
python scripts/evaluate.py \
  --pred_dir generated/encodec_small/pred \
  --ref_dir generated/encodec_small/ref \
  --out_csv outputs/encodec_small_results.csv \
  --eval_sample_rate 16000
```

For top-k generation:

```bash
python scripts/evaluate.py \
  --pred_dir generated/encodec_topk20/pred \
  --ref_dir generated/encodec_topk20/ref \
  --out_csv outputs/encodec_topk20_results.csv \
  --eval_sample_rate 16000
```

### DNSMOS

Compute DNSMOS for all main systems:

```bash
python scripts/compute_dnsmos.py --system all
```

The final DNSMOS summary is saved to:

```text
outputs/dnsmos_all_systems_summary.csv
```

---

## Reproducing Main Results

A typical reproduction workflow is:

```bash
# 1. Prepare filelists
python data/prepare_filelist.py \
  --wav_dir /path/to/wav_dataset \
  --out_dir data \
  --min_duration 3.0 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42

# 2. Extract codec tokens
python codec/extract_encodec_tokens.py \
  --filelist data/train.txt \
  --output_dir tokens/encodec/train \
  --bandwidth 6.0

python codec/extract_encodec_tokens.py \
  --filelist data/valid.txt \
  --output_dir tokens/encodec/valid \
  --bandwidth 6.0

python codec/extract_encodec_tokens.py \
  --filelist data/test.txt \
  --output_dir tokens/encodec/test \
  --bandwidth 6.0

# 3. Train model
python scripts/train.py \
  --train_token_dir tokens/encodec/train \
  --valid_token_dir tokens/encodec/valid \
  --output_dir checkpoints/encodec_small \
  --vocab_size 1024 \
  --seq_len 512 \
  --batch_size 8 \
  --epochs 20 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_dim 1024 \
  --lr 0.0003 \
  --seed 42

# 4. Generate future speech
python scripts/generate.py \
  --checkpoint checkpoints/encodec_small/best.pt \
  --test_filelist data/test.txt \
  --output_dir generated/encodec_small \
  --context_sec 2.0 \
  --pred_sec 1.0 \
  --sample_rate 24000 \
  --greedy \
  --fill_strategy repeat_last

# 5. Evaluate STOI/PESQ
python scripts/evaluate.py \
  --pred_dir generated/encodec_small/pred \
  --ref_dir generated/encodec_small/ref \
  --out_csv outputs/encodec_small_results.csv \
  --eval_sample_rate 16000

# 6. Compute DNSMOS
python scripts/compute_dnsmos.py --system all
```

---

## Evaluation Notes

### STOI and PESQ

STOI and PESQ are reference-based metrics. They compare the predicted 2s–3s future segment with the true 2s–3s future segment.

Before metric computation:

- Prediction and reference waveforms are loaded.
- Both are resampled to 16 kHz.
- Both are truncated or padded to the same length.
- STOI and PESQ are computed.

### DNSMOS

DNSMOS is a no-reference metric. It evaluates perceptual speech quality without comparing against the ground-truth future segment.

This means that a copied real speech segment may receive a high DNSMOS score even if it does not match the true future speech.

---

## Limitations

The current implementation is intentionally lightweight and has several limitations:

- The main EnCodec model predicts only the first codebook.
- Remaining codebooks are filled using a repeat-last heuristic.
- Full multi-codebook EnCodec prediction is not implemented.
- Full FACodec future waveform prediction is not implemented.
- The larger EnCodec medium model was trained, but full waveform-level evaluation was not completed.
- Objective metrics may not fully capture semantic or perceptual continuity of predicted speech.

---

## Future Work

Potential extensions include:

- Joint prediction of all EnCodec codebooks
- Hierarchical multi-codebook generation
- Conditional prediction of residual codebooks
- Better sampling and decoding strategies
- Longer-context and longer-horizon speech prediction
- Speaker-conditioned future speech modeling
- Subjective listening evaluation
- Comparison with diffusion-based speech continuation models

---

## Notes on Repository Contents

This repository excludes large or non-redistributable files, including:

- Raw speech datasets
- Generated waveform files
- Codec token caches
- Model checkpoints
- Large temporary experiment artifacts

These files can be regenerated using the scripts provided in the repository.

---

## License

This repository is intended for research and educational use.
