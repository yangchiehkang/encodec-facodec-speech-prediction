# EnCodec Speech Prediction Experiment Status

Date: 2026-04-29  
Project: speech_prediction_project  
Environment: speech_pred  
Platform: Windows  
Project Root: E:\Working\deep-learning\speech_prediction_project  
Dataset Root: E:\Working\deep-learning\datasets\wsjcam0_wavs  

---

## 1. Environment Status

Current conda environment:

```text
speech_pred
Core setup:

复制
Python environment: speech_pred
PyTorch CUDA: available
Device used by EnCodec: cuda
The EnCodec wrapper can be loaded successfully.

Command:

复制
python codec\encodec_wrapper.py
Observed output:

复制
EnCodec 24 kHz model loaded.
Device      : cuda
Sample rate : 24000
Channels    : 1
Bandwidth   : 6.0 kbps
Note:

复制
FutureWarning from torch.nn.utils.weight_norm appeared.
This is only a warning from PyTorch/EnCodec dependency and does not affect the current experiment.
2. EnCodec Encode-Decode Test
The encode-decode reconstruction pipeline has been tested successfully.

Command:

复制
powershell -Command "$wav = Get-Content data\train_debug.txt -TotalCount 1; python test_encodec.py --wav $wav --out outputs\recon.wav --bandwidth 6.0"
Observed output:

复制
Input wav shape: torch.Size([1, 132878])
Token shape: torch.Size([8, 416])
Reconstructed wav shape: torch.Size([1, 133120])
Saved to: outputs\recon.wav
Generated file:

复制
outputs\recon.wav
Status:

复制
PASSED
Notes:

复制
The reconstructed waveform length is slightly different from the input waveform length.
This is expected because EnCodec operates with frame/chunk alignment internally.
3. Token Extraction Script Status
Token extraction script has been created and verified.

Script:

复制
codec\extract_encodec_tokens.py
Help command:

复制
python codec\extract_encodec_tokens.py --help
Observed available arguments:

复制
--filelist FILELIST
--out_dir OUT_DIR
--bandwidth BANDWIDTH
--limit LIMIT
--skip_existing
Status:

复制
PASSED
Important note:

复制
The implemented argument name is --out_dir.
The document screenshot used --output_dir, but the current project uses --out_dir.
4. Debug Token Extraction Results
Debug token extraction was completed for train, valid, and test subsets.

4.1 Train Debug Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\train_debug.txt --out_dir tokens\encodec\train_debug --bandwidth 6.0
Observed result:

复制
Num wav files : 100
Success : 100
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\train_debug
Output directory:

复制
tokens\encodec\train_debug
File count:

复制
100 .pt files
Total size:

复制
3,410,096 bytes
Status:

复制
PASSED
4.2 Valid Debug Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\valid_debug.txt --out_dir tokens\encodec\valid_debug --bandwidth 6.0
Observed result:

复制
Num wav files : 20
Success : 20
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\valid_debug
Output directory:

复制
tokens\encodec\valid_debug
File count:

复制
20 .pt files
Total size:

复制
548,464 bytes
Status:

复制
PASSED
4.3 Test Debug Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\test_debug.txt --out_dir tokens\encodec\test_debug --bandwidth 6.0
Observed result:

复制
Num wav files : 20
Success : 20
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\test_debug
Output directory:

复制
tokens\encodec\test_debug
File count:

复制
20 .pt files
Total size:

复制
851,248 bytes
Status:

复制
PASSED
5. Full Token Extraction Results
Full token extraction was completed for train, valid, and test sets.

5.1 Train Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\train.txt --out_dir tokens\encodec\train --bandwidth 6.0 --skip_existing
Observed result:

复制
Num wav files : 12479
Success : 12479
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\train
Processing speed:

复制
12479/12479 [16:01, 12.98it/s]
Output directory:

复制
tokens\encodec\train
Status:

复制
PASSED
5.2 Valid Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\valid.txt --out_dir tokens\encodec\valid --bandwidth 6.0 --skip_existing
Observed result:

复制
Num wav files : 1176
Success : 1176
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\valid
Processing speed:

复制
1176/1176 [01:24, 13.84it/s]
Output directory:

复制
tokens\encodec\valid
Status:

复制
PASSED
5.3 Test Tokens
Command:

复制
python codec\extract_encodec_tokens.py --filelist data\test.txt --out_dir tokens\encodec\test --bandwidth 6.0 --skip_existing
Observed result:

复制
Num wav files : 644
Success : 644
Skipped : 0
Failed  : 0
Saved to: tokens\encodec\test
Processing speed:

复制
644/644 [00:53, 11.96it/s]
Output directory:

复制
tokens\encodec\test
Status:

复制
PASSED
6. Dataset Verification
Dataset file has been created and tested.

Script:

复制
data\dataset.py
Test command:

复制
python data\dataset.py --token_dir tokens\encodec\train_debug --block_size 224 --codebook_idx 0
Observed output:

复制
Token dir     : tokens\encodec\train_debug
Num samples   : 100
Input shape   : torch.Size([224])
Target shape  : torch.Size([224])
Input dtype    : torch.int64
Target dtype   : torch.int64
First pt file  : tokens\encodec\train_debug\00000000_011a0101_49a0f2bdde.pt
Original wav   : E:\Working\deep-learning\datasets\wsjcam0_wavs\si_tr_s\011\011a0101.wav
Dataset behavior:

复制
Input sequence : first-codebook EnCodec tokens except final token
Target sequence: first-codebook EnCodec tokens shifted by one step
Current modeling choice:

复制
first-codebook prediction
Status:

复制
PASSED
7. Transformer LM Verification
Transformer LM file has been created and tested.

Script:

复制
models\transformer_lm.py
Test command:

复制
python models\transformer_lm.py
Observed output:

复制
Vocab size      : 1024
Block size      : 224
Layers          : 4
Heads           : 4
Embedding dim   : 256
Num parameters  : 3,741,184
Input shape     : torch.Size([2, 224])
Logits shape    : torch.Size([2, 224, 1024])
Loss            : 7.0029
Generated shape : torch.Size([2, 15])
Model architecture includes:

复制
Token embedding
Positional embedding
Causal self-attention
Feed-forward network
Layer normalization
Vocabulary projection head
Training objective:

复制
Cross-entropy loss for next-token prediction
Status:

复制
PASSED
8. Current Generated Files and Directories
Important scripts:

复制
codec\encodec_wrapper.py
codec\extract_encodec_tokens.py
data\dataset.py
models\transformer_lm.py
test_encodec.py
Important generated outputs:

复制
outputs\recon.wav
tokens\encodec\train_debug
tokens\encodec\valid_debug
tokens\encodec\test_debug
tokens\encodec\train
tokens\encodec\valid
tokens\encodec\test
Important data filelists:

复制
data\train.txt
data\valid.txt
data\test.txt
data\train_debug.txt
data\valid_debug.txt
data\test_debug.txt
9. Dataset and Token Counts
Split	Filelist	Token Directory	Number of WAVs / PT files	Status
train_debug	data\train_debug.txt	tokens\encodec\train_debug	100	Done
valid_debug	data\valid_debug.txt	tokens\encodec\valid_debug	20	Done
test_debug	data\test_debug.txt	tokens\encodec\test_debug	20	Done
train	data\train.txt	tokens\encodec\train	12479	Done
valid	data\valid.txt	tokens\encodec\valid	1176	Done
test	data\test.txt	tokens\encodec\test	644	Done
10. Current Experiment Status Summary
Completed stages:

复制
Stage 1: Environment setup                          PASSED
Stage 2: WSJCAM0 filelist preparation               PASSED
Stage 3: EnCodec wrapper verification               PASSED
Stage 4: Encode-decode reconstruction test          PASSED
Stage 5: Debug token extraction                     PASSED
Stage 6: Full token extraction                      PASSED
Stage 7: Dataset loading test                       PASSED
Stage 8: Transformer LM forward/generation test     PASSED
The project is now ready for the next stage:

复制
Create training script for EnCodec Transformer LM.
Expected next script:

复制
scripts\train_transformer_lm.py
Recommended first training target:

复制
Use tokens\encodec\train_debug and tokens\encodec\valid_debug
to verify the full training loop before training on the full dataset.
11. Notes
The EnCodec tokenizer uses 24 kHz audio.
Current bandwidth is 6.0 kbps.
Current token shape example from one debug sample:
复制
Token shape: torch.Size([8, 416])
Current model only uses:
复制
tokens[0, :]
This means only the first EnCodec codebook is used for language modeling.

Full multi-codebook prediction is not implemented yet.

The PyTorch FutureWarning about weight_norm is non-blocking and can be ignored for now.

12. Final Checkpoint
As of 2026-04-29, the preprocessing and model sanity-check pipeline is complete.

The next meaningful experiment step is:

复制
Train the Transformer LM on debug tokens first.
Then train on full tokens after confirming the loss decreases correctly.
复制

---

## 3. 检查文件是否保存成功

保存后，在命令行执行：

```bat
dir experiment_status_20260429.md
---

## 13. Debug Training Results

Date: 2026-04-29  
Stage: 5.3.4 Debug training  
Status: PASSED  

Two debug training runs were completed successfully.

---

### 13.1 Debug Run A: transformer_lm_debug

Command:

```bat
python train.py ^
--train_token_dir tokens\encodec\train_debug ^
--valid_token_dir tokens\encodec\valid_debug ^
--output_dir checkpoints\transformer_lm_debug ^
--vocab_size 1024 ^
--seq_len 224 ^
--batch_size 8 ^
--epochs 2 ^
--d_model 256 ^
--n_heads 4 ^
--n_layers 4 ^
--ffn_dim 1024 ^
--lr 3e-4 ^
--seed 42
Configuration:

复制
Train token dir : tokens\encodec\train_debug
Valid token dir : tokens\encodec\valid_debug
Output dir      : checkpoints\transformer_lm_debug
Vocab size      : 1024
Seq len         : 224
Batch size      : 8
Epochs          : 2
d_model         : 256
n_heads         : 4
n_layers        : 4
ffn_dim         : 1024
LR              : 0.0003
Seed            : 42
Model parameters: 3,741,184
Train samples   : 100
Valid samples   : 20
Train batches   : 13
Valid batches   : 3
Training results:

Epoch	Train Loss	Valid Loss	Train PPL	Valid PPL	Best
1	5.938974	5.414161	379.5454	224.5642	Yes
2	4.847068	4.759075	127.3664	116.6380	Yes
Final result:

复制
Best valid loss : 4.759075
Final checkpoint: checkpoints\transformer_lm_debug\last.pt
Best checkpoint : checkpoints\transformer_lm_debug\best.pt
Log file        : checkpoints\transformer_lm_debug\logs\train_log.csv
Status:

复制
PASSED
13.2 Debug Run B: encodec_debug
Command:

复制
python train.py ^
--train_token_dir tokens\encodec\train_debug ^
--valid_token_dir tokens\encodec\valid_debug ^
--output_dir checkpoints\encodec_debug ^
--vocab_size 1024 ^
--seq_len 256 ^
--batch_size 8 ^
--epochs 3 ^
--d_model 128 ^
--n_heads 4 ^
--n_layers 2 ^
--ffn_dim 512 ^
--lr 0.0003 ^
--seed 42
Configuration:

复制
Train token dir : tokens\encodec\train_debug
Valid token dir : tokens\encodec\valid_debug
Output dir      : checkpoints\encodec_debug
Vocab size      : 1024
Seq len         : 256
Batch size      : 8
Epochs          : 3
d_model         : 128
n_heads         : 4
n_layers        : 2
ffn_dim         : 512
LR              : 0.0003
Seed            : 42
Model parameters: 691,712
Train samples   : 100
Valid samples   : 20
Train batches   : 13
Valid batches   : 3
Training results:

Epoch	Train Loss	Valid Loss	Train PPL	Valid PPL	Best
1	6.466403	6.144529	643.1662	466.1600	Yes
2	5.789757	5.574577	326.9334	263.6380	Yes
3	5.218153	5.151718	184.5930	172.7281	Yes
Final result:

复制
Best valid loss : 5.151718
Final checkpoint: checkpoints\encodec_debug\last.pt
Best checkpoint : checkpoints\encodec_debug\best.pt
Log file        : checkpoints\encodec_debug\logs\train_log.csv
Status:

复制
PASSED
Conclusion:

复制
The debug training pipeline is fully functional.
Dataset loading, batching, model forward pass, loss computation, backpropagation,
validation, checkpoint saving, and CSV logging all work correctly.
14. Full Training Results: encodec_small
Date: 2026-04-29

Stage: 5.3.5 Full training

Status: PASSED

The full training run was completed successfully using the complete EnCodec token dataset.

Command:

复制
python train.py ^
--train_token_dir tokens\encodec\train ^
--valid_token_dir tokens\encodec\valid ^
--output_dir checkpoints\encodec_small ^
--vocab_size 1024 ^
--seq_len 512 ^
--batch_size 8 ^
--epochs 20 ^
--d_model 256 ^
--n_heads 4 ^
--n_layers 4 ^
--ffn_dim 1024 ^
--lr 0.0003 ^
--seed 42
Configuration:

复制
Train token dir : tokens\encodec\train
Valid token dir : tokens\encodec\valid
Output dir      : checkpoints\encodec_small
Vocab size      : 1024
Seq len         : 512
Batch size      : 8
Epochs          : 20
d_model         : 256
n_heads         : 4
n_layers        : 4
ffn_dim         : 1024
LR              : 0.0003
Seed            : 42
Model parameters: 3,814,912
Train samples   : 12479
Valid samples   : 1176
Train batches   : 1560
Valid batches   : 147
Training results:

Epoch	Train Loss	Valid Loss	Train PPL	Valid PPL	Best Updated
1	2.727142	2.707173	15.2891	14.9869	Yes
2	2.556899	2.618148	12.8958	13.7103	Yes
3	2.481961	2.563526	11.9647	12.9815	Yes
4	2.443189	2.543187	11.5097	12.7201	Yes
5	2.419149	2.530182	11.2363	12.5558	Yes
6	2.402548	2.514984	11.0513	12.3664	Yes
7	2.387509	2.508371	10.8863	12.2849	Yes
8	2.377149	2.506053	10.7741	12.2565	Yes
9	2.367141	2.502262	10.6669	12.2101	Yes
10	2.358155	2.499215	10.5714	12.1729	Yes
11	2.351000	2.495421	10.4961	12.1268	Yes
12	2.342813	2.492646	10.4105	12.0932	Yes
13	2.336990	2.493140	10.3500	12.0992	No
14	2.331530	2.489521	10.2937	12.0555	Yes
15	2.324502	2.492446	10.2216	12.0908	No
16	2.319932	2.491170	10.1750	12.0754	No
17	2.315433	2.488646	10.1293	12.0450	Yes
18	2.310022	2.486881	10.0746	12.0237	Yes
19	2.305965	2.490096	10.0339	12.0624	No
20	2.301199	2.487984	9.9862	12.0370	No
Final result:

复制
Best valid loss : 2.486881
Final checkpoint: checkpoints\encodec_small\last.pt
Best checkpoint : checkpoints\encodec_small\best.pt
Log file        : checkpoints\encodec_small\logs\train_log.csv
Best epoch:

复制
Epoch 18
Best checkpoint:

复制
checkpoints\encodec_small\best.pt
Last checkpoint:

复制
checkpoints\encodec_small\last.pt
Checkpoint directory:

复制
checkpoints\encodec_small\checkpoints
Expected checkpoint files:

复制
epoch_001.pt
epoch_002.pt
epoch_003.pt
epoch_004.pt
epoch_005.pt
epoch_006.pt
epoch_007.pt
epoch_008.pt
epoch_009.pt
epoch_010.pt
epoch_011.pt
epoch_012.pt
epoch_013.pt
epoch_014.pt
epoch_015.pt
epoch_016.pt
epoch_017.pt
epoch_018.pt
epoch_019.pt
epoch_020.pt
15. Training Analysis
The full training run behaved normally.

Key observations:

复制
1. Train loss decreased steadily from 2.727142 to 2.301199.
2. Validation loss decreased from 2.707173 to the best value 2.486881.
3. Best validation performance occurred at epoch 18.
4. Epochs 19 and 20 did not improve over epoch 18.
5. There is no severe overfitting, but validation improvement becomes very small after epoch 12.
6. The model appears to have reached a stable plateau around valid loss 2.49.
Best validation metric:

复制
Best valid loss : 2.486881
Best valid ppl  : 12.0237
Best epoch      : 18
Final epoch metric:

复制
Final train loss: 2.301199
Final valid loss: 2.487984
Final train ppl : 9.9862
Final valid ppl : 12.0370
Interpretation:

复制
The model successfully learned meaningful first-codebook token transitions.
The validation loss curve is stable and suggests that the trained model is usable
for the next generation/sampling stage.
Recommended checkpoint for generation:

复制
checkpoints\encodec_small\best.pt
Do not prefer this checkpoint unless specifically testing final-epoch behavior:

复制
checkpoints\encodec_small\last.pt
16. Current Project Status After Full Training
Completed stages:

复制
Stage 1: Environment setup                          PASSED
Stage 2: Dataset filelist preparation               PASSED
Stage 3: EnCodec wrapper verification               PASSED
Stage 4: Encode-decode reconstruction test          PASSED
Stage 5: Debug token extraction                     PASSED
Stage 6: Full token extraction                      PASSED
Stage 7: Dataset loading test                       PASSED
Stage 8: Transformer LM sanity check                PASSED
Stage 9: Debug training                             PASSED
Stage 10: Full Transformer LM training              PASSED
Current best model:

复制
checkpoints\encodec_small\best.pt
Current training log:

复制
checkpoints\encodec_small\logs\train_log.csv
Next recommended stage:

复制
Create generation script to sample EnCodec first-codebook tokens from the trained Transformer LM.
Then reconstruct audio or analyze generated token sequences.
Expected next script:

复制
generate.py
复制

---

## 3. 保存后检查文件

执行：

```bat
dir experiment_status_20260429.md