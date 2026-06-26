# Future Speech Generation Experiment Status

Date: 2026-04-30  
Project: speech_prediction_project  
Environment: speech_pred  
Platform: Windows  
Project Root: E:\Working\deep-learning\speech_prediction_project  
Stage: 5.4 Future Speech Generation  

---

## 1. Objective

This stage aims to generate future speech from a given speech context.

Main task:

```text
Given the first 2 seconds of speech, predict the future 1 second of speech.
Time segmentation:

复制
Context    : 0s - 2s
Prediction : 2s - 3s
Reference  : 2s - 3s
Evaluation should only compare:

复制
Prediction waveform vs Reference waveform
The context segment is used only as model input and should not be included in future-speech evaluation.

2. Environment and Working Directory
Conda environment:

复制
speech_pred
Command used:

复制
conda activate speech_pred
cd /d "E:\Working\deep-learning\speech_prediction_project"
Working directory:

复制
E:\Working\deep-learning\speech_prediction_project
Device used:

复制
cuda
3. Checkpoint Used for Generation
The generation stage uses the best checkpoint from the full Transformer LM training stage.

Checkpoint:

复制
checkpoints\encodec_small\best.pt
Checkpoint information loaded by generate.py:

复制
Vocab size       : 1024
Block size       : 512
Layers           : 4
Heads            : 4
Embedding dim    : 256
Checkpoint epoch : 18
Best valid loss  : 2.486880951998185
This checkpoint was selected because it achieved the lowest validation loss during full training.

4. Initial Generation Script Verification
Before updating the first-codebook fill strategy interface, the initial generate.py script supported the following parameter:

复制
--codebook_fill {repeat_last,repeat_mean,cycle_context,zero}
Help command:

复制
python generate.py --help
Observed relevant output:

复制
--codebook_fill {repeat_last,repeat_mean,cycle_context,zero}
Status:

复制
PASSED
5. Initial Generation Run with codebook_fill repeat_last
Command:

复制
python generate.py ^
--checkpoint checkpoints\encodec_small\best.pt ^
--test_filelist data\test.txt ^
--out_dir generated\encodec_small ^
--num_examples 1 ^
--context_sec 2.0 ^
--pred_sec 1.0 ^
--bandwidth 6.0 ^
--temperature 1.0 ^
--top_k 50 ^
--top_p 1.0 ^
--codebook_fill repeat_last ^
--seed 42
Observed configuration:

复制
Device        : cuda
Checkpoint    : checkpoints\encodec_small\best.pt
Output dir    : generated\encodec_small
Context sec   : 2.0
Predict sec   : 1.0
Bandwidth     : 6.0
Temperature   : 1.0
Top-k         : 50
Top-p         : 1.0
Greedy        : False
Codebook fill : repeat_last
Generation details:

复制
Generated first-codebook future tokens: 75 tokens
Input WAV: E:\Working\deep-learning\datasets\wsjcam0_wavs\si_et_05\440\440c0201.wav
Generated files:

复制
generated\encodec_small\0000_440c0201_context_0_2s.wav
generated\encodec_small\0000_440c0201_pred_future_2_3s.wav
generated\encodec_small\0000_440c0201_reference_future_2_3s.wav
generated\encodec_small\0000_440c0201_combined_context_pred_0_3s.wav
generated\encodec_small\0000_440c0201_tokens.pt
generated\encodec_small\0000_440c0201_meta.json
generated\encodec_small\generation_summary.json
Status:

复制
PASSED
Note:

复制
A PyTorch FutureWarning related to torch.nn.utils.weight_norm appeared.
This warning comes from the EnCodec dependency and does not block generation.
6. Updated First-Codebook Fill Strategy Interface
The generation script was then updated according to section 5.4.2.

Old parameter:

复制
--codebook_fill
New parameter:

复制
--fill_strategy
Supported strategies after update:

复制
repeat_last
repeat_context
Additional parameter:

复制
--repeat_context_tokens
Updated help command:

复制
python generate.py --help
Observed relevant output:

复制
--fill_strategy {repeat_last,repeat_context}
--repeat_context_tokens REPEAT_CONTEXT_TOKENS
Status:

复制
PASSED
7. Fill Strategy Explanation
The current mainline model only predicts future tokens for the first EnCodec codebook.

Predicted by Transformer LM:

复制
codebook 0 future tokens
Not predicted by Transformer LM:

复制
codebook 1 to codebook N future tokens
However, EnCodec decoding requires all codebooks. Therefore, non-first future codebooks must be filled using an approximation strategy.

7.1 repeat_last
Strategy:

复制
--fill_strategy repeat_last
Meaning:

复制
For each non-first codebook, repeat the last context token of that codebook
across all future token positions.
In other words:

复制
future_codes[q, :] = context_codes[q, -1]
for q >= 1
This is a simple and stable local approximation.

7.2 repeat_context
Strategy:

复制
--fill_strategy repeat_context
Meaning:

复制
For each non-first codebook, take the tail segment of context tokens and
cyclically repeat it to fill the future segment.
Current setting:

复制
--repeat_context_tokens 75
Since EnCodec 24 kHz has an approximate frame rate of 75 tokens per second, this means:

复制
Use approximately the last 1 second of context tokens to fill the future non-first codebooks.
8. Generation Run with fill_strategy repeat_last
Command:

复制
python generate.py ^
--checkpoint checkpoints\encodec_small\best.pt ^
--test_filelist data\test.txt ^
--out_dir generated\encodec_small_repeat_last ^
--num_examples 1 ^
--context_sec 2.0 ^
--pred_sec 1.0 ^
--bandwidth 6.0 ^
--temperature 1.0 ^
--top_k 50 ^
--top_p 1.0 ^
--fill_strategy repeat_last ^
--seed 42
Observed configuration:

复制
Device                : cuda
Checkpoint            : checkpoints\encodec_small\best.pt
Output dir            : generated\encodec_small_repeat_last
Context sec           : 2.0
Predict sec           : 1.0
Bandwidth             : 6.0
Temperature           : 1.0
Top-k                 : 50
Top-p                 : 1.0
Greedy                : False
Fill strategy         : repeat_last
Repeat context tokens : 75
Loaded checkpoint:

复制
Checkpoint epoch: 18
Best valid loss : 2.486880951998185
Generation details:

复制
Generated first-codebook future tokens: 75 tokens
Input WAV: E:\Working\deep-learning\datasets\wsjcam0_wavs\si_et_05\440\440c0201.wav
Generated files:

复制
generated\encodec_small_repeat_last\0000_440c0201_context_0_2s.wav
generated\encodec_small_repeat_last\0000_440c0201_pred_future_2_3s.wav
generated\encodec_small_repeat_last\0000_440c0201_reference_future_2_3s.wav
generated\encodec_small_repeat_last\0000_440c0201_combined_context_pred_0_3s.wav
generated\encodec_small_repeat_last\0000_440c0201_tokens.pt
generated\encodec_small_repeat_last\0000_440c0201_meta.json
generated\encodec_small_repeat_last\generation_summary.json
Status:

复制
PASSED
9. Generation Run with fill_strategy repeat_context
Command:

复制
python generate.py ^
--checkpoint checkpoints\encodec_small\best.pt ^
--test_filelist data\test.txt ^
--out_dir generated\encodec_small_repeat_context ^
--num_examples 1 ^
--context_sec 2.0 ^
--pred_sec 1.0 ^
--bandwidth 6.0 ^
--temperature 1.0 ^
--top_k 50 ^
--top_p 1.0 ^
--fill_strategy repeat_context ^
--repeat_context_tokens 75 ^
--seed 42
Observed configuration:

复制
Device                : cuda
Checkpoint            : checkpoints\encodec_small\best.pt
Output dir            : generated\encodec_small_repeat_context
Context sec           : 2.0
Predict sec           : 1.0
Bandwidth             : 6.0
Temperature           : 1.0
Top-k                 : 50
Top-p                 : 1.0
Greedy                : False
Fill strategy         : repeat_context
Repeat context tokens : 75
Loaded checkpoint:

复制
Checkpoint epoch: 18
Best valid loss : 2.486880951998185
Generation details:

复制
Generated first-codebook future tokens: 75 tokens
Input WAV: E:\Working\deep-learning\datasets\wsjcam0_wavs\si_et_05\440\440c0201.wav
Generated files:

复制
generated\encodec_small_repeat_context\0000_440c0201_context_0_2s.wav
generated\encodec_small_repeat_context\0000_440c0201_pred_future_2_3s.wav
generated\encodec_small_repeat_context\0000_440c0201_reference_future_2_3s.wav
generated\encodec_small_repeat_context\0000_440c0201_combined_context_pred_0_3s.wav
generated\encodec_small_repeat_context\0000_440c0201_tokens.pt
generated\encodec_small_repeat_context\0000_440c0201_meta.json
generated\encodec_small_repeat_context\generation_summary.json
Status:

复制
PASSED
10. Summary of Generated Samples
Run	Output Directory	Fill Strategy	Num Examples	Status
Initial generation	generated\encodec_small	codebook_fill repeat_last	1	PASSED
Updated strategy A	generated\encodec_small_repeat_last	fill_strategy repeat_last	1	PASSED
Updated strategy B	generated\encodec_small_repeat_context	fill_strategy repeat_context	1	PASSED
All runs used the same test utterance:

复制
E:\Working\deep-learning\datasets\wsjcam0_wavs\si_et_05\440\440c0201.wav
All runs used the same checkpoint:

复制
checkpoints\encodec_small\best.pt
11. Important Output Files for Evaluation
For repeat_last, the prediction and reference files are:

复制
generated\encodec_small_repeat_last\0000_440c0201_pred_future_2_3s.wav
generated\encodec_small_repeat_last\0000_440c0201_reference_future_2_3s.wav
For repeat_context, the prediction and reference files are:

复制
generated\encodec_small_repeat_context\0000_440c0201_pred_future_2_3s.wav
generated\encodec_small_repeat_context\0000_440c0201_reference_future_2_3s.wav
Only these 2s-3s future segments should be used for objective evaluation.

Do not evaluate the full combined 0s-3s waveform directly, because the first 2 seconds are given context rather than predicted speech.

12. Technical Note on the Approximation
The current system is a first-codebook-only speech prediction system.

This means:

复制
The Transformer LM predicts only the future token sequence of codebook 0.
The remaining codebooks are filled using heuristic strategies.

This approximation is necessary because of local GPU memory and modeling constraints.

Current approximation methods:

复制
repeat_last:
    Stable and simple. Uses the final context token for each non-first codebook.

repeat_context:
    More dynamic than repeat_last. Uses the tail context pattern to fill future tokens.
Limitation:

复制
The generated waveform quality may be limited because fine acoustic details are not
fully predicted. Non-first codebooks contain important residual and fine-grained
audio information.
Potential future improvement:

复制
Train a multi-codebook prediction model.
Use hierarchical coarse-to-fine token generation.
Predict codebook 0 first, then predict later codebooks conditioned on previous codebooks.
13. Current Stage Status
Completed generation-related stages:

复制
5.4.1 Create generation script                         PASSED
5.4.2 Implement first-codebook fill strategies          PASSED
Single-sample generation with repeat_last               PASSED
Single-sample generation with repeat_context            PASSED
Current best checkpoint for generation:

复制
checkpoints\encodec_small\best.pt
Current generated output directories:

复制
generated\encodec_small
generated\encodec_small_repeat_last
generated\encodec_small_repeat_context
Next recommended stage:

复制
Run generation on multiple test samples.
Then implement objective evaluation for prediction vs reference.
Recommended next script:

复制
evaluate_generation.py
复制

---

## 3. 检查文档是否保存成功

保存后执行：

```bat
dir generation_status_20260430.md
也可以查看末尾内容：

复制
powershell -Command "Get-Content generation_status_20260430.md -Tail 40"
4. 同步追加到总实验记录，可选但推荐
如果你还想把这部分也追加到总记录：

复制
notepad experiment_status_20260429.md
在末尾追加一个简短索引：

复制
---

## 17. Future Speech Generation Stage

Date: 2026-04-30  
Status: PASSED  

Generation script `generate.py` was created and updated to support first-codebook decoding fill strategies.

Verified strategies:

```text
--fill_strategy repeat_last
--fill_strategy repeat_context
Successful output directories:

复制
generated\encodec_small
generated\encodec_small_repeat_last
generated\encodec_small_repeat_context
Best checkpoint used:

复制
checkpoints\encodec_small\best.pt
Detailed generation-stage record:

复制
generation_status_20260430.md
复制

---

## 当前阶段结论

可以把这一步标记为：

```text
5.4.1 generate.py 创建与运行             PASSED
5.4.2 fill_strategy repeat_last 验证      PASSED
5.4.2 fill_strategy repeat_context 验证   PASSED

---

# 5.5 Evaluation: STOI, PESQ and DNSMOS

Date: 2026-04-30  
Environment: speech_pred  
Project Root: E:\Working\deep-learning\speech_prediction_project  

---

## 5.5.1 Create Evaluation Script

Created evaluation script:

```text
evaluate.py
The script reads generated prediction waveforms and reference waveforms, then computes:

复制
STOI
PESQ
DNSMOS placeholder columns
Evaluation sampling rate:

复制
16000 Hz
Since the EnCodec generation output is 24 kHz, both prediction and reference waveforms are resampled to 16 kHz before computing STOI and PESQ.

DNSMOS is not integrated in the current local script. The DNSMOS columns are reserved as NaN.

Status:

复制
PASSED
5.5.2 Run Evaluation on encodec_small
Evaluation command:

复制
python evaluate.py ^
--pred_dir generated\encodec_small\pred ^
--ref_dir generated\encodec_small\ref ^
--out_csv outputs\eval_encodec_small.csv ^
--out_json outputs\eval_encodec_small_summary.json ^
--eval_sr 16000
Evaluation progress:

复制
Evaluating: 100%|████████████████████████████████████████████████████████████████████| 644/644 [00:44<00:00, 14.61it/s]
Output files:

复制
outputs\eval_encodec_small.csv
outputs\eval_encodec_small_summary.json
Status:

复制
PASSED
5.5.3 Evaluation Results
Summary loaded from:

复制
outputs\eval_encodec_small_summary.json
Result:

复制
{
  "pred_dir": "generated\\encodec_small\\pred",
  "ref_dir": "generated\\encodec_small\\ref",
  "eval_sr": 16000,
  "num_total": 644,
  "num_valid": 644,
  "stoi_mean": 0.3506506825906483,
  "pesq_mean": 1.1083880492619105,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "STOI and PESQ are computed after resampling prediction and reference to 16 kHz. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Main metrics:

Metric	Value
Number of evaluated files	644
Number of valid files	644
Mean STOI	0.3506506825906483
Mean PESQ	1.1083880492619105
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
5.5.4 Warning Observed During STOI Computation
During evaluation, the following warning appeared:

复制
RuntimeWarning: Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. Returning 1e-5. Please check you wav files
This warning came from pystoi.

Affected stage:

复制
STOI computation
The evaluation still completed successfully:

复制
Total files : 644
Valid files : 644
Status:

复制
NON-BLOCKING WARNING
5.5.5 Current Evaluation Conclusion
The objective evaluation stage was successfully completed for the official encodec_small generation outputs.

Final evaluation result:

复制
Mean STOI : 0.3506506825906483
Mean PESQ : 1.1083880492619105
DNSMOS was not computed in this run because the external DNSMOS model was not integrated.

Current stage status:

复制
5.5.1 Create evaluate.py                         PASSED
5.5.2 Run STOI/PESQ evaluation on encodec_small  PASSED
5.5.3 Save per-utterance CSV results             PASSED
5.5.4 Save summary JSON results                  PASSED
DNSMOS placeholder columns                       RECORDED AS NaN
复制

---

## 3. 可选：检查记录文件末尾

```bat
powershell -Command "Get-Content generation_status_20260430.md -Tail 80"
4. 当前结果一句话版
可以在报告里写：

复制
For the official encodec_small generation results, 644 utterances were evaluated. 
The mean STOI was 0.3507 and the mean PESQ was 1.1084 after resampling both prediction and reference signals to 16 kHz. DNSMOS was left as NaN because the external DNSMOS model was not integrated in the local evaluation script.

---

# 5.5 Evaluation Results: Debug and Formal Runs

Date: 2026-04-30  
Environment: speech_pred  
Project Root: E:\Working\deep-learning\speech_prediction_project  

---

## 5.5.2 Run Debug Evaluation

Debug prediction directory:

```text
generated\encodec_debug\pred
Debug reference directory:

复制
generated\encodec_debug\ref
Command:

复制
python evaluate.py ^
--pred_dir generated\encodec_debug\pred ^
--ref_dir generated\encodec_debug\ref ^
--out_csv outputs\encodec_debug_results.csv ^
--out_json outputs\encodec_debug_summary.json ^
--eval_sr 16000
Progress:

复制
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 20/20 [00:03<00:00,  5.88it/s]
Output files:

复制
outputs\encodec_debug_results.csv
outputs\encodec_debug_summary.json
Summary:

复制
{
  "pred_dir": "generated\\encodec_debug\\pred",
  "ref_dir": "generated\\encodec_debug\\ref",
  "eval_sr": 16000,
  "num_total": 20,
  "num_valid": 20,
  "stoi_mean": 0.29033523705493636,
  "pesq_mean": 1.0755568325519562,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "STOI and PESQ are computed after resampling prediction and reference to 16 kHz. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Debug evaluation result table:

Metric	Value
Number of files	20
Valid files	20
Mean STOI	0.29033523705493636
Mean PESQ	1.0755568325519562
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
Status:

复制
PASSED
5.5.3 Run Formal Evaluation
Formal prediction directory:

复制
generated\encodec_small\pred
Formal reference directory:

复制
generated\encodec_small\ref
Command:

复制
python evaluate.py ^
--pred_dir generated\encodec_small\pred ^
--ref_dir generated\encodec_small\ref ^
--out_csv outputs\encodec_small_results.csv ^
--out_json outputs\encodec_small_summary.json ^
--eval_sr 16000
Progress:

复制
Evaluating: 100%|████████████████████████████████████████████████████████████████████| 644/644 [00:42<00:00, 15.10it/s]
Output files:

复制
outputs\encodec_small_results.csv
outputs\encodec_small_summary.json
Summary:

复制
{
  "pred_dir": "generated\\encodec_small\\pred",
  "ref_dir": "generated\\encodec_small\\ref",
  "eval_sr": 16000,
  "num_total": 644,
  "num_valid": 644,
  "stoi_mean": 0.3506506825906483,
  "pesq_mean": 1.1083880492619105,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "STOI and PESQ are computed after resampling prediction and reference to 16 kHz. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Formal evaluation result table:

Metric	Value
Number of files	644
Valid files	644
Mean STOI	0.3506506825906483
Mean PESQ	1.1083880492619105
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
Status:

复制
PASSED
5.5.4 Evaluation Warning
During formal evaluation, one non-blocking STOI warning appeared:

复制
RuntimeWarning: Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. Returning 1e-5. Please check you wav files
Source:

复制
pystoi
Impact:

复制
The evaluation completed successfully.
All 644 files were evaluated as valid files.
Status:

复制
NON-BLOCKING WARNING
5.5.5 Evaluation Output Summary
Generated evaluation files:

复制
outputs\encodec_debug_results.csv
outputs\encodec_debug_summary.json
outputs\encodec_small_results.csv
outputs\encodec_small_summary.json
Final formal evaluation metrics:

复制
Mean STOI : 0.3506506825906483
Mean PESQ : 1.1083880492619105
DNSMOS    : NaN, not computed in current local setup
Current stage status:

复制
5.5.1 Create evaluate.py                 PASSED
5.5.2 Run debug evaluation                PASSED
5.5.3 Run formal evaluation               PASSED
Save per-utterance CSV files              PASSED
Save summary JSON files                   PASSED
DNSMOS placeholder columns                RECORDED AS NaN
复制

---

## 3. 检查记录是否追加成功

```bat
powershell -Command "Get-Content generation_status_20260430.md -Tail 120"
4. 一句话结论
复制
Evaluation completed successfully. The debug set achieved STOI 0.2903 and PESQ 1.0756 over 20 utterances, while the formal encodec_small test set achieved STOI 0.3507 and PESQ 1.1084 over 644 utterances.
---

# 5.6 Top-k Sampling Comparison Experiment

Date: 2026-04-30  
Environment: speech_pred  
Project Root: E:\Working\deep-learning\speech_prediction_project  

---

## 5.6.1 Run Top-k Generation

Experiment purpose:

```text
Compare greedy decoding and top-k sampling for EnCodec token-based speech prediction.
Top-k generation setting:

复制
Sampling method : top-k sampling
top_k           : 20
temperature     : 0.8
fill_strategy   : repeat_last
context_sec     : 2.0
pred_sec        : 1.0
sample_rate     : 24000
Generated directory:

复制
generated\encodec_topk20
Organized output directories:

复制
generated\encodec_topk20\pred
generated\encodec_topk20\ref
generated\encodec_topk20\full
Status:

复制
PASSED
5.6.2 Evaluate Top-k Sampling
Evaluation command:

复制
python evaluate.py ^
--pred_dir generated\encodec_topk20\pred ^
--ref_dir generated\encodec_topk20\ref ^
--out_csv outputs\encodec_topk20_results.csv ^
--out_json outputs\encodec_topk20_summary.json ^
--eval_sr 16000
Evaluation progress:

复制
Evaluating: 100%|████████████████████████████████████████████████████████████████████| 644/644 [00:49<00:00, 13.12it/s]
Output files:

复制
outputs\encodec_topk20_results.csv
outputs\encodec_topk20_summary.json
Summary:

复制
{
  "pred_dir": "generated\\encodec_topk20\\pred",
  "ref_dir": "generated\\encodec_topk20\\ref",
  "eval_sr": 16000,
  "num_total": 644,
  "num_valid": 644,
  "stoi_mean": 0.203996640806126,
  "pesq_mean": 1.0862229399799561,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "STOI and PESQ are computed after resampling prediction and reference to 16 kHz. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Top-k evaluation result table:

Metric	Value
Number of files	644
Valid files	644
Mean STOI	0.203996640806126
Mean PESQ	1.0862229399799561
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
Status:

复制
PASSED
5.6.3 Compare Greedy Decoding and Top-k Sampling
Compared systems:

复制
Greedy decoding : generated\encodec_small
Top-k sampling  : generated\encodec_topk20
Evaluation files:

复制
outputs\encodec_small_summary.json
outputs\encodec_topk20_summary.json
Comparison table:

Method	Num Files	STOI Mean	PESQ Mean	DNSMOS
Greedy decoding / encodec_small	644	0.3506506825906483	1.1083880492619105	NaN
Top-k sampling / encodec_topk20	644	0.203996640806126	1.0862229399799561	NaN
Difference table:

Metric	Greedy	Top-k20	Difference, Top-k20 - Greedy
STOI	0.3506506825906483	0.203996640806126	-0.1466540417845223
PESQ	1.1083880492619105	1.0862229399799561	-0.0221651092819544
Relative change:

Metric	Relative Change
STOI	-41.82%
PESQ	-2.00%
5.6.4 Warning Observed During Top-k Evaluation
During top-k evaluation, one non-blocking STOI warning appeared:

复制
RuntimeWarning: Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. Returning 1e-5. Please check you wav files
Source:

复制
pystoi
Impact:

复制
The evaluation completed successfully.
All 644 files were evaluated as valid files.
Status:

复制
NON-BLOCKING WARNING
5.6.5 Top-k Sampling Experiment Conclusion
The top-k sampling experiment was completed successfully.

Main conclusion:

复制
Under the current setting, greedy decoding outperformed top-k sampling.
Detailed conclusion:

复制
Top-k sampling with top_k=20 and temperature=0.8 produced lower objective scores than greedy decoding.
The STOI score dropped from 0.3507 to 0.2040, a decrease of approximately 41.82%.
The PESQ score dropped slightly from 1.1084 to 1.0862, a decrease of approximately 2.00%.
Interpretation:

复制
The current model appears to benefit from deterministic greedy decoding.
Top-k sampling may introduce token-level randomness that hurts short-term intelligibility and alignment with the reference signal.
This effect is especially visible in STOI, while PESQ changes only slightly.
Current stage status:

复制
5.6.1 Run top-k generation                    PASSED
5.6.2 Evaluate top-k sampling                 PASSED
5.6.3 Compare greedy and top-k results        PASSED
Save top-k per-utterance CSV                  PASSED
Save top-k summary JSON                       PASSED
DNSMOS placeholder columns                    RECORDED AS NaN
复制

---

## 3. 检查记录是否追加成功

执行：

```bat
powershell -Command "Get-Content generation_status_20260430.md -Tail 140"
4. 报告里可以直接写的一句话
复制
In the top-k sampling comparison experiment, top-k20 sampling achieved a mean STOI of 0.2040 and a mean PESQ of 1.0862 over 644 utterances, while greedy decoding achieved a mean STOI of 0.3507 and a mean PESQ of 1.1084. Therefore, greedy decoding performed better under the current experimental setting, especially in terms of STOI.
---

# 6 Local Baselines

Date: 2026-04-30  
Environment: speech_pred  
Project Root: E:\Working\deep-learning\speech_prediction_project  

---

## 6.1 Copy Baseline

### Purpose

The copy baseline evaluates a simple non-learning strategy for future speech prediction.

The logic is:

```text
Input audio 0s-2s      : context
Input audio 1s-2s      : prediction
Input audio 2s-3s      : reference
Metric computation     : prediction vs reference
This baseline tests whether simply copying the previous one second of speech can predict the next one second.

Script
Created script:

复制
evaluate_copy_baseline.py
Command
复制
python evaluate_copy_baseline.py ^
--test_filelist data\test.txt ^
--output_dir generated\copy_baseline ^
--out_csv outputs\copy_baseline_results.csv ^
--out_json outputs\copy_baseline_summary.json ^
--context_sec 2.0 ^
--pred_sec 1.0 ^
--eval_sample_rate 16000
Progress
复制
Copy baseline: 100%|█████████████████████████████████████████████████████████████████| 644/644 [00:51<00:00, 12.56it/s]
Output Directories
复制
generated\copy_baseline\context
generated\copy_baseline\pred
generated\copy_baseline\ref
generated\copy_baseline\full
File counts:

复制
context : 644 wav files
pred    : 644 wav files
ref     : 644 wav files
full    : 644 wav files
Output Files
复制
outputs\copy_baseline_results.csv
outputs\copy_baseline_summary.json
Summary
复制
{
  "test_filelist": "data\\test.txt",
  "output_dir": "generated\\copy_baseline",
  "pred_dir": "generated\\copy_baseline\\pred",
  "ref_dir": "generated\\copy_baseline\\ref",
  "eval_sr": 16000,
  "context_sec": 2.0,
  "pred_sec": 1.0,
  "num_total": 644,
  "num_valid": 644,
  "stoi_mean": 0.13708409757084303,
  "pesq_mean": 1.0531656005367729,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "Copy baseline uses audio from context_sec - pred_sec to context_sec as prediction, and audio from context_sec to context_sec + pred_sec as reference. STOI and PESQ are computed after resampling prediction and reference to eval_sr. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Result Table
Metric	Value
Number of files	644
Valid files	644
Mean STOI	0.13708409757084303
Mean PESQ	1.0531656005367729
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
Status:

复制
PASSED
6.2 EnCodec Oracle Reconstruction
Purpose
The EnCodec oracle reconstruction evaluates the reconstruction upper bound of the EnCodec codec.

The logic is:

复制
Ground-truth audio 0s-3s
        -> EnCodec encoder
        -> ground-truth EnCodec tokens
        -> EnCodec decoder
        -> reconstructed waveform
Then the reconstructed future segment is compared with the original future segment:

复制
reconstructed waveform 2s-3s
vs
original reference waveform 2s-3s
This experiment measures the upper-bound quality introduced by EnCodec reconstruction itself.

Script
Created script:

复制
evaluate_oracle.py
EnCodec Package Check
复制
python -c "import encodec; print('encodec ok')"
Output:

复制
encodec ok
Command
复制
python evaluate_oracle.py ^
--codec encodec ^
--test_filelist data\test.txt ^
--output_dir generated\encodec_oracle ^
--out_csv outputs\encodec_oracle_results.csv ^
--out_json outputs\encodec_oracle_summary.json ^
--context_sec 2.0 ^
--pred_sec 1.0 ^
--eval_sample_rate 16000 ^
--bandwidth 6.0
Runtime configuration:

复制
Device          : cuda
Codec SR        : 24000
Bandwidth       : 6.0
Progress
复制
Oracle reconstruction: 100%|█████████████████████████████████████████████████████████| 644/644 [01:22<00:00,  7.85it/s]
Output Directories
复制
generated\encodec_oracle\context
generated\encodec_oracle\pred
generated\encodec_oracle\ref
generated\encodec_oracle\full
File counts:

复制
context : 644 wav files
pred    : 644 wav files
ref     : 644 wav files
full    : 644 wav files
Output Files
复制
outputs\encodec_oracle_results.csv
outputs\encodec_oracle_summary.json
Summary
复制
{
  "codec": "encodec",
  "test_filelist": "data\\test.txt",
  "output_dir": "generated\\encodec_oracle",
  "pred_dir": "generated\\encodec_oracle\\pred",
  "ref_dir": "generated\\encodec_oracle\\ref",
  "eval_sr": 16000,
  "codec_sample_rate": 24000,
  "bandwidth": 6.0,
  "context_sec": 2.0,
  "pred_sec": 1.0,
  "num_total": 644,
  "num_valid": 644,
  "stoi_mean": 0.9101189329379848,
  "pesq_mean": 2.5057239819387473,
  "dnsmos_sig_mean": NaN,
  "dnsmos_bak_mean": NaN,
  "dnsmos_ovrl_mean": NaN,
  "note": "Oracle reconstruction encodes and decodes the ground-truth 0-3s audio with EnCodec, then compares the reconstructed 2-3s segment against the original 2-3s reference. STOI and PESQ are computed after resampling prediction and reference to eval_sr. DNSMOS columns are reserved as NaN unless an external DNSMOS model is integrated."
}
Result Table
Metric	Value
Number of files	644
Valid files	644
Mean STOI	0.9101189329379848
Mean PESQ	2.5057239819387473
DNSMOS SIG	NaN
DNSMOS BAK	NaN
DNSMOS OVRL	NaN
Status:

复制
PASSED
6.3 Warning Notes
During both Copy baseline and EnCodec oracle reconstruction evaluation, the following non-blocking warning appeared for some samples:

复制
RuntimeWarning: Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. Returning 1e-5. Please check you wav files
Source:

复制
pystoi
Impact:

复制
The evaluations completed successfully.
All 644 files were evaluated as valid files in both baseline experiments.
A PyTorch FutureWarning also appeared during EnCodec model loading:

复制
FutureWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
Impact:

复制
This warning did not affect oracle reconstruction or metric computation.
Status:

复制
NON-BLOCKING WARNINGS
6.4 Local Baseline Comparison
The local baseline experiments now include:

复制
Copy baseline
Greedy decoding / encodec_small
Top-k20 sampling
EnCodec oracle reconstruction
Comparison table:

Method	Num Files	STOI Mean	PESQ Mean	Role
Copy baseline	644	0.13708409757084303	1.0531656005367729	Simple non-learning lower baseline
Top-k20 sampling	644	0.203996640806126	1.0862229399799561	Sampling-based model generation
Greedy decoding / encodec_small	644	0.3506506825906483	1.1083880492619105	Main model generation
EnCodec oracle reconstruction	644	0.9101189329379848	2.5057239819387473	Codec reconstruction upper-bound reference
6.5 Local Baseline Conclusion
The local baseline experiments were completed successfully.

Main observations:

复制
1. Copy baseline achieved the lowest objective scores:
   STOI = 0.1371, PESQ = 1.0532.

2. Top-k20 sampling improved over the copy baseline:
   STOI = 0.2040, PESQ = 1.0862.

3. Greedy decoding achieved the best score among actual prediction methods:
   STOI = 0.3507, PESQ = 1.1084.

4. EnCodec oracle reconstruction achieved much higher scores:
   STOI = 0.9101, PESQ = 2.5057.
Interpretation:

复制
The copy baseline confirms that simply repeating the previous second of speech is a weak strategy for future speech prediction.

The greedy model clearly outperforms both the copy baseline and top-k20 sampling, indicating that the trained model learned useful future-token prediction patterns beyond direct waveform copying.

The large gap between greedy decoding and EnCodec oracle reconstruction suggests that the main bottleneck is future token prediction quality rather than EnCodec waveform reconstruction quality alone.

The oracle result provides a reasonable upper-bound reference for the current EnCodec-based pipeline.
Current stage status:

复制
6.1 Copy baseline                       PASSED
6.2 EnCodec oracle reconstruction        PASSED
Generated baseline waveforms             PASSED
Saved baseline CSV files                 PASSED
Saved baseline JSON summaries            PASSED
DNSMOS placeholder columns               RECORDED AS NaN
复制

---

## 3. 检查记录是否追加成功

执行：

```bat
powershell -Command "Get-Content generation_status_20260430.md -Tail 180"
4. 报告里可以直接写的一段结论
复制
Two local baselines were implemented to contextualize the model performance. The copy baseline, which directly copies the 1s-2s segment as the predicted 2s-3s future speech, achieved a mean STOI of 0.1371 and a mean PESQ of 1.0532 over 644 utterances. In contrast, the EnCodec oracle reconstruction, which encodes and decodes the ground-truth 0s-3s audio and evaluates the reconstructed 2s-3s segment, achieved a mean STOI of 0.9101 and a mean PESQ of 2.5057. The main greedy decoding model achieved STOI 0.3507 and PESQ 1.1084, outperforming the copy baseline but remaining far below the oracle reconstruction upper-bound.
---

## 7.4 Kaggle Dataset Upload via Kaggle CLI

The Kaggle upload package was uploaded directly from Anaconda Prompt using Kaggle CLI.

Uploaded dataset:

```text
williamjkyoung/speech-prediction-facodec-kaggle-files
Uploaded files:

复制
speech_prediction_project.zip
wsjcam0_subset.zip
Local upload directory:

复制
kaggle_upload
Expected Kaggle input path:

复制
/kaggle/input/speech-prediction-facodec-kaggle-files
Expected files on Kaggle:

复制
/kaggle/input/speech-prediction-facodec-kaggle-files/speech_prediction_project.zip
/kaggle/input/speech-prediction-facodec-kaggle-files/wsjcam0_subset.zip
Status:

复制
KAGGLE DATASET UPLOAD COMPLETED
复制

---

## 10. 小安全提醒

因为 API key 已经在聊天中暴露过，建议上传完成后去 Kaggle：

```text
Account → API → Expire API Token → Create New Token

阶段性成果记录：EnCodec Transformer LM 训练完成 ✅
本阶段目标是在 Kaggle 环境中，基于 WSJCAM0 子集音频数据，完成 EnCodec token 提取，并训练一个 medium 规模的 Transformer 语言模型，用于建模离散音频 token 序列。

截至目前，该阶段已经完整完成，训练流程验证通过，模型 checkpoint 已成功保存。

1. 实验环境与运行平台
本次实验运行于 Kaggle Notebook 环境。

硬件环境
项目	配置
平台	Kaggle
GPU	Tesla T4 × 2
实际训练设备	cuda，主要使用 GPU 0
GPU 0	Tesla T4，Compute Capability 7.5
GPU 1	Tesla T4，Compute Capability 7.5
项目路径
复制
/kaggle/working/speech_prediction_project
数据路径
复制
/kaggle/working/wsjcam0_subset/wavs
2. 数据准备结果
原始音频数据检查通过，共检测到：

复制
wav files: 644
之后按固定随机种子进行 train / valid 划分。

数据划分
数据集	数量
Train wavs	580
Valid wavs	64
Total wavs	644
生成的文件列表
复制
data/kaggle_lists/train_wavs.txt
data/kaggle_lists/valid_wavs.txt
文件路径均验证存在，train / valid 列表数量正确。

3. EnCodec Token 提取结果
使用 EnCodec 24kHz 模型，将 wav 音频转换为离散 token。

Token 提取配置
参数	值
EnCodec bandwidth	6.0 kbps
Sample rate	24000
Channels	1
Device	cuda
Token dtype	torch.int64
Codebooks	8
Vocab size	1024
Token range	0–1023
Train tokens
输出目录：

复制
tokens/encodec/train
提取结果：

复制
Success : 580
Failed  : 0
最终文件数：

复制
Train token files: 580
Valid tokens
输出目录：

复制
tokens/encodec/valid
提取结果：

复制
Success : 64
Failed  : 0
最终文件数：

复制
Valid token files: 64
Token 文件结构示例
.pt 文件内容为字典，包含：

复制
tokens
sample_rate
wav_path
num_codebooks
num_frames
示例 shape：

复制
Train tokens shape: torch.Size([8, 651])
Valid tokens shape: torch.Size([8, 616])
4. 训练前验证结果
在正式训练前，已完成训练脚本和数据完整性检查。

train.py 参数检查
train.py --help 正常，支持以下关键参数：

复制
--train_token_dir
--valid_token_dir
--output_dir
--vocab_size
--seq_len
--batch_size
--epochs
--d_model
--n_heads
--n_layers
--ffn_dim
--lr
--dropout
--weight_decay
--grad_clip
--num_workers
--codebook_idx
--seed
Smoke training 测试
先进行了一个 1 epoch 小模型训练测试。

输出目录：

复制
checkpoints/smoke_train
测试模型配置：

参数	值
seq_len	256
batch_size	4
epochs	1
d_model	128
n_heads	4
n_layers	2
ffn_dim	512
测试结果：

复制
Train loss : 4.817568
Valid loss : 3.621118
Train ppl  : 123.6640
Valid ppl  : 37.3793
并成功生成：

复制
checkpoints/smoke_train/best.pt
checkpoints/smoke_train/last.pt
checkpoints/smoke_train/logs/train_log.csv
说明训练流程、反向传播、验证流程和 checkpoint 保存均正常。

5. Medium 模型正式训练结果
正式训练已完成，对应截图中的 “Kaggle 上运行更大 EnCodec 模型” 阶段。

训练输出目录
复制
checkpoints/encodec_medium
Medium 模型训练配置
参数	值
vocab_size	1024
seq_len	768
batch_size	8
epochs	30
d_model	384
n_heads	6
n_layers	6
ffn_dim	1536
lr	0.0003
dropout	0.1
weight_decay	0.01
grad_clip	1.0
num_workers	2
codebook_idx	0
seed	42
模型参数量：

复制
Model parameters: 11,728,896
训练批次数：

复制
Train batches: 73
Valid batches: 8
6. 最终训练结果
训练共完成：

复制
30 / 30 epochs
训练过程中 checkpoint 正常保存，日志正常记录。

最佳模型结果
最佳验证结果出现在：

复制
Epoch 15
对应指标：

指标	数值
Best epoch	15
Train loss	2.023035
Valid loss	2.157997
Train ppl	7.561237
Valid ppl	8.653789
最终记录：

复制
Best valid loss : 2.157997
Best valid ppl  : 8.653789
最后一轮结果
第 30 epoch：

指标	数值
Train loss	1.563756
Valid loss	2.412188
Train ppl	4.776730
Valid ppl	11.158349
训练后期出现一定过拟合，因此后续使用模型时应优先选择最佳验证集 checkpoint。

7. 已生成的重要文件
最重要模型文件
复制
checkpoints/encodec_medium/best.pt
这是验证集表现最好的模型，后续推理、采样或评估应优先使用它。

最后一轮模型
复制
checkpoints/encodec_medium/last.pt
这是第 30 epoch 后的最终模型，但验证集表现不如 best.pt。

训练日志
复制
checkpoints/encodec_medium/logs/train_log.csv
包含每个 epoch 的：

复制
epoch
train_loss
valid_loss
train_ppl
valid_ppl
best_valid_loss
epoch_time_sec
checkpoint
is_best
训练参数
复制
checkpoints/encodec_medium/args.json
checkpoints/encodec_medium/planned_run_config.json
中间 checkpoint
复制
checkpoints/encodec_medium/checkpoints/epoch_001.pt
...
checkpoints/encodec_medium/checkpoints/epoch_030.pt
每个 checkpoint 大约：

复制
134 MB
best.pt 和 last.pt 大小约：

复制
134.30 MB
8. 阶段性结论
本阶段已经成功完成：

复制
WSJCAM0 subset wav 数据准备
train / valid 数据划分
EnCodec token 提取
训练脚本验证
medium Transformer LM 训练
checkpoint 保存
训练日志检查
最重要的阶段成果是：

复制
checkpoints/encodec_medium/best.pt
该模型在验证集上的最佳结果为：

复制
Valid loss: 2.157997
Valid ppl : 8.653789
可以认为当前已经完成了 Kaggle 上训练 medium EnCodec Transformer LM 的阶段目标。

9. 后续建议记录
后续如果继续实验，可以从以下方向推进：

优先使用 best.pt 做推理或采样

复制
checkpoints/encodec_medium/best.pt
考虑清理中间 checkpoint 节省空间

可保留：

复制
best.pt
last.pt
args.json
logs/train_log.csv
planned_run_config.json
如需继续提升泛化能力

可尝试：

减少训练 epoch，例如 15；
增大 dropout；
加强 weight decay；
使用 early stopping；
扩大训练数据；
尝试多 codebook 建模，而不仅是 codebook_idx 0。
当前模型已出现过拟合趋势

从 epoch 16 开始，valid loss 不再持续下降，后续 valid loss 明显升高。因此：

复制
best.pt > last.pt
在当前任务上更加可靠。

当前阶段状态
复制
阶段：Kaggle medium EnCodec Transformer LM 训练
状态：已完成
最佳模型：checkpoints/encodec_medium/best.pt
最佳 epoch：15
最佳 valid loss：2.157997
最佳 valid ppl：8.653789

7.7 Kaggle 上运行 FACodec Oracle Reconstruction 实验记录
本阶段完成了 FACodec 在 Kaggle 环境下的 oracle reconstruction 流程验证，并额外使用同一批测试音频重新运行了 EnCodec oracle reconstruction 作为对照。实验目标是验证以下流程是否能够稳定运行：

复制
wav -> FACodec encoder -> FACodec tokens -> FACodec decoder -> reconstructed wav
同时，为了便于与已有 EnCodec 系统比较，实验统一在 test_wavs_small16.txt 中的 16 条测试音频上计算 waveform-level 重建指标，包括 MSE、MAE、RMSE、SNR 和 SI-SDR。

1. 实验环境与输入数据
1.1 运行环境
实验运行于 Kaggle Notebook，主要环境信息如下：

项目	配置
Working directory	/kaggle/working/speech_prediction_project
PyTorch	2.10.0+cu128
CUDA	Available
GPU	Tesla T4
FACodec sample rate	16000 Hz
EnCodec sample rate	24000 Hz
EnCodec bandwidth	6.0 kbps
1.2 测试集
本次实验使用小规模测试列表：

复制
/kaggle/working/speech_prediction_project/data/kaggle_lists/test_wavs_small16.txt
共包含：

复制
16 条 wav 文件
示例文件包括：

复制
/kaggle/working/wsjcam0_subset/wavs/0000_440c0201.wav
/kaggle/working/wsjcam0_subset/wavs/0001_440c0202.wav
/kaggle/working/wsjcam0_subset/wavs/0002_440c0203.wav
...
所有文件均成功读取，未发现缺失文件。

2. FACodec Oracle Reconstruction
2.1 FACodec Wrapper 实现
本阶段已创建并验证：

复制
codec/facodec_wrapper.py
该 wrapper 封装了 Amphion NaturalSpeech3 FACodec 的 encoder 和 decoder，主要接口包括：

复制
load_audio()
encode_waveform()
decode_frames()
reconstruct_waveform()
reconstruct_file()
加载的 FACodec 资源路径如下：

复制
Amphion dir:
  /kaggle/working/external/Amphion

FACodec encoder checkpoint:
  /kaggle/working/external/facodec_ckpts/ns3_facodec_encoder.bin

FACodec decoder checkpoint:
  /kaggle/working/external/facodec_ckpts/ns3_facodec_decoder.bin
模型参数规模：

模块	参数量
FACodec Encoder	4,210,528
FACodec Decoder	98,120,652
2.2 FACodec 单条重建 Smoke Test
首先使用测试集第一条音频完成了单条重建测试：

复制
Input wav:
  /kaggle/working/wsjcam0_subset/wavs/0000_440c0201.wav

Output wav:
  /kaggle/working/speech_prediction_project/outputs/facodec_smoke_recon.wav
输入音频信息：

项目	值
Sample rate	16000 Hz
Channels	1
Frames	91600
Duration	5.725 s
Format	WAV PCM_16
重建输出：

项目	值
Sample rate	16000 Hz
Channels	1
Frames	91600
Duration	5.725 s
FACodec 编码 token 信息：

复制
vq_id_shape = [6, 1, 458]
说明 FACodec 成功将原始 waveform 编码为 6 组离散 codebook token，并成功解码回 waveform。

2.3 FACodec Small16 批量重建
随后对 test_wavs_small16.txt 中全部 16 条音频执行 FACodec oracle reconstruction。

输出文件：

复制
Reconstructed wavs:
  /kaggle/working/speech_prediction_project/outputs/facodec_oracle_small16_recons/*.wav

Results CSV:
  /kaggle/working/speech_prediction_project/outputs/facodec_oracle_small16_results.csv

Summary JSON:
  /kaggle/working/speech_prediction_project/outputs/facodec_oracle_small16_summary.json
运行结果：

项目	值
测试音频数	16
成功数	16
失败数	0
总耗时	5.445 s
平均单条耗时	0.329 s
平均 token frames	661.69
FACodec waveform-level 指标均值如下：

指标	FACodec small16
MSE mean	0.000103
MAE mean	0.005827
RMSE mean	0.010026
SNR mean	2.514 dB
SI-SDR mean	-0.197 dB
Ref RMS mean	0.013486
Est RMS mean	0.012606
Ref peak mean	0.169241
Est peak mean	0.154861
FACodec 的 16 条重建均成功生成，说明 FACodec oracle reconstruction 流程已经在 Kaggle 上完整跑通。

3. EnCodec Oracle Reconstruction 对照实验
3.1 修正 EnCodec 重建流程
为了与 FACodec 进行公平对比，本阶段重新对同一批 16 条测试音频运行 EnCodec oracle reconstruction。

初始版本中，使用 wrapper 的 encode_waveform() 和 decode_frames() 时出现结构不匹配问题：

复制
ValueError: too many values to unpack (expected 2)
原因是 EnCodec 原生 model.decode() 需要输入：

复制
encoded_frames = [
    (codes, scale),
    ...
]
而之前 adapter 错误地将 encoded_frames 拆成了纯 codes tensor。

修正后，EnCodec 重建改为直接调用原生接口：

复制
encoded_frames = model.encode(wav)
recon = model.decode(encoded_frames)
修正后 smoke test 成功，第一条音频的 EnCodec 编码信息为：

复制
{
  "encoded_type": "<class 'list'>",
  "num_chunks": 1,
  "num_codebooks": 8,
  "num_frames": 430,
  "encoded_shape": [[1, 8, 430]],
  "scale_shapes": [null],
  "method": "model.encode_model.decode_native"
}
3.2 EnCodec Small16 批量重建结果
EnCodec 对同一批 16 条音频全部重建成功。

输出文件：

复制
Reconstructed wavs:
  /kaggle/working/speech_prediction_project/outputs/encodec_oracle_small16_recons/*.wav

Results CSV:
  /kaggle/working/speech_prediction_project/outputs/encodec_oracle_small16_results.csv

Summary JSON:
  /kaggle/working/speech_prediction_project/outputs/encodec_oracle_small16_summary.json
运行结果：

项目	值
测试音频数	16
成功数	16
失败数	0
总耗时	1.904 s
平均单条耗时	0.111 s
平均 token frames	620.50
Codebooks	8
Bandwidth	6.0 kbps
Sample rate	24000 Hz
EnCodec waveform-level 指标均值如下：

指标	EnCodec small16
MSE mean	0.000084
MAE mean	0.004830
RMSE mean	0.008937
SNR mean	3.618 dB
SI-SDR mean	2.072 dB
Ref RMS mean	0.013467
Est RMS mean	0.013717
Ref peak mean	0.169790
Est peak mean	0.182431
4. FACodec 与 EnCodec 对比结果
重新运行对比脚本后，成功生成统一对比文件：

复制
/kaggle/working/speech_prediction_project/outputs/codec_oracle_comparison_summary.csv
/kaggle/working/speech_prediction_project/outputs/codec_oracle_comparison_summary.json
当前有效对比结果如下：

Codec	Items	Success	MSE ↓	MAE ↓	RMSE ↓	SNR ↑	SI-SDR ↑	Frames	Time / item
FACodec	16	16	0.000103	0.005827	0.010026	2.514 dB	-0.197 dB	661.69	0.329 s
EnCodec	16	16	0.000084	0.004830	0.008937	3.618 dB	2.072 dB	620.50	0.111 s
5. 结果分析
5.1 重建质量对比
在当前 small16 测试集和 waveform-level 指标下，EnCodec 的重建指标优于 FACodec：

EnCodec 的 MSE 更低；
EnCodec 的 MAE / RMSE 更低；
EnCodec 的 SNR 更高；
EnCodec 的 SI-SDR 明显高于 FACodec。
具体来看：

复制
FACodec SI-SDR mean: -0.197 dB
EnCodec SI-SDR mean:  2.072 dB
这说明在当前测试设置下，EnCodec 的 waveform 对齐重建效果更好。

不过需要注意，FACodec 和 EnCodec 的训练目标、采样率、码本设计和感知优化方向不同。FACodec 更偏向 NaturalSpeech3 系统中的语音表示分解和生成用途，而 EnCodec 是通用神经音频 codec。因此，仅使用 MSE / SNR / SI-SDR 并不能完全代表两者在下游语音生成任务中的表现。

5.2 速度对比
在当前 Kaggle Tesla T4 环境下，EnCodec 的推理速度更快：

复制
FACodec 平均单条耗时: 0.329 s
EnCodec 平均单条耗时: 0.111 s
EnCodec 约为 FACodec 的：

复制
0.329 / 0.111 ≈ 2.96 倍速度优势
主要原因可能包括：

FACodec decoder 参数量较大，约 98M；
FACodec 包含 prosody、content、residual 等多组 VQ 表示；
EnCodec 的实现更轻量，且重建路径更直接。
5.3 Token / frame 对比
两者平均帧数接近：

复制
FACodec num_frames mean: 661.69
EnCodec num_frames mean: 620.50
但两者 token 结构不同：

Codec	Token / codebook 结构
FACodec	vq_id_shape = [6, 1, T]，包含 prosody/content/residual 等多组码本
EnCodec	encoded_shape = [1, 8, T]，8 个 RVQ codebooks
因此，虽然两者都有离散 token 序列，但 token 的语义和组织方式并不完全等价。

6. 当前完成状态
本阶段任务完成情况如下：

任务	状态
创建 codec/facodec_wrapper.py	✅ 完成
FACodec 权重加载	✅ 完成
FACodec dummy forward	✅ 完成
FACodec 单条真实音频重建	✅ 完成
FACodec small16 oracle reconstruction	✅ 完成
生成 FACodec CSV / summary	✅ 完成
修正 EnCodec oracle reconstruction	✅ 完成
EnCodec small16 oracle reconstruction	✅ 完成
生成 EnCodec CSV / summary	✅ 完成
FACodec vs EnCodec waveform-level 对比	✅ 完成
STOI / PESQ / DNSMOS	尚未完成
7. 小结
本阶段成功在 Kaggle 环境中跑通了 FACodec oracle reconstruction。实验确认 FACodec 可以完成从 waveform 到离散 token，再从 token 重建 waveform 的完整流程，并成功对 16 条测试语音生成重建音频。

同时，实验使用同一批测试数据重新运行了 EnCodec oracle reconstruction，并修正了 EnCodec encoded frames 结构不匹配的问题。最终得到 FACodec 与 EnCodec 在同一 small16 测试集上的可比结果。

当前 waveform-level 指标显示，EnCodec 在 MSE、MAE、RMSE、SNR、SI-SDR 以及推理速度上均优于 FACodec。但 FACodec 的主要价值可能不完全体现在 waveform 重建误差上，而在于其面向语音生成任务的结构化 token 表示能力。

后续若要严格完成原计划中的评估要求，还需要进一步计算：

复制
STOI
PESQ
DNSMOS