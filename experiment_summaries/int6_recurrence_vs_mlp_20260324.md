# Int6 Recurrence Quantization Degradation

## Goal

Measure quantization degradation on the exact same recurrent architecture, and test whether fp16 passthrough for recurrent block weights reduces that degradation.

Both runs use the same architecture:

- `4/3x3/4`
- `MLP_MULT=3`

Both runs also use:

- commit `f7c3c6d57b034f6a6b581f9e61e754ef19a75b8e`
- `1x NVIDIA H100 80GB HBM3`
- `MAX_WALLCLOCK_SECONDS=600`
- doc-separated strided eval
- `USE_INT6=1`

## Runs

| run | architecture | key export setting | notes |
| --- | --- | --- | --- |
| `int6_recur_4_3x3_4_mlp3_baseline_h100_10min_20260324` | `4/3x3/4`, `MLP_MULT=3`, `NUM_RECURRENT_LOOPS=3` | `FP16_PASSTHROUGH_PATTERNS=tok_emb` | baseline int6 export with only the tied embedding protected |
| `int6_recur_4_3x3_4_mlp3_fp16recur_h100_10min_20260324` | `4/3x3/4`, `MLP_MULT=3`, `NUM_RECURRENT_LOOPS=3` | `FP16_PASSTHROUGH_PATTERNS=tok_emb,recurrent_blocks` | same model, but recurrent block tensors also kept in fp16 passthrough |

## Results

## Shared Setup

- GPU: `1x NVIDIA H100 80GB HBM3`
- cloud: `SECURE`
- commit: `f7c3c6d57b034f6a6b581f9e61e754ef19a75b8e`
- training budget: `MAX_WALLCLOCK_SECONDS=600`
- data:
  - `DATA_PATH=./data/datasets/fineweb10B_sp1024/`
  - `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
  - `VOCAB_SIZE=1024`
- eval path:
  - `EVAL_STRIDED_ATTN=1`
  - `EVAL_DOC_SEPARATED=1`
- quantization:
  - `USE_INT6=1`
- architecture:
  - `NUM_ENCODER_LAYERS=4`
  - `NUM_RECURRENT_LAYERS=3`
  - `NUM_RECURRENT_LOOPS=3`
  - `NUM_DECODER_LAYERS=4`
  - `MLP_MULT=3`

Both runs trained the same model with `26,505,816` parameters. The only difference was which tensors were protected from int6 quantization at export time.

## Main Result

Protecting `recurrent_blocks` with fp16 passthrough reduced the post-quant degradation, but only partially:

- baseline int6 export penalty: `+0.04920890 val_bpb`
- fp16 recurrent passthrough penalty: `+0.03386294 val_bpb`
- recovered improvement: `0.01534596 val_bpb`

The cost was severe:

- baseline total submission: `12,244,826` bytes
- fp16 recurrent passthrough total submission: `22,341,848` bytes
- size increase: `+10,097,022` bytes

So the experiment supports the hypothesis that recurrence is unusually quantization-sensitive, but naive fp16 passthrough of the recurrent stack is not remotely budget-compatible.

## Non-Recurrent Control

A follow-up control run used a non-recurrent model on the same branch with the same int6 export path:

- run id: `20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324`
- architecture: `4/0/5`, `MLP_MULT=3`
- parameter count: about `21,778,504`
- fp16 passthrough: `tok_emb`
- recovered final roundtrip `val_loss`: `2.15506004`
- recovered final roundtrip `val_bpb`: `1.27634794`
- recovered compressed model file size: `13,802,789` bytes

This result was only partially salvaged because the pod was stopped before `scp` completed, so the pre-quant main eval line and `metrics.json` were lost. That means the exact quantization tax for the non-recurrent control is not known.

Still, the post-quant score alone is informative:

- non-recurrent int6 post-quant `val_bpb`: `1.27634794`
- recurrent int6 baseline post-quant `val_bpb`: `1.35800890`
- recurrent int6 + fp16 recurrent passthrough post-quant `val_bpb`: `1.34106294`

So even with the caveat, the non-recurrent model appears substantially more robust to this simple int6 export path than either recurrent variant.

Estimated non-recurrent quant tax:

- This is an inference, not a measured value.
- If the non-recurrent pre-quant score was in the low `1.23` to `1.25` range, which is plausible given earlier 1x H100 non-recurrent strided baselines on nearby code, the quantization tax would be roughly `+0.03` to `+0.05` BPB.
- That estimated range is materially smaller than the measured recurrent penalties of `+0.04920890` and `+0.03386294`.

## Detailed Metrics

| Metric | Int6 baseline | Int6 + fp16 recurrent passthrough |
| --- | --- | --- |
| run id | `20260324T165236Z_int6_recur_4_3x3_4_mlp3_baseline_h100_10min_20260324` | `20260324T165315Z_int6_recur_4_3x3_4_mlp3_fp16recur_h100_10min_20260324` |
| fp16 passthrough patterns | `tok_emb` | `tok_emb,recurrent_blocks` |
| params | `26,505,816` | `26,505,816` |
| last train step | `800` | `800` |
| last train loss | `2.2444` | `2.2435` |
| main eval step | `852` | `858` |
| main eval loss | `2.2098` | `2.2072` |
| main eval bpb | `1.3088` | `1.3072` |
| roundtrip eval loss | `2.29294114` | `2.26432860` |
| roundtrip eval bpb | `1.35800890` | `1.34106294` |
| quantization loss penalty | `+0.08314114` | `+0.05712860` |
| quantization bpb penalty | `+0.04920890` | `+0.03386294` |
| compressed model bytes | `12,180,579` | `22,277,601` |
| total submission bytes | `12,244,826` | `22,341,848` |
| elapsed wall time | `1985.899 s` | `1963.020 s` |
| train time to main eval | `600368 ms` | `600648 ms` |
| average step time at main eval | `704.66 ms` | `700.06 ms` |

## Interpretation

What this tells us:

- The degradation is not just a size-budget artifact. Even when the recurrent stack is kept at higher precision, a significant roundtrip penalty remains.
- The recurrent stack is still an important contributor to the problem. Protecting it recovered about `31%` of the original BPB penalty (`0.01535 / 0.04921`).
- The rest of the quantized model still contributes substantial error, or the recurrent architecture amplifies errors introduced elsewhere in the network.

The clean takeaway is:

- recurrence is genuinely quantization-sensitive
- recurrent fp16 passthrough helps
- but this naive mitigation is far too expensive in bytes to be a usable contest solution
- a non-recurrent control appears much more robust to the same simple int6 export, though its exact quantization tax remains estimated rather than measured

## Raw Results

- baseline raw results: [.runpod/results/20260324T165236Z_int6_recur_4_3x3_4_mlp3_baseline_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T165236Z_int6_recur_4_3x3_4_mlp3_baseline_h100_10min_20260324)
- fp16 recurrent passthrough raw results: [.runpod/results/20260324T165315Z_int6_recur_4_3x3_4_mlp3_fp16recur_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T165315Z_int6_recur_4_3x3_4_mlp3_fp16recur_h100_10min_20260324)
- non-recurrent partial result dir: [.runpod/results/20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf-run-int6-nonrec-wt/.runpod/results/20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324)

## Notes

- The first launch attempt for this experiment failed immediately because the payload omitted the standard dataset/tokenizer env vars. Those failed runs are not part of the comparison.
- The final fp16-recurrent run ended with controller status `delete_failed`, but the result bundle is present locally and `runpodctl pod list` was empty afterward, so there is no live pod left over.
- The non-recurrent control run finished successfully, but its artifact salvage was incomplete because `scp` and pod shutdown overlapped.
