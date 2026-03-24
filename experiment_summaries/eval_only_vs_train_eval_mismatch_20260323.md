# Eval-Only vs Train-Eval Mismatch

## Problem

An eval-only replay of a previously trained checkpoint does not reproduce the checkpoint's original training-path validation score.

For the same checkpoint and nominally the same strided/doc-separated eval settings:
- original training-path run: `val_bpb = 1.27940328`
- eval-only replay: `val_bpb = 1.87817407`

That gap is far too large to treat as noise.

## Main Question

Why does eval-only scoring differ so much from the normal training-path scoring, even when the loaded checkpoint artifact appears correct?

## Most Important Runs

### Source checkpoint, good training-path result

Run:
- [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323)

Key facts:
- commit: `14f112e090d4868b99b2c2d112dc099198fa5045`
- full normal training path
- `EVAL_STRIDED_ATTN=1`
- `EVAL_DOC_SEPARATED=1`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=1024`
- final roundtrip `val_bpb = 1.27940328`
- final roundtrip `val_loss = 2.16021885`

Important log evidence:
- first logged strided-eval batch: `running_bpb=1.283913`

### Eval-only control, bad result

Run:
- [.runpod/results/20260323T183217Z_eval_seq1024_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T183217Z_eval_seq1024_from_strided_20260323)

Key facts:
- commit: `e44d19fd015d03d1424367f9e36f13cb99091a93`
- eval-only replay of the checkpoint above
- `EVAL_STRIDED_ATTN=1`
- `EVAL_DOC_SEPARATED=1`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=256`
- final roundtrip `val_bpb = 1.87817407`
- final roundtrip `val_loss = 3.17121824`

Important log evidence:
- first logged strided-eval batch: `running_bpb=1.901588`

### Eval-only control with `EVAL_BATCH_SEQS=1024`

Run:
- [.runpod/results/20260323T191007Z_eval_seq1024_batch1024_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T191007Z_eval_seq1024_batch1024_from_strided_20260323)

Key facts:
- same eval-only setup, but `EVAL_BATCH_SEQS=1024`
- final roundtrip `val_bpb = 1.87817407`
- final roundtrip `val_loss = 3.17121824`

Conclusion:
- `EVAL_BATCH_SEQS` is not the cause of the mismatch

### Current-code full training-path sanity rerun

Run:
- [.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323)

Key facts:
- commit: `e44d19fd015d03d1424367f9e36f13cb99091a93`
- full normal training path
- `EVAL_STRIDED_ATTN=1`
- `EVAL_DOC_SEPARATED=1`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=1024`
- final roundtrip `val_bpb = 1.28649006`
- final roundtrip `val_loss = 2.17218458`
- main validation `val_bpb = 1.2854`

Conclusion:
- current code can still produce the expected `~1.28` score range through the normal training path
- the problem is not “current code universally broke strided eval”

## What Was Ruled Out

### 1. Wrong checkpoint file

This was ruled out.

The source checkpoint:
- [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/final_model.pt](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/final_model.pt)

and the eval-only replay output checkpoint:
- [.runpod/results/20260323T183217Z_eval_seq1024_from_strided_20260323/final_model.pt](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T183217Z_eval_seq1024_from_strided_20260323/final_model.pt)

were checked and found to be byte-identical.

The quantized artifacts were also byte-identical:
- source `.ptz`
- eval-only replay `.ptz`

So:
- upload path worked
- `INIT_MODEL_PATH` was correct
- checkpoint loading did not obviously corrupt weights

### 2. Strided helper drift

This was ruled out.

`eval_val_strided(...)` did not materially change between:
- source commit `14f112e...`
- eval-only commit `e44d19f...`

So the mismatch is not explained by a simple regression in the strided-eval helper itself.

### 3. `EVAL_BATCH_SEQS`

This was ruled out.

The eval-only `1024` control produced the same bad result with:
- `EVAL_BATCH_SEQS=256`
- `EVAL_BATCH_SEQS=1024`

### 4. Compile vs eager disagreement on current hosts

This was tested directly on a held H100 pod.

On the same current host, same checkpoint, same first 128 chunks:
- eager bf16 + flash: `val_bpb ~= 1.9162`
- eager bf16 + math: `val_bpb ~= 1.9163`
- compiled bf16 + flash: `val_bpb ~= 1.9162`

So the bad result on current hosts is not explained by:
- `torch.compile` alone
- flash SDPA vs math SDPA alone

### 5. Simple compiled-model state desync

This was tested directly.

On a held H100 pod:
- direct parameter mutation changed both `base_model` and compiled `model` outputs
- `base_model.load_state_dict(...)` reset both back to the same outputs
- a small real optimizer step also kept `base_model` and compiled `model` aligned in the targeted probe

So the simplest “compiled model ignores base-model weights” theory was not confirmed.

## Strongest New Evidence

On a held H100 pod, forcing the same small eval slice to full fp32 changed the score dramatically:

For the first 128 chunks on the current host:
- bf16 math: `val_bpb ~= 1.9163`
- fp32 math: `val_bpb ~= 1.3336`

That is much closer to the original good training-path behavior.

This is the strongest evidence found so far.

## Current Best Interpretation

The mismatch appears to be tied to eval runtime/precision conditions, not to the checkpoint bytes and not to the main strided-eval geometry.

The most likely explanation at this point is:
- current eval-only runs are using a low-precision runtime path that is numerically bad for this model/checkpoint on current H100 hosts
- full training-path runs still land in the expected `~1.28` range, so whatever is “saving” the normal path is not reproduced by naive eval-only replay

What remains unclear is exactly why the full training path and eval-only path diverge if both nominally run bf16 eval. The explanation is still incomplete.

## Concrete Code / Design Issues Found

### Eval-only replay is under-specified

[`load_model_state_dict(...)`](/Users/robertgordan/Projects/parameter-golf/train_gpt.py#L609) restores tensors only. It does not restore or validate non-state hyperparameters such as:
- `ROPE_BASE`
- `LOGIT_SOFTCAP`
- any other architecture-affecting env vars not encoded in the state dict

That means eval-only replay is not fully reproducible by design, even though this did not appear to be the main cause of the current mismatch.

### Eval precision is easy to misunderstand

Both normal validation and eval-only validation run the model forward in bf16 autocast:
- strided eval forward at [`train_gpt.py:323`](/Users/robertgordan/Projects/parameter-golf/train_gpt.py#L323)
- baseline eval forward at [`train_gpt.py:403`](/Users/robertgordan/Projects/parameter-golf/train_gpt.py#L403)

Only the CE/logit reduction is fp32.

This matters because parts of the code/comments imply “higher precision” evaluation after quantization, but today there is no explicit full-fp32 eval mode.

### Roundtrip validation is indirect

[`serialize_and_validate_roundtrip(...)`](/Users/robertgordan/Projects/parameter-golf/train_gpt.py#L626) reloads weights into `base_model`, then validates through `model` at [`train_gpt.py:677`](/Users/robertgordan/Projects/parameter-golf/train_gpt.py#L677).

That usually works, but it is still an indirect validation path and makes debugging harder.

### Run metadata is missing runtime details

The structured Runpod artifacts do not record the host runtime stack in first-class JSON fields:
- driver version
- CUDA runtime version
- effective SDPA backend mode
- eval precision mode

Those details only exist in free-form logs, even though they appear relevant here.

## Useful Log/Environment Clues

The original good run and later bad eval-only runs landed on different host driver stacks:
- good training-path run log: driver `570.195.03`, CUDA `12.8`
- bad eval-only run log: driver `580.126.09`, CUDA `13.0`

This does not prove causality, but it makes runtime-stack sensitivity plausible.

## Cheapest Next Steps

### 1. Add an explicit fp32 eval mode

Add a flag such as:
- `EVAL_FORCE_FP32=1`

Use it only for validation forward passes.

This is the best next test because fp32 already showed the strongest positive signal in the held-pod probe.

### 2. Rerun the eval-only `1024` control in fp32

If it returns to roughly the `~1.28` range, then the main practical issue is precision/runtime, not checkpoint replay.

### 3. If fp32 fixes it, keep fp32 as a serious-comparison eval mode

This would give a reproducible eval path even if it is slower.

### 4. Tighten eval-only reproducibility

At minimum:
- persist source-run architecture/env config with the checkpoint
- validate that replay-time non-state hyperparameters match

## Related Files

Primary summaries:
- [experiment_summaries/eval_path_compare_h100_20260322.md](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/eval_path_compare_h100_20260322.md)
- [experiment_summaries/eval_seq_len_sweep_20260323.md](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/eval_seq_len_sweep_20260323.md)

Visible wrappers:
- [run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323/README.md](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323/README.md)
- [run_results/eval_seq_len_sweep_20260323/seq1024_batch1024/README.md](/Users/robertgordan/Projects/parameter-golf/run_results/eval_seq_len_sweep_20260323/seq1024_batch1024/README.md)

Core code:
- [train_gpt.py](/Users/robertgordan/Projects/parameter-golf/train_gpt.py)
- [ops/runpod/controller.py](/Users/robertgordan/Projects/parameter-golf/ops/runpod/controller.py)
- [ops/runpod/worker.py](/Users/robertgordan/Projects/parameter-golf/ops/runpod/worker.py)
