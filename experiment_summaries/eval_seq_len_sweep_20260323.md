# Eval Sequence Length Sweep

Goal: measure how eval-only `val_bpb` changes for the same trained checkpoint when only `EVAL_SEQ_LEN` changes.

Checkpoint under test:
- [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/final_model.pt](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/final_model.pt)

Shared setup:
- commit `e44d19fd015d03d1424367f9e36f13cb99091a93`
- `EVAL_ONLY=1`
- `EVAL_STRIDED_ATTN=1`
- `EVAL_DOC_SEPARATED=1`
- `EVAL_STRIDE=64`
- `train_shards=0`
- `1x NVIDIA H100 80GB HBM3`

Batching:
- `EVAL_SEQ_LEN=512`, `EVAL_BATCH_SEQS=512`
- `EVAL_SEQ_LEN=1024`, tested at both `EVAL_BATCH_SEQS=256` and `1024`
- `EVAL_SEQ_LEN=2048`, `EVAL_BATCH_SEQS=128`
- `EVAL_SEQ_LEN=4096`, `EVAL_BATCH_SEQS=64`

Prior datapoint already measured:

| Eval Seq Len | Run | Final Roundtrip Val BPB | Notes |
|---|---|---:|---|
| 2048 | [20260323T180659Z_eval_only_ctx2048_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T180659Z_eval_only_ctx2048_from_strided_20260323) | 1.95029362 | Eval-only rerun on the same checkpoint |

Completed runs:

| Eval Seq Len | Run | Final Roundtrip Val BPB | Main Eval BPB | Main Eval Time |
|---|---|---:|---:|---:|
| 512 | [20260323T183700Z_eval_seq512_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T183700Z_eval_seq512_from_strided_20260323) | 1.61511903 | 1.6138 | 397975 ms |
| 1024 | [20260323T183217Z_eval_seq1024_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T183217Z_eval_seq1024_from_strided_20260323) | 1.87817407 | 1.8773 | 271199 ms |
| 1024 | [20260323T191007Z_eval_seq1024_batch1024_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T191007Z_eval_seq1024_batch1024_from_strided_20260323) | 1.87817407 | 1.8773 | 278736 ms |
| 2048 | [20260323T180659Z_eval_only_ctx2048_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T180659Z_eval_only_ctx2048_from_strided_20260323) | 1.95029362 | 1.9492 | 320360 ms |
| 4096 | [20260323T183348Z_eval_seq4096_from_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T183348Z_eval_seq4096_from_strided_20260323) | 1.99395612 | 1.9929 | 412159 ms |

Observed trend within the eval-only pathway:
- `512` is best
- `1024` is worse than `512`
- `2048` is worse than `1024`
- `4096` is worst

Important caveat:
- The same checkpoint’s original training-path final roundtrip score was `1.27940328`, from [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/metrics.json](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/metrics.json).
- The eval-only `1024` control came out at `1.87817407`, which does not reproduce the original `1024` result.
- Re-running that `1024` control with `EVAL_BATCH_SEQS=1024` instead of `256` produced the same `1.87817407` result, so the mismatch is not explained by `EVAL_BATCH_SEQS`.
- So this sweep is internally consistent as an eval-only comparison, but it is not yet a trustworthy apples-to-apples measurement of the original checkpoint under the original scoring path.

Debug follow-up:
- The mismatch appears immediately, not only after full-dataset accumulation. In the original good run, the first logged strided batch was `1.283913`; on current eval-only runs it is `1.901588` for the same first `1024` chunks.
- On a held H100 debug pod, eager bf16 and compiled bf16 produced the same bad first-batch score, and switching from flash SDPA to math SDPA did not help.
- On that same pod, forcing the model forward to full fp32 on the first `128` chunks dropped the score from `1.9163` to `1.3336`.
- That points away from checkpoint loading and toward a low-precision runtime / host-stack issue on current H100 hosts. The strongest current suspect is bf16 eval numerics on the newer driver stack rather than `EVAL_ONLY` logic itself.

Reference comparison:
- original training run for this checkpoint at the default strided/doc-separated eval: [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/metrics.json](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323/metrics.json)
- original final roundtrip `val_bpb`: `1.27940328`
