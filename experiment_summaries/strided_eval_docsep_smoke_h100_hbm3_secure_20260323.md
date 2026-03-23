## Strided Eval Doc-Separated Smoke

Goal: confirm the updated strided eval path runs end to end on remote H100 with document-aware chunking enabled.

Setup:
- GPU: `NVIDIA H100 80GB HBM3`
- Cloud: `SECURE`
- Final tested commit: `14f112e090d4868b99b2c2d112dc099198fa5045`
- Key eval flags: `EVAL_STRIDED_ATTN=1`, `EVAL_DOC_SEPARATED=1`, `EVAL_SEQ_LEN=256`, `EVAL_STRIDE=64`
- Smoke settings: `train_shards=1`, `ITERATIONS=20`, `VAL_LOSS_EVERY=0`

Related prior implementations:
- [records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py)
- [records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md)

Runs:
- `20260323T163642Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323`: invalid smoke run. The payload pinned the wrong full SHA, so the pod checked out `main`. The pod was deleted after remote verification.
- `20260323T163733Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323_v2`: valid smoke run against commit `14f112e090d4868b99b2c2d112dc099198fa5045`. Remote `HEAD` was verified. Training completed, then the new eval path processed document-aware chunks successfully.

Observed result:
- The smoke objective passed: the new eval code executed on H100 with the intended commit and entered the document-aware `sliding_eval` loop without crashing.
- Live eval progress reached at least `294464 / 847347` chunks (`34.8%`) with stable running BPB values around `3.16`.
- No terminal metrics were captured because the pod was stopped early once the new eval path had been validated.

Visible run wrapper:
- [run_results/strided_eval_docsep_smoke_h100_hbm3_secure_20260323/README.md](/Users/robertgordan/Projects/parameter-golf/run_results/strided_eval_docsep_smoke_h100_hbm3_secure_20260323/README.md)
