# Train Sequence Length Headroom

## Goal

Measure the score headroom available from longer training-time context while keeping the total number of training tokens fixed.

## Setup

Shared settings:
- commit: `e44d19fd015d03d1424367f9e36f13cb99091a93`
- GPU: `1x NVIDIA H100 80GB HBM3`
- cloud: `SECURE`
- model params: `17,059,912`
- tokenizer / vocab: SentencePiece, `VOCAB_SIZE=1024`
- validation tokens: `62,021,632`
- fixed token budget:
  - `TRAIN_BATCH_TOKENS=524288`
  - `ITERATIONS=1535`
  - total processed training tokens: `804782080`
- no wallclock stop:
  - `MAX_WALLCLOCK_SECONDS=0`
- doc-aware strided eval:
  - `EVAL_STRIDED_ATTN=1`
  - `EVAL_DOC_SEPARATED=1`
  - `EVAL_STRIDE=64`
- validation cadence:
  - `VAL_LOSS_EVERY=0`

Reference `1024` run already available:
- [20260323T194305Z_full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323)
- final roundtrip `val_bpb = 1.28649006`

Completed new runs:
- `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`, `EVAL_BATCH_SEQS=128`
- `TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`, `EVAL_BATCH_SEQS=64`

Decision on `8192`:
- not launched initially
- after seeing `2048` and `4096` complete in `22.5m` and `29.6m`, respectively, `8192` looked likely to stay within `~1 hour` on `1x H100`
- `TRAIN_SEQ_LEN=8192` was then launched with `EVAL_BATCH_SEQS=32` and completed successfully on the second attempt

## Result So Far

The sweep shows modest but real headroom from longer training-time context at constant token budget, with the best result at `4096`. `2048` is flat versus `1024`, `4096` improves slightly, and `8192` gives back some of that gain while remaining better than `1024`.

| Seq Len | Final Roundtrip `val_bpb` | Delta vs 1024 | Final Eval Time | Elapsed Wall Time |
| --- | --- | --- | --- | --- |
| 1024 | `1.28649006` | baseline | `279916 ms` | `1213.903 s` |
| 2048 | `1.28654086` | `+0.00005080` | `324633 ms` | `1352.226 s` |
| 4096 | `1.28450175` | `-0.00198831` | `413596 ms` | `1773.459 s` |
| 8192 | `1.28524499` | `-0.00124507` | `614844 ms` | `2449.725 s` |

## Metrics

Shared experiment constants:
- model parameters: `17,059,912`
- validation tokens: `62,021,632`
- raw serialized model size before compression: `67,224,983` bytes
- code size: `59,981` bytes
- raw total submission size before compression: `67,284,964` bytes
- training batch tokens per step: `524,288`
- target training steps: `1,535`
- train tokens seen per completed run: `804,782,080`
- eval mode: doc-separated strided eval with `EVAL_STRIDE=64`

Per-run metrics:

| Seq Len | Train Step / Limit | Train Tokens Seen | Last Train Loss | Main Eval Loss | Main Eval BPB | Roundtrip Eval Loss | Roundtrip Eval BPB | Roundtrip Delta vs Main | Final Eval Time | Train Time to Main Eval | Avg Step Time | Elapsed Wall Time | Peak Mem Alloc / Reserved | Compressed Model Bytes | Compressed Submission Bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | `1535 / 20000` | `804,782,080` | `2.2864` | `2.1704` | `1.2854` | `2.17218458` | `1.28649006` | `+0.00109006` | `279916 ms` | `600262 ms` | `391.05 ms` | `1213.903 s` | `20715 / 35416 MiB` | `14,103,019` | `14,163,000` |
| 2048 | `1535 / 1535` | `804,782,080` | `2.2569` | `2.1702` | `1.2853` | `2.17227035` | `1.28654086` | `+0.00124086` | `324633 ms` | `641240 ms` | `417.75 ms` | `1352.226 s` | `10748 / 17496 MiB` | `14,051,345` | `14,111,326` |
| 4096 | `1535 / 1535` | `804,782,080` | `2.2413` | `2.1669` | `1.2834` | `2.16882740` | `1.28450175` | `+0.00110175` | `413596 ms` | `843371 ms` | `549.43 ms` | `1773.459 s` | `10750 / 17500 MiB` | `14,042,489` | `14,102,470` |
| 8192 | `1535 / 1535` | `804,782,080` | `2.2359` | `2.1680` | `1.2840` | `2.17008233` | `1.28524499` | `+0.00124499` | `614844 ms` | `1150444 ms` | `749.47 ms` | `2449.725 s` | `10754 / 17504 MiB` | `14,028,210` | `14,088,191` |

Notes on the table:
- `1024` used the same effective token budget but stopped on the standard `600s` wallclock cap rather than an explicit fixed-step cap; it still reached step `1535`, so its processed token count matches the longer-context runs.
- “Main Eval” is the validation score on the trained fp32/bf16 runtime model before quantization.
- “Roundtrip Eval” is the final int8+zlib serialize/dequantize/validate score.
- “Roundtrip Delta vs Main” captures the quantization-and-roundtrip penalty.
- Compressed sizes are the actual submission-relevant sizes; raw serialized sizes are shared because the uncompressed state dict shape is unchanged across runs.

## Interpretation

At this model size and fixed token budget, longer context does not unlock a large score jump. The improvement from `1024` to `4096` is about `0.002 val_bpb`, and pushing further to `8192` does not continue the trend.

That suggests:
- there is some real headroom from longer training-time context
- the headroom is small relative to the cost increase
- `4096` looks like the best tradeoff in this sweep
- naive extension all the way to validation-doc `p99` scale is not obviously worth it on score alone

Secondary observations:
- step time grows materially with sequence length: `391 ms -> 418 ms -> 549 ms -> 749 ms`
- final eval time also grows materially: `280 s -> 325 s -> 414 s -> 615 s`
- compressed artifact size improves monotonically with longer context in this run set, but only by about `75 KB` from `1024` to `8192`
- the quantization roundtrip penalty stays small and fairly stable at roughly `0.0011-0.00125 val_bpb`

## Runs

| Seq Len | Run ID | Status | Raw results | Visible results |
| --- | --- | --- | --- | --- |
| 1024 | `20260323T194305Z_full_train_sanity_strided_20260323` | complete | [.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323) | [run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323) |
| 2048 | `20260324T013029Z_train_seq2048_headroom_20260323` | complete | [.runpod/results/20260324T013029Z_train_seq2048_headroom_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T013029Z_train_seq2048_headroom_20260323) | [run_results/train_seq_len_headroom_20260323/seq2048](/Users/robertgordan/Projects/parameter-golf/run_results/train_seq_len_headroom_20260323/seq2048) |
| 4096 | `20260324T013049Z_train_seq4096_headroom_20260323` | complete | [.runpod/results/20260324T013049Z_train_seq4096_headroom_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T013049Z_train_seq4096_headroom_20260323) | [run_results/train_seq_len_headroom_20260323/seq4096](/Users/robertgordan/Projects/parameter-golf/run_results/train_seq_len_headroom_20260323/seq4096) |
| 8192 | `20260324T025158Z_train_seq8192_headroom_20260323` | complete | [.runpod/results/20260324T025158Z_train_seq8192_headroom_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T025158Z_train_seq8192_headroom_20260323) | [run_results/train_seq_len_headroom_20260323/seq8192](/Users/robertgordan/Projects/parameter-golf/run_results/train_seq_len_headroom_20260323/seq8192) |

Retry note:
- first `8192` attempt `20260324T025013Z_train_seq8192_headroom_20260323` never reached SSH readiness and was deleted
- second attempt `20260324T025158Z_train_seq8192_headroom_20260323` completed successfully
