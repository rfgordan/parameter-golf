# LoopFormer Consistency Compare

Goal: compare the `4 / 3x3 / 4` loopformer-style recurrent model with `CONSISTENCY_LOSS_WEIGHT=0.0` against the already-completed `CONSISTENCY_LOSS_WEIGHT=0.1` `int8` run from the previous quantization experiment.

Shared setup:
- commit: `050cde2a01b86d856819d0933cbdb4aa9ad82b26`
- branch: `strided-eval-recurse-loopformer-v1`
- GPU: `1x NVIDIA H100 80GB HBM3`
- cloud: `SECURE`
- train shards: `80`
- `VAL_LOSS_EVERY=0`
- doc-separated strided eval enabled
- architecture: `4 encoder / 3 recurrent blocks x 3 loops / 4 decoder`
- `MLP_MULT=2`
- export: `int8`

Reference run from previous experiment:
- `loopformer_recur_4_3x3_4_int8_h100_10min_20260324`
  - run id: `20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324`
  - `CONSISTENCY_LOSS_WEIGHT=0.1`

New run:
- `loopformer_recur_4_3x3_4_consistency00_int8_h100_10min_20260324`
  - run id: `20260325T012955Z_loopformer_recur_4_3x3_4_consistency00_int8_h100_10min_20260324`
  - `CONSISTENCY_LOSS_WEIGHT=0.0`

To fill after completion:
- train steps / tokens seen
- train loss at matched checkpoints
- main eval loss / BPB
- roundtrip eval loss / BPB
- quantization penalty
- final eval time
- elapsed wall time
- average step time
- compressed artifact sizes

Primary questions:
- Does the consistency loss improve quality at a fixed 10-minute budget?
- Does turning it off materially change throughput or stop-step count?
- Is any train-time overhead from the consistency branch visible at this coarse wallclock scale?
