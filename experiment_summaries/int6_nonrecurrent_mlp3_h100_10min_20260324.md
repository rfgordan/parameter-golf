# Int6 Non-Recurrent MLP3

## Goal

Run a non-recurrent `4/0/5`, `MLP_MULT=3` baseline on `1x H100` with the same 10-minute budget and the same int6 export path used in the recurrence experiments.

## Setup

- commit: `f7c3c6d57b034f6a6b581f9e61e754ef19a75b8e`
- GPU: `1x NVIDIA H100 80GB HBM3`
- cloud: `SECURE`
- training budget: `MAX_WALLCLOCK_SECONDS=600`
- quantization: `USE_INT6=1`
- fp16 passthrough: `FP16_PASSTHROUGH_PATTERNS=tok_emb`
- architecture:
  - `NUM_ENCODER_LAYERS=4`
  - `NUM_RECURRENT_LAYERS=0`
  - `NUM_RECURRENT_LOOPS=0`
  - `NUM_DECODER_LAYERS=5`
  - `MLP_MULT=3`

Expected parameter count: about `21,778,504`.

## Result

Recovered final result only; artifact salvage was incomplete.

## Recovered Metrics

- run id: `20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324`
- pod: cleaned up
- final roundtrip `val_loss`: `2.15506004`
- final roundtrip `val_bpb`: `1.27634794`
- remote compressed model file size: `13,802,789` bytes

## What Was Lost

The pod was stopped before `scp` finished, so the local artifact bundle is incomplete. I did not recover:

- `metrics.json`
- full `stdout.log`
- the pre-quant main eval line
- total submission bytes

So the exact quantization tax for this run is not recoverable from local artifacts.

## Interpretation

Even with incomplete salvage, the key result is still useful:

- this non-recurrent `4/0/5`, `MLP_MULT=3`, int6 run reached `1.27634794` post-quant `val_bpb`
- that is much better than the recurrent int6 runs from the same branch:
  - recurrent int6 baseline: `1.35800890`
  - recurrent int6 + fp16 recurrent passthrough: `1.34106294`

So the non-recurrent model appears substantially more robust to this simple int6 export path than the recurrent model.

## Raw Results

- local result dir: [.runpod/results/20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf-run-int6-nonrec-wt/.runpod/results/20260324T180118Z_int6_nonrecurrent_4_0_5_mlp3_h100_10min_20260324)

## Notes

- The run itself finished successfully on the pod.
- The cleanup mistake was operational: `scp` and pod shutdown overlapped, and the SSH session closed before the copy completed.
