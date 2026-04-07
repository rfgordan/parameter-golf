# Recurrent Loop Schedule Strided 10-Minute Runs

Goal: compare two train-time loop curricula for the `4 / 3x4 / 4` loopformer-style recurrent model using the correct doc-separated strided eval path.

## Planned runs

1. `loopformer_recur_4_3x4_4_10min_curriculum_strided_20260324`
   - original curriculum: `50% @ 1x`, `25% @ 2x`, `25% @ 4x`
2. `loopformer_recur_4_3x4_4_10min_curriculum801010_strided_20260324`
   - more aggressive curriculum: `80% @ 1x`, `10% @ 2x`, `10% @ 4x`

## Shared setup

- Commit: `f498660bf426ab622c492a36847f327735cc2f4c`
- GPU: `1x NVIDIA H100 80GB HBM3`
- Cloud: `SECURE`
- Validation: doc-separated strided eval
- `VAL_LOSS_EVERY=0`
- Quantization: `int8+zlib`
- Model shape: encoder `4`, recurrent layers `3`, recurrent loops `4`, decoder `4`

## Related prior work

- Flat-eval schedule experiment with accidental eval mismatch: [recurrent_loop_schedule_sampling_10min_20260324.md](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/recurrent_loop_schedule_sampling_10min_20260324.md)
- Fixed-depth recurrent control: [loopformer_quant_compare_20260324.md](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/loopformer_quant_compare_20260324.md)
