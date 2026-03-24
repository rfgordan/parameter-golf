# Eval Path Compare H100 2026-03-22

## Goal

Directly compare the baseline validation path against the new sliding-window / strided evaluation path on the same code commit and the same 10-minute H100 training budget.

Primary questions:
- Does the new eval path complete successfully under the standard 10-minute training recipe?
- How much does `val_bpb` change versus baseline?
- How much slower is evaluation, especially the final post-quant roundtrip eval?

## Setup

Shared settings:
- GPU: `NVIDIA H100 80GB HBM3`
- Cloud: `SECURE`
- Training budget: `MAX_WALLCLOCK_SECONDS=600`
- Dataset/tokenizer:
  - `DATA_PATH=./data/datasets/fineweb10B_sp1024/`
  - `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
  - `VOCAB_SIZE=1024`

Run pair 1, original comparison:
- Commit: `60b9138ebe68f89376aed49136c1b132e5362b55`
- `VAL_LOSS_EVERY=1000`
- Baseline path: flat `eval_val`
- Strided path: early flat-stream `eval_val_strided`

Run pair 2, cleaner rerun:
- Commit: `14f112e090d4868b99b2c2d112dc099198fa5045`
- `VAL_LOSS_EVERY=0`
- Baseline path: flat `eval_val`
- Strided path: doc-aware `eval_val_strided` with `EVAL_DOC_SEPARATED=1`
- `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`, `EVAL_BATCH_SEQS=1024`

## Runs

| Run | Eval path | Run ID | Status | Raw results | Visible results |
| --- | --- | --- | --- | --- | --- |
| baseline | non-strided `eval_val` | `20260322T230118Z_eval_path_compare_h100_baseline_20260322` | success | [.runpod/results/20260322T230118Z_eval_path_compare_h100_baseline_20260322](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260322T230118Z_eval_path_compare_h100_baseline_20260322) | [run_results/eval_path_compare_h100_20260322/baseline](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/baseline) |
| strided | `eval_val_strided` | `20260322T230203Z_eval_path_compare_h100_strided_20260322` | success | [.runpod/results/20260322T230203Z_eval_path_compare_h100_strided_20260322](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260322T230203Z_eval_path_compare_h100_strided_20260322) | [run_results/eval_path_compare_h100_20260322/strided](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/strided) |
| baseline no-mid-val | non-strided `eval_val` | `20260323T164728Z_eval_path_compare_h100_baseline_noval_20260323` | success, partial local copy | [.runpod/results/20260323T164728Z_eval_path_compare_h100_baseline_noval_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164728Z_eval_path_compare_h100_baseline_noval_20260323) | [run_results/eval_path_compare_h100_20260322/baseline_noval_20260323](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/baseline_noval_20260323) |
| strided no-mid-val | doc-aware `eval_val_strided` | `20260323T164756Z_eval_path_compare_h100_strided_noval_20260323` | success | [.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T164756Z_eval_path_compare_h100_strided_noval_20260323) | [run_results/eval_path_compare_h100_20260322/strided_noval_20260323](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/strided_noval_20260323) |
| strided sanity rerun | doc-aware `eval_val_strided` | `20260323T194305Z_full_train_sanity_strided_20260323` | success, partial local copy after controller egress failure | [.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260323T194305Z_full_train_sanity_strided_20260323) | [run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323](/Users/robertgordan/Projects/parameter-golf/run_results/eval_path_compare_h100_20260322/full_train_sanity_strided_20260323) |

## Metrics To Capture

For each run, record:
- final roundtrip `val_loss`
- final roundtrip `val_bpb`
- artifact size
- full elapsed runtime
- final eval time from trainer logs
- whether the Pod cleaned up successfully

## Primary Result

The cleaner rerun with `VAL_LOSS_EVERY=0` is the main result for this experiment. It removes repeated full-validation passes during training, so the measured slowdown reflects the final eval path much more directly.

| Metric | Baseline no-mid-val | Strided no-mid-val |
| --- | --- | --- |
| commit | `14f112e090d4868b99b2c2d112dc099198fa5045` | `14f112e090d4868b99b2c2d112dc099198fa5045` |
| final roundtrip val_loss | `2.22880086` | `2.16021885` |
| final roundtrip val_bpb | `1.32002141` | `1.27940328` |
| compressed model bytes | `13972827` | `14432880` |
| total submission bytes | `14029251` | `14489304` |
| final eval time | `10810ms` | `283660ms` |
| wallclock stop step | `1557` | `1656` |
| pod cleanup | yes | yes |

Direct deltas:
- `val_bpb`: `-0.04061813` in favor of strided eval
- `val_loss`: `-0.06858201` in favor of strided eval
- `val_bpb` relative improvement: about `3.08%`
- final eval slowdown: about `26.24x`

Interpretation:
- Removing mid-train validation substantially reduced the practical cost of the strided path relative to the earlier run.
- The strided path still improves score by about the same amount as before.
- The eval-time penalty remains large, but it is much smaller than the earlier `49x` slowdown once repeated full-validation passes are removed.

## Supporting Result

The earlier run pair is still useful as a reference point because it shows how much intermediate validation inflated the practical runtime cost.

| Metric | Baseline original | Strided original |
| --- | --- | --- |
| commit | `60b9138ebe68f89376aed49136c1b132e5362b55` | `60b9138ebe68f89376aed49136c1b132e5362b55` |
| final roundtrip val_loss | `2.27669004` | `2.20541013` |
| final roundtrip val_bpb | `1.34838408` | `1.30616810` |
| compressed model bytes | `12,909,961` | `13,133,948` |
| total submission bytes | `12,963,972` | `13,187,959` |
| worker elapsed seconds | `758.135` | `2813.045` |
| final eval time | `10,930 ms` | `540,423 ms` |
| wallclock stop step | `1159` | `1260` |
| pod cleanup | yes | yes |

Direct deltas:
- `val_bpb`: `-0.04221598` in favor of strided eval
- `val_loss`: `-0.07127991` in favor of strided eval
- `val_bpb` relative improvement: about `3.13%`
- final eval slowdown: about `49.44x`

## Conclusion

The `VAL_LOSS_EVERY=0` rerun is the right apples-to-apples answer for this experiment.

On that cleaner run pair:
- strided eval improved final `val_bpb` by `0.04061813`
- strided eval improved final `val_loss` by `0.06858201`
- final roundtrip eval was `26.24x` slower than baseline

The older run pair shows the same score direction, but with a much larger end-to-end penalty because repeated full-validation passes during training magnified the cost of the strided path.

So the main conclusion is stable:
- the strided/doc-aware eval path materially improves reported score
- most of the practical pain comes from eval cost, not training instability
- if this metric is used for serious comparisons, `VAL_LOSS_EVERY=0` is the cleaner way to benchmark it

## Sanity Rerun On Current Code

A follow-up sanity rerun on the current eval-only-debug branch checked whether the normal full training path still lands in the expected score range.

| Metric | Current-code strided sanity rerun |
| --- | --- |
| commit | `e44d19fd015d03d1424367f9e36f13cb99091a93` |
| final roundtrip val_loss | `2.17218458` |
| final roundtrip val_bpb | `1.28649006` |
| main val_bpb | `1.2854` |
| compressed model bytes | `14103019` |
| total submission bytes | `14163000` |
| final eval time | `279916ms` |
| wallclock stop step | `1535` |
| pod cleanup | yes |

Interpretation:
- The current-code full training path behaves normally and stays in the same `~1.28` range as the earlier good strided training-path run (`1.27940328`).
- It does not reproduce the bad eval-only `1024` control (`1.87817407`).
- That makes the remaining bug much more specific: it is tied to the eval-only pathway or its runtime conditions, not to the ordinary training-path validation logic.

## Relevant Prior Work

Related Parameter Golf records to cite when results are in:
- [records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md)
- [records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md)
- [records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md)
- [records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md](/Users/robertgordan/Projects/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md)

## Notes

- This experiment isolates eval-path differences only; both runs should share the same training settings.
- The controller records for these two runs may lag behind the manually salvaged artifacts because the remote run directories were copied down directly before pod deletion.
- The `20260323T164728Z_eval_path_compare_h100_baseline_noval_20260323` local artifact copy is partial because the finished pod was deleted before the copy completed, but the copied `stdout.log` still contains the terminal metrics used above.
