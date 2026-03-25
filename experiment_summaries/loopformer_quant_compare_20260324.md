# LoopFormer Quant Compare

## Goal

Compare the `4 / 3x3 / 4` loopformer-style recurrent model under the same `1x H100`, 10-minute training budget with:
- standard `int8` export
- standard `int6` export

Primary questions:
- Does the loopformer-style recurrent model train cleanly under the standard 10-minute recipe?
- How much quantization damage does `int6` introduce relative to `int8` on the same trained architecture?
- How much throughput does this recurrent model give up relative to the earlier non-recurrent strided baseline?

## Shared Setup

- commit: `b80a49baddd031e0d5de7d53aea30935bbf4d7e0`
- branch: `strided-eval-recurse-loopformer-v1`
- GPU: `1x NVIDIA H100 80GB HBM3`
- cloud: `SECURE`
- train shards: `80`
- train token budget per optimizer step: `524,288`
- validation tokens: `62,021,632`
- `VAL_LOSS_EVERY=0`
- eval path: doc-separated strided eval
- architecture:
  - `4` encoder blocks
  - `3` recurrent middle blocks
  - `3` recurrent loops
  - `4` decoder blocks
  - `MLP_MULT=2`
- model params: `24,274,008`
- pods launched with `hold_open=true` so artifacts could be copied down safely before deletion

## Runs

| Run | Quant mode | Run ID | Raw results |
| --- | --- | --- | --- |
| int8 | `USE_INT6=0` | `20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324` | [.runpod/results/20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324) |
| int6 | `USE_INT6=1` | `20260324T232035Z_loopformer_recur_4_3x3_4_int6_h100_10min_20260324` | [.runpod/results/20260324T232035Z_loopformer_recur_4_3x3_4_int6_h100_10min_20260324](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T232035Z_loopformer_recur_4_3x3_4_int6_h100_10min_20260324) |

## Results

| Metric | int8 | int6 |
| --- | --- | --- |
| status | success | success |
| model params | `24,274,008` | `24,274,008` |
| last logged train step | `800` | `800` |
| wallclock stop step | `934` | `929` |
| train tokens seen at stop | `489,684,992` | `487,063,552` |
| last train loss | `2.3224` | `2.3183` |
| main eval val_loss | `2.2671` | `2.2669` |
| main eval val_bpb | `1.3427` | `1.3426` |
| roundtrip val_loss | `2.27089509` | `2.33627894` |
| roundtrip val_bpb | `1.34495199` | `1.38367599` |
| quantization penalty, val_loss | `+0.00379509` | `+0.06937894` |
| quantization penalty, val_bpb | `+0.00225199` | `+0.04107599` |
| final eval time | `527,465 ms` | `524,087 ms` |
| train time to stop | `600,567 ms` | `600,530 ms` |
| average step time at stop | `643.01 ms` | `646.43 ms` |
| elapsed wall time | `1,753.366 s` | `1,742.069 s` |
| peak memory allocated | `19,414 MiB` | `19,414 MiB` |
| peak memory reserved | `19,436 MiB` | `19,436 MiB` |
| compressed model bytes | `12,929,416` | `8,692,883` |
| total submission bytes | `12,995,921` | `8,759,388` |

## Main Takeaways

- Pre-quant performance is effectively identical.
  - `int8` main eval `val_bpb`: `1.3427`
  - `int6` main eval `val_bpb`: `1.3426`
- The meaningful difference appears only after roundtrip quantization.
  - `int8` quant tax: `+0.00225199 val_bpb`
  - `int6` quant tax: `+0.04107599 val_bpb`
- So for this loopformer-style recurrent model, `int6` is materially more destructive than `int8` even though both fit comfortably under the byte budget.

Direct deltas:
- roundtrip `val_bpb`: `int6` is worse than `int8` by `0.03872400`
- compressed model size: `int6` saves `4,236,533` bytes vs `int8`
- total submission size: `int6` saves `4,236,533` bytes vs `int8`

Interpretation:
- This is a clean quantization comparison because training behavior is nearly identical between the two runs.
- `int8` preserves the trained recurrent model well.
- `int6` buys a large size reduction, but the quantization damage is large enough to erase a meaningful fraction of model quality.

## Comparison To Non-Recurrent Strided Baseline

The most relevant earlier non-recurrent baseline is the doc-aware strided no-mid-val run from [eval_path_compare_h100_20260322.md](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/eval_path_compare_h100_20260322.md):
- run id: `20260323T164756Z_eval_path_compare_h100_strided_noval_20260323`
- params: `17,059,912`
- wallclock stop step: `1656`
- train tokens seen at stop: `868,220,928`
- average step time at stop: `362.41 ms`
- main eval `val_bpb`: `1.2781`
- roundtrip `val_bpb`: `1.27940328`

Throughput comparison vs that baseline:

| Metric | Non-recurrent strided baseline | Loopformer int8 | Loopformer int6 |
| --- | --- | --- | --- |
| model params | `17,059,912` | `24,274,008` | `24,274,008` |
| wallclock stop step | `1656` | `934` | `929` |
| train tokens seen at stop | `868,220,928` | `489,684,992` | `487,063,552` |
| avg step time | `362.41 ms` | `643.01 ms` | `646.43 ms` |
| main eval val_bpb | `1.2781` | `1.3427` | `1.3426` |
| roundtrip val_bpb | `1.27940328` | `1.34495199` | `1.38367599` |

Derived comparisons:
- loopformer int8 saw only about `56.4%` as many training tokens as the non-recurrent baseline in the same 10-minute budget
- loopformer int6 saw only about `56.1%` as many training tokens
- loopformer step time is about `1.77x` to `1.78x` slower
- loopformer parameter count is about `1.42x` larger

Interpretation:
- The recurrent model is paying a substantial wallclock-throughput penalty.
- That penalty is large enough that the model sees only a bit over half the training tokens of the non-recurrent baseline in the same contest budget.
- So the disappointing absolute `val_bpb` here is not just a quantization story; it is also a training-efficiency story.

## Conclusion

This experiment gives a clean answer to the quantization question for the current loopformer-style recurrent architecture:

- `int8` is viable and adds only a tiny roundtrip penalty (`+0.00225199 val_bpb`)
- `int6` is much more damaging (`+0.04107599 val_bpb`)
- the model already starts from a weaker absolute score than the earlier non-recurrent baseline because it trains much more slowly within the same 10-minute budget

So the current bottleneck is two-part:
- recurrence still costs too much throughput during training
- and `int6` is too destructive for this architecture in its current form

If the goal is to make this recurrent direction competitive, the next work should probably focus on:
- improving train-time efficiency first
- then making more targeted low-bit quantization choices than uniform `int6`

## Follow-Up: Consistency Loss Off

A follow-up ablation reran the `int8` loopformer model with the same architecture and training setup, but with `CONSISTENCY_LOSS_WEIGHT=0.0` instead of the previous default `0.1`.

Run:
- `20260325T014223Z_loopformer_recur_4_3x3_4_consistency00_int8_h100_10min_rerun_20260325`
- raw results: [.runpod/results/20260325T014223Z_loopformer_recur_4_3x3_4_consistency00_int8_h100_10min_rerun_20260325](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260325T014223Z_loopformer_recur_4_3x3_4_consistency00_int8_h100_10min_rerun_20260325)
- commit: `050cde2431ffaaaa524b9664007268550d55a017`

Comparison against the earlier `int8` run with `CONSISTENCY_LOSS_WEIGHT=0.1`:

| Metric | int8, consistency `0.1` | int8, consistency `0.0` |
| --- | --- | --- |
| wallclock stop step | `934` | `933` |
| train tokens seen at stop | `489,684,992` | `489,160,704` |
| last train loss | `2.3224` | `2.3195` |
| main eval val_loss | `2.2671` | `2.2662` |
| main eval val_bpb | `1.3427` | `1.3422` |
| roundtrip val_loss | `2.27089509` | `2.26969370` |
| roundtrip val_bpb | `1.34495199` | `1.34424046` |
| quantization penalty, val_bpb | `+0.00225199` | `+0.00204046` |
| final eval time | `527,465 ms` | `528,375 ms` |
| average step time at stop | `643.01 ms` | `643.40 ms` |
| elapsed wall time | `1,753.366 s` | `1,756.860 s` |
| peak memory allocated | `19,414 MiB` | `19,414 MiB` |
| compressed model bytes | `12,929,416` | `12,956,510` |
| total submission bytes | `12,995,921` | `13,023,289` |

Observed deltas from turning the consistency loss off:
- main `val_bpb`: `-0.0005`
- roundtrip `val_bpb`: `-0.00071153`
- stop step: effectively unchanged (`934` vs `933`)
- average step time: effectively unchanged (`643.01 ms` vs `643.40 ms`)

Interpretation:
- At this coarse 10-minute scale, the consistency-loss branch does not appear to carry a meaningful throughput penalty.
- Quality also changed only minimally, with the `0.0` run slightly ahead, but by less than a thousandth of a BPB.
- So this ablation is weak evidence that the current shortcut-consistency loss is not doing much, positive or negative, in the present setup.

## Notes

- Both runs were copied down manually from `inspection_hold` and then the pods were deleted to stop spend.
- This experiment did not need additional visible `run_results/` wrappers because the raw result directories are complete and the summary links them directly.

## Follow-Up: Recurrent Block Probe

I later probed the saved loopformer checkpoints directly to answer a more basic question than quantization: are the recurrent blocks doing anything at all on real hidden states?

Probe script:
- [notebooks/recurrent_block_probe.py](/Users/robertgordan/Projects/parameter-golf/notebooks/recurrent_block_probe.py)

Method:
- load a saved `final_model.pt` and matching `final_model.int8.ptz`
- run a real validation batch through the encoder once
- roll the recurrent stack forward loop by loop
- measure:
  - relative update norm `||x_{k+1} - x_k|| / ||x_k||`
  - cosine with previous state
  - cosine with encoder output
  - sensitivity to the timestep embedding
  - raw-vs-quantized state drift at each loop depth

### Probe 1: Fixed-Depth `4 / 3x3 / 4` Loopformer Int8 Run

Artifacts:
- checkpoint run: [`.runpod/results/20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324`](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324)
- probe summary: [summary.json](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324/summary.json)

Key plots:
- [relative update norm](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324/rel_update_norm.png)
- [cosine with previous state](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324/cos_prev.png)
- [time sensitivity](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324/time_sensitivity_rel.png)
- [raw vs quantized drift](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324/raw_quant_rel_diff.png)

Findings:
- raw recurrent rollout:
  - loop 1 `rel_update_norm = 0.0`
  - loop 2 `rel_update_norm = 0.0`
  - loop 3 `rel_update_norm = 0.0`
  - cosine with previous state = `1.0`
  - cosine with encoder output = `1.0`
  - time sensitivity = `0.0`
- quantized recurrent rollout is identical for these metrics
- raw vs quantized drift is about `0.0663` relative norm, but it is constant across loop depth

Interpretation:
- the recurrent stack is exact identity on this checkpoint
- quantization is not causing the recurrence to vanish here; it was already dead before quantization

I also checked the recurrent gate/modulation weights directly:
- every `recurrent_blocks.{0,1,2}.adaLN_modulation.1.weight` norm is `0.0`
- every matching bias norm is `0.0`

That strongly suggests the recurrent path never turned on because the modulation output head stayed exactly zero.

### Probe 2: Curriculum-Trained `4 / 3x4 / 4` Checkpoint

Artifacts:
- checkpoint run: [`.runpod/results/20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324`](/Users/robertgordan/Projects/parameter-golf/.runpod/results/20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324)
- probe summary: [summary.json](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324/summary.json)

Key plots:
- [relative update norm](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324/rel_update_norm.png)
- [cosine with previous state](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324/cos_prev.png)
- [time sensitivity](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324/time_sensitivity_rel.png)
- [raw vs quantized drift](/Users/robertgordan/Projects/parameter-golf/experiment_summaries/assets/recurrent_block_probe_20260325T033817Z_loopformer_recur_4_3x4_4_10min_curriculum_20260324/raw_quant_rel_diff.png)

Findings:
- raw recurrent rollout:
  - loop 1 `rel_update_norm = 0.0`
  - loop 2 `rel_update_norm = 0.0`
  - loop 3 `rel_update_norm = 0.0`
  - loop 4 `rel_update_norm = 0.0`
  - cosine with previous state = `1.0`
  - cosine with encoder output = `1.0`
  - time sensitivity = `0.0`
- quantized recurrent rollout is again identical for these metrics
- raw vs quantized drift is about `0.0535` relative norm, again constant across loop depth

The recurrent gate/modulation weights show the same pattern:
- every `recurrent_blocks.{0,1,2}.adaLN_modulation.1.weight` norm is `0.0`
- every matching bias norm is `0.0`

### Recurrent-Probe Takeaway

This changes the interpretation of the whole loopformer line of experiments:

- the primary problem is not that quantization is killing an otherwise useful recurrent computation
- the primary problem is that the recurrent blocks are already degenerate in the raw checkpoints
- the modulation head staying exactly zero is the clearest proximate cause found so far

So the next useful work is likely upstream of quantization:
- get the recurrent path to produce non-identity updates on real hidden states
- then revisit whether quantization preserves or destroys that learned recurrence
