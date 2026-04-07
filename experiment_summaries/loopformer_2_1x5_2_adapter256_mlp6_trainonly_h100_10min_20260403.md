# `2 / 1x5 / 2` Adapter Bank, `MLP_MULT=6`, Train-Only 10-Minute H100 Run

## Goal

Test a larger `2 / 1x5 / 2` loopformer-style recurrent model with a per-loop adapter bank, sized aggressively enough that the eventual compressed artifact was intended to land near the contest budget. This run was launched in `TRAIN_ONLY=1` mode to measure training quality and recurrent usage without paying for final roundtrip eval.

## Run

| Field | Value |
| --- | --- |
| run id | `20260403T185917Z_loopformer_2_1x5_2_adapter256_mlp6_trainonly_wandb_h100_10min_20260403` |
| commit | `538a7b4f86cea47f0741eaeaace8fedcbea6fa00` |
| GPU | `1x NVIDIA H100 80GB HBM3` |
| cloud | `SECURE` |
| local result dir | [raw results](../.runpod/results/20260403T185917Z_loopformer_2_1x5_2_adapter256_mlp6_trainonly_wandb_h100_10min_20260403) |

Config:
- architecture: `2 / 1x5 / 2`
- `RECURRENT_ADAPTER_DIM=256`
- `MLP_MULT=6`
- `TRAIN_ONLY=1`
- `VAL_LOSS_EVERY=0`
- `USE_INT6=0`
- `QAT_ENABLED=0`

## Main Result

This run is not competitive with the better recurrent baselines already in the repo.

Key metrics:

| Metric | Value |
| --- | --- |
| model params | `22,952,488` |
| stop step | `1123` |
| train tokens seen | `588,775,424` |
| last logged train step | `1000` |
| last logged train loss | `2.3742` |
| final validation step | `1123` |
| final validation loss | `2.3230` |
| final validation bpb | `1.3758` |
| avg step time | `534.56 ms` |
| peak memory allocated | `16,401 MiB` |
| elapsed wall time | `706.384 s` |
| raw `final_model.pt` bytes | `90,789,121` |
| raw submission bytes with code | `90,876,685` |

Important caveats:
- This was a `TRAIN_ONLY` run, so there is no quantized artifact size or roundtrip BPB.
- `VAL_LOSS_EVERY=0`, so validation BPB was only measured once at the end.
- There is no train BPB metric in this trainer, only train loss and validation BPB.

## Probe Result

Despite the weak validation result, the recurrence is not dead. Later recurrent passes are active and time-conditioned.

Raw recurrent probe on the saved checkpoint:

| Metric | Loop 1 | Loop 2 | Loop 3 | Loop 4 | Loop 5 |
| --- | --- | --- | --- | --- | --- |
| rel_update_norm | `6.138` | `0.266` | `0.181` | `0.148` | `0.143` |
| cos_prev | `0.283` | `0.972` | `0.987` | `0.991` | `0.991` |
| cos_encoder | `0.283` | `0.280` | `0.272` | `0.264` | `0.256` |
| time_sensitivity | `0.000` | `0.068` | `0.138` | `0.221` | `0.314` |

Interpretation:
- loop 1 makes a large transformation
- later loops are still used, with nonzero update norms through loop 5
- time sensitivity increases with depth, so the loop index signal is being used
- this is clearly not the old identity-collapse failure mode

`adaLN_modulation` norms:

| Tensor | Norm |
| --- | --- |
| `recurrent_blocks.0.adaLN_modulation.1.weight` | `209.04` |
| `recurrent_blocks.0.adaLN_modulation.1.bias` | `5.56` |

So the issue here is not dead recurrence. The model is using the recurrent passes, but the resulting quality is still poor.

## Comparison

The most important comparison is against the cleaner `2 / 1x6 / 2` non-adapter run from April 2:

- [recurrent_adapter_vs_1x6_5min_20260402.md](recurrent_adapter_vs_1x6_5min_20260402.md)

That earlier `1x6` run was only a 5-minute run, but it already looked structurally cleaner:
- lower validation BPB at its budget
- more contractive recurrent rollout
- much smaller model artifact

This new `2 / 1x5 / 2` + large-adapter + wide-MLP run does show active later loops, but the extra capacity does not appear to buy useful quality.

## Operational Notes

- The run trained successfully and saved `final_model.pt`.
- W&B did not log because the remote image did not have `wandb` installed:
  - `wandb:init_failed error=No module named 'wandb'`
- The worker initially marked the run as failed because it expected roundtrip metrics, but that success-path bug has since been fixed for future `TRAIN_ONLY` runs.
- The controller record ended at `delete_failed`, but the pod was already gone; there was no live pod left after completion.
