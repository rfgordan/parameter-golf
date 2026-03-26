# Recurrent Block Probe: Gate-Init Fix Activates Recurrence, Reveals Quantization Drift

## Goal

After previous probes found that the loopformer recurrent blocks were collapsing to exact identity (modulation weights stuck at zero), a gate initialization fix was applied. This experiment verifies:

1. Are the recurrent blocks now actually learning non-identity transformations?
2. How does recurrent activity evolve with more training?
3. Does int8 quantization preserve the learned recurrence, or does error compound across loops?

## Context

Previous probes (documented in [loopformer_quant_compare_20260324.md](loopformer_quant_compare_20260324.md)) found:
- Every `recurrent_blocks.{0,1,2}.adaLN_modulation.1.weight` norm was `0.0`
- `rel_update_norm = 0.0` at all loops, `cos_prev = 1.0`, `time_sensitivity = 0.0`
- The recurrent blocks were exact identity — dead on arrival

The gate-init fix (commit `8bbf505`) seeds the modulation gates to non-zero values so the recurrent path is active from the start.

## Runs

Both runs use the `4 / 3x3 / 4` loopformer on branch `strided-eval-recurse-loopformer-v1`, commit `8bbf5054d052aff88b21f94dedfe68ae03126831`.

| Run | Training budget | Steps completed | Run ID |
| --- | --- | --- | --- |
| Short | 50 iterations | 50 | `20260326T005037Z_smoketest_loopformer_20260325` |
| 2-min | 120s wallclock | 186 | `20260326T012005Z_smoketest_loopformer_2min_20260325` |

Shared config:
- GPU: `1x NVIDIA H100 80GB HBM3` (SECURE)
- Architecture: `4` encoder / `3` recurrent mid / `3` loops / `4` decoder
- `train_shards=1`, `VAL_LOSS_EVERY=0`, no strided eval
- model params: `24,274,008`

Raw results:
- [50-step results](../.runpod/results/20260326T005037Z_smoketest_loopformer_20260325)
- [2-min results](../.runpod/results/20260326T012005Z_smoketest_loopformer_2min_20260325)

## Training Metrics

| Metric | 50 steps | 186 steps |
| --- | --- | --- |
| last train loss | `6.070` (step 10) | `3.399` (step 150) |
| final val loss | `4.040` | `3.307` |
| final val bpb | `2.393` | `1.958` |
| int8 roundtrip val bpb | `2.397` | `2.267` |
| **quantization penalty (bpb)** | **`+0.004`** | **`+0.309`** |
| int8+zlib model bytes | `13,458,521` | `10,637,311` |
| elapsed seconds | `220` | `322` |
| step avg | `666 ms` | `~645 ms` |

## Finding 1: Recurrent Blocks Are Now Active

The gate-init fix works. The recurrent blocks are producing non-identity transformations at all training checkpoints.

### Raw recurrent rollout comparison

| Metric | Loop | 50 steps | 186 steps | Pre-fix (identity) |
| --- | --- | --- | --- | --- |
| rel_update_norm | 1 | `8.84` | `4.72` | `0.0` |
| | 2 | `0.25` | `0.95` | `0.0` |
| | 3 | `0.19` | `0.49` | `0.0` |
| cos_prev | 1 | `0.323` | `0.243` | `1.0` |
| | 2 | `0.987` | `0.869` | `1.0` |
| | 3 | `0.994` | `0.966` | `1.0` |
| cos_encoder | 1 | `0.323` | `0.243` | `1.0` |
| | 2 | `0.307` | `0.108` | `1.0` |
| | 3 | `0.294` | `0.046` | `1.0` |
| time_sensitivity | 1 | `0.000` | `0.000` | `0.0` |
| | 2 | `0.014` | `0.016` | `0.0` |
| | 3 | `0.034` | `0.035` | `0.0` |

Observations:
- Loop 1 does the heaviest transformation at both checkpoints. With more training, it calms down (8.84 → 4.72) as the model stabilizes.
- Loops 2-3 become **more** active with training (update norms increase 2.5-4x from 50 to 186 steps).
- The representation drifts progressively further from the encoder output across loops. By 186 steps, loop 3 output is nearly orthogonal to encoder output (cos = 0.046).
- Loops are also more differentiated from each other at 186 steps (cos_prev drops from 0.99 to 0.87 at loop 2).

## Finding 2: Time Embeddings Are Inert

Time sensitivity is near zero at both checkpoints and barely changes with training:
- Loop 1: `0.000` → `0.000`
- Loop 2: `0.014` → `0.016`
- Loop 3: `0.034` → `0.035`

The model is not learning to use the time embedding to do different things on different recurrent passes. Each loop applies approximately the same transformation, with differentiation coming only from the changing input state.

This may mean:
- The time embedding capacity is too small or the learning rate too low
- The model finds it easier to rely on implicit state evolution than explicit loop-index conditioning
- More training might eventually activate this channel, but there is no sign of it starting

## Finding 3: Quantization Error Compounds Across Loops

This is the most important new finding. At 50 steps the quantization drift is negligible, but at 186 steps it compounds severely:

### Raw vs quantized state drift

| State | 50 steps cos(raw, quant) | 186 steps cos(raw, quant) |
| --- | --- | --- |
| encoder (loop 0) | `0.999` | `0.998` |
| after loop 1 | `0.996` | **`0.827`** |
| after loop 2 | `0.996` | **`0.783`** |
| after loop 3 | `0.995` | **`0.741`** |

| State | 50 steps rel_diff | 186 steps rel_diff |
| --- | --- | --- |
| encoder (loop 0) | `0.052` | `0.067` |
| after loop 1 | `0.093` | **`0.583`** |
| after loop 2 | `0.093` | **`0.651`** |
| after loop 3 | `0.095` | **`0.717`** |

At 50 steps, int8 quantization barely disturbs the recurrence (cos > 0.995 everywhere). At 186 steps, the encoder output is still well-preserved (cos = 0.998), but after just one recurrent loop the quantized state diverges sharply (cos = 0.827), and it gets worse with each additional loop (0.783, 0.741).

The roundtrip bpb penalty confirms this matters: `+0.004` at 50 steps vs `+0.309` at 186 steps. The recurrent blocks amplify quantization error because:
- The first loop makes a large transformation, so any error in the weight representation gets amplified into the hidden state
- Each subsequent loop re-processes a noisy state through the same (noisy) weights, compounding the drift

This is fundamentally different from the previous identity-collapsed case where quantization drift was constant across loops (because the loops were no-ops).

## Probe Assets

- [50-step probe summary](assets/recurrent_block_probe_20260326T005037Z_smoketest_loopformer_20260325/summary.json)
- [2-min probe summary](assets/recurrent_block_probe_20260326T012005Z_smoketest_loopformer_2min_20260325/summary.json)

Plots (2-min run):
- [Relative update norm](assets/recurrent_block_probe_20260326T012005Z_smoketest_loopformer_2min_20260325/rel_update_norm.png)
- [Cosine with previous state](assets/recurrent_block_probe_20260326T012005Z_smoketest_loopformer_2min_20260325/cos_prev.png)
- [Time sensitivity](assets/recurrent_block_probe_20260326T012005Z_smoketest_loopformer_2min_20260325/time_sensitivity_rel.png)
- [Quantization drift](assets/recurrent_block_probe_20260326T012005Z_smoketest_loopformer_2min_20260325/raw_quant_rel_diff.png)

## Conclusions

1. **Gate-init fix resolved the identity collapse.** The recurrent blocks are now producing meaningful, non-degenerate transformations. This is the prerequisite for recurrence to help.

2. **Recurrence deepens with training.** Later loops become more active and the representation diverges further from the encoder output as training progresses. The model is learning to use the recurrent depth.

3. **Time embeddings are not yet useful.** The model ignores the loop index and relies on implicit state evolution. This is a potential area for improvement but may not matter if the implicit approach works.

4. **Int8 quantization compounds across recurrent loops.** This is the binding new constraint. At 186 steps of training, the quantized state after 3 loops is only 0.74 cosine with the raw state, and the roundtrip bpb penalty is 0.31. This penalty will likely grow with longer training as the recurrent transformations become sharper.

## Implications for Next Steps

The quantization drift problem creates a tension:
- More recurrent loops = more model capacity in raw fp32/bf16
- More recurrent loops = more quantization damage in the int8 submission
- The marginal value of each additional loop decreases (diminishing update norms) while the marginal quantization cost increases (compounding error)

Possible directions:
- **Quantization-aware training:** Add a quantization-roundtrip penalty to the training loss so the model learns representations that survive int8
- **Per-loop quantization calibration:** Calibrate int8 scales separately for each recurrent loop rather than globally
- **Fewer but stronger loops:** Trade loop count for per-loop capacity (e.g., 2 loops with more layers per loop instead of 3 loops)
- **Residual dampening:** Scale the recurrent update by a learned factor < 1 to reduce error amplification
