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

## Finding 4: Shortcut Consistency Loss — Improves Quant Robustness but Reduces Loop Differentiation

A shortcut consistency loss was added (weight=0.1): during training, the recurrence is randomly truncated at a uniformly sampled depth and the intermediate state is decoded through the shared decoder head with an auxiliary CE loss. This encourages each intermediate recurrent state to be independently useful for prediction.

### Run

| Run | Training budget | Steps completed | Run ID |
| --- | --- | --- | --- |
| Shortcut 3-min | 180s wallclock | 235 | `20260326T022417Z_shortcut_consistency_3min_20260325` |

Same architecture (`4/3x3/4`), same branch, commit `41a81087`. `SHORTCUT_CONSISTENCY_WEIGHT=0.1`, `USE_INT6=0`.

Raw results: [shortcut consistency 3-min results](../.runpod/results/20260326T022417Z_shortcut_consistency_3min_20260325)

### Training metrics comparison (baseline 2-min vs shortcut 3-min)

| Metric | Baseline 2-min (186 steps) | Shortcut 3-min (235 steps) |
| --- | --- | --- |
| last train loss | `3.399` | `3.349` |
| final val bpb | `1.958` | `1.785` |
| int8 roundtrip val bpb | `2.267` | `1.971` |
| **quantization penalty (bpb)** | **`+0.309`** | **`+0.186`** |
| int8+zlib model bytes | `10,637,311` | `11,424,384` |
| step avg | `~645 ms` | `~767 ms` |

The shortcut run trained longer (235 vs 186 steps) due to 3-min budget, so raw bpb is better. The key comparison is the quant penalty: **+0.186 vs +0.309**, a 40% reduction. Step time is ~19% slower due to the extra decoder forward pass.

### Probe comparison: recurrent dynamics

| Metric | Loop | Baseline 2-min | Shortcut 3-min | Direction |
| --- | --- | --- | --- | --- |
| rel_update_norm | 1 | `4.72` | `5.78` | higher (more active) |
| | 2 | `0.95` | `0.83` | lower |
| | 3 | `0.49` | `0.45` | lower |
| cos_prev | 1 | `0.243` | `0.232` | ≈same |
| | 2 | `0.869` | **`0.890`** | higher (less differentiated) |
| | 3 | `0.966` | **`0.971`** | higher (less differentiated) |
| cos_encoder | 1 | `0.243` | `0.232` | ≈same |
| | 2 | `0.108` | `0.149` | higher (closer to encoder) |
| | 3 | `0.046` | **`0.110`** | higher (closer to encoder) |
| time_sensitivity | 1 | `0.000` | `0.000` | same |
| | 2 | `0.016` | `0.020` | slightly higher |
| | 3 | `0.035` | `0.041` | slightly higher |

### Quantization drift comparison

| State | Baseline cos(raw,quant) | Shortcut cos(raw,quant) | Improvement |
| --- | --- | --- | --- |
| encoder (loop 0) | `0.998` | `0.998` | — |
| after loop 1 | `0.827` | **`0.894`** | +0.067 |
| after loop 2 | `0.783` | **`0.866`** | +0.083 |
| after loop 3 | `0.741` | **`0.835`** | +0.094 |

| State | Baseline rel_diff | Shortcut rel_diff | Improvement |
| --- | --- | --- | --- |
| encoder (loop 0) | `0.067` | `0.069` | — |
| after loop 1 | `0.583` | **`0.456`** | -0.127 |
| after loop 2 | `0.651` | **`0.514`** | -0.137 |
| after loop 3 | `0.717` | **`0.573`** | -0.144 |

### Interpretation

Shortcut consistency achieves its goal of improving quantization robustness — the quantized state stays substantially closer to the raw state at every loop depth. However, it comes with a trade-off:

**The loops become less differentiated.** cos_prev increases at loops 2–3, and cos_encoder increases at loop 3 (from 0.046 → 0.110), meaning the later loops are doing less novel transformation and staying closer to earlier states. This is the expected consequence of requiring each intermediate state to decode well independently — the model can't afford to move far from a decodable representation at any step.

This creates a tension: the loss improves quant robustness by constraining the representation to a decoder-friendly subspace, but that same constraint limits how much the recurrence can transform the representation, potentially capping the expressiveness benefit of multiple loops.

### Probe assets

- [Shortcut 3-min probe summary](assets/recurrent_block_probe_20260326T022417Z_shortcut_consistency_3min_20260325/summary.json)
- [Quantization drift](assets/recurrent_block_probe_20260326T022417Z_shortcut_consistency_3min_20260325/raw_quant_rel_diff.png)
- [Cosine with previous](assets/recurrent_block_probe_20260326T022417Z_shortcut_consistency_3min_20260325/cos_prev.png)

## Finding 5: Late QAT — Near-Zero Quant Penalty, Same Loop Dynamics Trade-off

Two QAT variants were tested, both activated in the last 30% of training (`QAT_FRACTION=0.3`, wallclock-based):
- **Noise QAT** (`QAT_MODE=noise`): Injects uniform noise calibrated to per-row int8 step size. Gradients flow naturally.
- **STE QAT** (`QAT_MODE=ste`): Straight-through estimator fake-quantize matching the exact export format (round, clamp, dequant per row).

### Runs

| Run | QAT mode | Steps | QAT activated at | Run ID |
| --- | --- | --- | --- | --- |
| Noise 3-min | noise | 214 | step 193 (126.6s) | `20260326T043602Z_noisy_qat_3min_20260325` |
| STE 3-min | ste | 212 | step 193 (126.2s) | `20260326T050543Z_ste_qat_3min_20260325` |

Same architecture (`4/3x3/4`), same branch, commit `ca4e2fd`. `QAT_FRACTION=0.3`, `USE_INT6=0`.

### Training metrics comparison

| Metric | Baseline 2-min | Shortcut 3-min | Noise QAT 3-min | STE QAT 3-min |
| --- | --- | --- | --- | --- |
| steps | 186 | 235 | 214 | 212 |
| val bpb | 1.958 | 1.785 | 1.749 | 1.772 |
| roundtrip bpb | 2.267 | 1.971 | 1.836 | **1.780** |
| **quant penalty** | **+0.309** | **+0.186** | **+0.087** | **+0.009** |
| step avg (pre-QAT) | 645 ms | 767 ms | 657 ms | 654 ms |

STE QAT achieves near-zero quant penalty (+0.009 bpb). Noise QAT gets +0.087 — much better than baseline/shortcut but worse than STE because uniform noise only approximates the rounding behavior.

**Performance note**: QAT triggers a one-time `torch.compile` retrace (~40s) when the branch flips. Post-recompile per-step time is identical to pre-QAT (~655ms). The retrace cost is 22% of a 3-min budget but only 6.7% of a 10-min run. This can be eliminated by pre-warming both code paths during the existing warmup phase.

### Probe comparison: recurrent dynamics

| Metric | Loop | Baseline | Shortcut | Noise QAT | STE QAT |
| --- | --- | --- | --- | --- | --- |
| rel_update_norm | 1 | `4.72` | `5.78` | `5.97` | `5.79` |
| | 2 | `0.95` | `0.83` | `0.78` | `0.78` |
| | 3 | `0.49` | `0.45` | `0.44` | `0.44` |
| cos_prev | 1 | `0.243` | `0.232` | `0.294` | `0.308` |
| | 2 | `0.869` | `0.890` | `0.907` | `0.908` |
| | 3 | `0.966` | `0.971` | `0.972` | `0.973` |
| cos_encoder | 1 | `0.243` | `0.232` | `0.294` | `0.308` |
| | 2 | `0.108` | `0.149` | `0.207` | `0.217` |
| | 3 | `0.046` | `0.110` | `0.158` | `0.165` |

### Quantization drift comparison

| State | Baseline | Shortcut | Noise QAT | STE QAT |
| --- | --- | --- | --- | --- |
| encoder (loop 0) | `0.998` | `0.998` | `0.998` | `0.998` |
| after loop 1 | `0.827` | `0.894` | **`0.941`** | **`0.945`** |
| after loop 2 | `0.783` | `0.866` | **`0.926`** | **`0.932`** |
| after loop 3 | `0.741` | `0.835` | **`0.909`** | **`0.916`** |

### Interpretation

Both QAT variants dramatically improve quantization robustness — STE nearly eliminates the quant penalty entirely. However, both show **higher cos_prev and cos_encoder than even shortcut consistency**, meaning loop differentiation decreases similarly regardless of whether the regularization targets the weights (QAT) or the representations (shortcut).

The key difference: despite similar loop dynamics, QAT achieves much better roundtrip bpb because it makes the *weights* robust to rounding, not just the *representations*. Shortcut consistency constrains representations to a decoder-friendly subspace; QAT constrains weights to survive quantization. The weight-level approach is strictly more effective for the actual competition metric.

STE > noise because it exactly matches the export format's rounding behavior rather than approximating it with continuous noise. The model learns to place weights where rounding will land them correctly.

### Probe assets

- [Noise QAT probe summary](assets/recurrent_block_probe_20260326T043602Z_noisy_qat_3min_20260325/summary.json)
- [STE QAT probe summary](assets/recurrent_block_probe_20260326T050543Z_ste_qat_3min_20260325/summary.json)

## Finding 6: 10-Min STE QAT — Quant Drift Eliminated, Loops 2-3 Collapse to Identity

A full 10-minute STE QAT run with strided eval (stride 64) confirms the quant penalty is fully solved at scale, but reveals that later loops become near-identity with longer training.

### Run

| Run | Training budget | Steps | QAT activated at | Run ID |
| --- | --- | --- | --- | --- |
| STE QAT 10-min | 600s wallclock | 910 | step 673 (420.5s) | `20260326T053304Z_ste_qat_10min_strided_20260325` |

Same architecture (`4/3x3/4`), commit `ca4e2fd`. `QAT_FRACTION=0.3`, `QAT_MODE=ste`, `EVAL_STRIDED_ATTN=1`, `EVAL_STRIDE=64`.

### Training metrics

| Metric | Value |
| --- | --- |
| steps | 910 |
| val bpb (strided) | 1.3487 |
| roundtrip bpb | **1.3490** |
| **quant penalty** | **+0.0003** |
| compressed model | 17.7 MB |
| step avg | 660 ms |

For comparison, the non-recurrent baseline from the int6 experiment achieved **1.276 roundtrip bpb** — the recurrent model is still 0.073 bpb behind despite near-zero quant penalty.

### Probe comparison: 3-min vs 10-min

| Metric | Loop | 3-min baseline | 3-min STE QAT | **10-min STE QAT** |
| --- | --- | --- | --- | --- |
| rel_update_norm | 1 | `4.72` | `5.79` | **`9.36`** |
| | 2 | `0.95` | `0.78` | **`0.18`** |
| | 3 | `0.49` | `0.44` | **`0.15`** |
| cos_prev | 1 | `0.243` | `0.308` | `0.303` |
| | 2 | `0.869` | `0.908` | **`0.979`** |
| | 3 | `0.966` | `0.973` | **`0.989`** |
| cos_encoder | 1 | `0.243` | `0.308` | `0.303` |
| | 2 | `0.108` | `0.217` | **`0.276`** |
| | 3 | `0.046` | `0.165` | **`0.255`** |
| time_sensitivity | 1 | `0.000` | `0.000` | `0.000` |
| | 2 | `0.016` | `0.017` | **`0.029`** |
| | 3 | `0.035` | `0.037` | **`0.074`** |

### Quantization drift — fully eliminated

| State | 3-min baseline | 3-min STE QAT | **10-min STE QAT** |
| --- | --- | --- | --- |
| encoder | `0.998` | `0.998` | `0.998` |
| loop 1 | `0.827` | `0.945` | **`0.999`** |
| loop 2 | `0.783` | `0.932` | **`0.998`** |
| loop 3 | `0.741` | `0.916` | **`0.998`** |

### Interpretation

**Quant drift is completely solved.** cos(raw,quant) ≥ 0.998 at every loop — the model has fully learned to survive int8 rounding with more training time for the STE path to work.

**Loops 2-3 collapse to near-identity.** Update norms at loops 2-3 dropped to 0.18/0.15 (vs 0.95/0.49 at 3-min baseline), and cos_prev reached 0.979/0.989. Loop 1 does all the heavy lifting (update norm 9.36). The model is paying for 3 loops but effectively only using 1.

**Time sensitivity is finally emerging.** Loop 3 reached 0.074 (2x the 3-min value). The model is starting to learn loop-index-dependent behavior, but loop 1 remains completely insensitive (0.000 — it always sees t=0).

**The bottleneck is speed, not quantization.** At 660ms/step × 910 steps, the recurrent model trains much less than a non-recurrent model would in the same 10-min budget. The 0.073 bpb gap to the non-recurrent baseline is a training-step deficit, not a quantization problem.

### Probe assets

- [10-min STE QAT probe summary](assets/recurrent_block_probe_20260326T053304Z_ste_qat_10min_strided_20260325/summary.json)

## Finding 7: 2/3x3/2 with Loop Curriculum — Faster Steps, Higher Time Sensitivity, Worse bpb

A smaller architecture (2 encoder / 3 recurrent mid / 2 decoder, 4 loops) was tested with a loop curriculum (2→3→4 loops) and always-on STE QAT, aiming to recover training steps through faster per-step time.

### Run

| Run | Training budget | Steps | Run ID |
| --- | --- | --- | --- |
| 2/3x3/2 curriculum 10-min | 600s wallclock | 985 | `20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325` |

Branch `strided-eval-recurse-loopformer-v1`, commit `a405556`. `QAT_FRACTION=0.3`, `QAT_MODE=ste`, `EVAL_STRIDED_ATTN=1`, `EVAL_STRIDE=64`.
- Architecture: `2` encoder / `3` recurrent mid / `4` loops / `2` decoder
- Model params: `16,924,728` (vs `24,274,008` for 4/3x3/4)
- Curriculum: 2 loops for first 80% (831 steps), 3 loops for next 10% (54 steps), 4 loops for final 10% (100 steps)
- QAT activated at step 764 (70% of wallclock)

### Training metrics comparison

| Metric | 4/3x3/4 10-min (Finding 6) | **2/3x3/2 curriculum 10-min** |
| --- | --- | --- |
| steps | 910 | **985** (+8.2%) |
| params | 24.3M | **16.9M** (-30%) |
| step avg | 660 ms | **610 ms** (-7.6%) |
| val bpb (strided) | 1.3487 | 1.3606 |
| roundtrip bpb | **1.3490** | 1.3611 |
| **quant penalty** | **+0.0003** | **+0.0005** |
| compressed model | 17.7 MB | **12.3 MB** (-30%) |

The smaller model trains 8.2% more steps in the same budget but reaches 0.012 worse roundtrip bpb. Quant penalty remains negligible.

### Probe comparison

| Metric | Loop | 4/3x3/4 10-min | **2/3x3/2 curriculum** |
| --- | --- | --- | --- |
| rel_update_norm | 1 | `9.36` | **`25.06`** |
| | 2 | `0.18` | `0.29` |
| | 3 | `0.15` | `0.22` |
| | 4 | — | `0.19` |
| cos_prev | 1 | `0.303` | `0.295` |
| | 2 | `0.979` | `0.980` |
| | 3 | `0.989` | `0.988` |
| | 4 | — | `0.990` |
| cos_encoder | 1 | `0.303` | `0.295` |
| | 2 | `0.276` | `0.281` |
| | 3 | `0.255` | `0.265` |
| | 4 | — | `0.247` |
| time_sensitivity | 1 | `0.000` | `0.000` |
| | 2 | `0.029` | `0.037` |
| | 3 | `0.074` | `0.091` |
| | 4 | — | **`0.160`** |

### Quantization drift — still fully eliminated

| State | 4/3x3/4 10-min | **2/3x3/2 curriculum** |
| --- | --- | --- |
| encoder | `0.999` | `0.999` |
| loop 1 | `0.999` | `0.999` |
| loop 2 | `0.998` | `0.999` |
| loop 3 | `0.998` | `0.999` |
| loop 4 | — | `0.999` |

### Interpretation

**Loop 1 does dramatically more work.** With fewer encoder/decoder layers to pre/post-process, the recurrent block compensates with a much larger first-loop transformation (update norm 25.1 vs 9.4). Loops 2-4 still collapse to near-identity (cos_prev ≥ 0.980), but slightly less than the 4/3x3/4 case.

**Time sensitivity is 2x higher.** Loop 4 reaches 0.160 time sensitivity (vs 0.074 at loop 3 in the 4/3x3/4 model). The curriculum may help here — the model only sees loops 3-4 for the last 20% of training, so it must learn to differentiate them quickly. The gradient pressure to use the time embedding is higher when loops are scarce.

**The step count advantage doesn't compensate for model capacity.** Despite 8.2% more steps, the smaller model is 0.012 bpb worse. The 30% reduction in encoder/decoder parameters removes capacity that the recurrent blocks can't fully replace, at least at this training duration.

**Quant drift is architecture-invariant.** STE QAT eliminates quant penalty regardless of encoder/decoder depth or number of loops — cos(raw,quant) ≥ 0.999 everywhere.

### Curriculum step timing

| Phase | Steps | Loops | Approx ms/step |
| --- | --- | --- | --- |
| Steps 1–764 | 764 | 2 | ~551 |
| Steps 764–831 | 67 | 2 + QAT (recompile) | ~750* |
| Steps 831–885 | 54 | 3 + QAT | ~1057* |
| Steps 885–985 | 100 | 4 + QAT | ~604 |

*Includes one-time torch.compile retrace costs from QAT activation and loop count changes. The always-on QAT simplification (now implemented) will eliminate the QAT retrace. Loop count transitions will still cause retraces.

### Probe assets

- [2/3x3/2 curriculum probe summary](assets/recurrent_block_probe_20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325/summary.json)
- [Relative update norm](assets/recurrent_block_probe_20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325/rel_update_norm.png)
- [Cosine with previous](assets/recurrent_block_probe_20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325/cos_prev.png)
- [Time sensitivity](assets/recurrent_block_probe_20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325/time_sensitivity_rel.png)
- [Quantization drift](assets/recurrent_block_probe_20260326T150828Z_loopformer_2_3x3_2_curriculum234_10min_20260325/raw_quant_rel_diff.png)

## Conclusions

1. **Gate-init fix resolved the identity collapse.** The recurrent blocks are now producing meaningful, non-degenerate transformations. This is the prerequisite for recurrence to help.

2. **Recurrence deepens with training.** Later loops become more active and the representation diverges further from the encoder output as training progresses. The model is learning to use the recurrent depth.

3. **Time embeddings are not yet useful.** The model ignores the loop index and relies on implicit state evolution. This is a potential area for improvement but may not matter if the implicit approach works.

4. **Int8 quantization compounds across recurrent loops.** This is the binding new constraint. At 186 steps of training, the quantized state after 3 loops is only 0.74 cosine with the raw state, and the roundtrip bpb penalty is 0.31. This penalty will likely grow with longer training as the recurrent transformations become sharper.

5. **Shortcut consistency reduces quant penalty but limits loop expressiveness.** With weight=0.1, quant penalty drops 40% (+0.309 → +0.186 bpb) and cos(raw,quant) at loop 3 improves from 0.741 → 0.835. But loops 2–3 become less differentiated (cos_prev increases) and stay closer to the encoder output. The loss constrains the recurrence to a decoder-friendly subspace — good for quantization, but potentially capping the value of additional loops.

6. **STE QAT nearly eliminates the quant penalty.** Late STE fake-quantize (last 30% of training) reduces the quant penalty to +0.009 bpb at 3 min and **+0.0003 bpb at 10 min**. This is strictly better than shortcut consistency (+0.186) and noise QAT (+0.087). STE achieves the best roundtrip bpb because it exactly matches the export format's rounding behavior.

7. **Loops 2-3 collapse to near-identity at scale.** With 10 min of training + STE QAT, loop 1 does all the work (update norm 9.36) while loops 2-3 are near-no-ops (0.18/0.15). cos_prev at loop 3 reaches 0.989. The model is paying for 3 loops but effectively only using 1. Time sensitivity is finally emerging at loop 3 (0.074) but remains negligible.

8. **The bottleneck is speed, not quantization.** The recurrent 10-min model achieves 1.349 roundtrip bpb vs 1.276 for the non-recurrent baseline — a 0.073 gap. Quant penalty is zero, so the entire gap is from fewer training steps at 660ms/step.

9. **Smaller encoder/decoder with curriculum trades capacity for speed, net negative.** The 2/3x3/2 model trains 8.2% more steps (985 vs 910) but reaches 0.012 worse roundtrip bpb (1.361 vs 1.349). The step count gain from 30% fewer parameters doesn't compensate for the lost capacity.

10. **Loop curriculum boosts time sensitivity.** The 2/3x3/2 curriculum run shows 2x the time sensitivity of the 4/3x3/4 run (0.160 at loop 4 vs 0.074 at loop 3). Introducing loops late in training forces the model to differentiate them quickly.

11. **Always-on STE QAT eliminates recompile overhead.** QAT per-step cost is ~2-10ms (negligible). Removing the branch that activates QAT mid-training avoids a ~40s torch.compile retrace. Now implemented as a simple `QAT_ENABLED` boolean with no scheduling.

## Implications for Next Steps

The quantization problem is fully solved by STE QAT. The binding constraint is now step throughput — the recurrent model trains fewer steps in the same wallclock budget.

Remaining directions:
- **Improve recurrence utilization:** Loops 2+ collapse to near-identity at scale. The model needs architectural changes (e.g., damped residuals, forced bottlenecks) to make later loops do meaningful work.
- **Curriculum for time sensitivity:** Loop curriculum shows promise for activating time embeddings. Explore more aggressive schedules or longer tier-3/4 phases.
- **Reduce step time without losing capacity:** The 2/3x3/2 experiment shows that simply shrinking encoder/decoder is net negative. Other options: attention optimization, fewer recurrent mid layers with more loops, mixed precision improvements.
