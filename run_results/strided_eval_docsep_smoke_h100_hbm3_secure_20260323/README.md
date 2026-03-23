## Strided Eval Doc-Separated Smoke

This wrapper documents a partial smoke run for the updated doc-aware strided eval path.

Relevant controller records:
- [20260323T163642Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323](/Users/robertgordan/Projects/parameter-golf/.runpod/controller/runs/20260323T163642Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323.json)
- [20260323T163733Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323_v2](/Users/robertgordan/Projects/parameter-golf/.runpod/controller/runs/20260323T163733Z_strided_eval_docsep_smoke_h100_hbm3_secure_20260323_v2.json)

Observed outcome:
- First attempt used the wrong `git_ref` and checked out `main`; it was discarded.
- Second attempt used commit `14f112e090d4868b99b2c2d112dc099198fa5045`.
- Remote `HEAD` was verified before trusting the run.
- Training completed and the new eval path entered `sliding_eval` over document-aware chunks.
- The run was manually stopped after confirming progress through the new eval loop without runtime errors.

Captured evidence from live inspection:
- validation tokens: `62021632`
- eval config: `EVAL_DOC_SEPARATED=1`, `EVAL_SEQ_LEN=256`, `EVAL_STRIDE=64`, `EVAL_BATCH_SEQS=64`
- chunk count observed in eval: `847347`
- live progress reached at least `294464 / 847347` chunks (`34.8%`)

This smoke test confirmed the new eval logic executes on H100 with the intended commit, but it did not finish to terminal metrics because the pod was stopped early.
