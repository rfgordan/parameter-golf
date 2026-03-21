# Runpod Experiment Setup

This directory contains the minimal Runpod execution layer for this repository:

- `worker.py`: runs exactly one Parameter Golf experiment and persists results
- `controller.py`: submits async Runpod Serverless jobs, polls status, and renders summaries
- `serverless_handler.py`: Runpod Serverless handler entrypoint

## Ordering

1. Create a restricted Runpod API key.
2. Create one network volume in the datacenter you want to use.
3. Choose the exact H100 SKU for the endpoint.
4. Build and deploy a Serverless image that invokes `python3 /workspace/parameter-golf/ops/runpod/serverless_handler.py`.
5. Create a Serverless endpoint with `gpuCount=1`, `workersMin=0`, and `workersMax=2`.
6. Mount the network volume on the endpoint at `/runpod-volume`.
7. Use `controller.py submit` to launch runs and `controller.py summarize` to render summaries.
8. Use a separate debug Pod for interactive debugging only.

## What You Need To Do

- Create the Runpod API key and export it as `RUNPOD_API_KEY`.
- Create the network volume and endpoint, then provide/export:
  - `RUNPOD_ENDPOINT_ID`
  - `RUNPOD_VOLUME_ID`
- Build/deploy the Serverless image in Runpod.
- Choose whether the worker image preloads dataset assets or downloads them into `/runpod-volume/data/`.

## Worker Payload

The worker accepts a single JSON payload:

```json
{
  "run_name": "baseline_seq2048_lr004",
  "git_ref": "main",
  "env_overrides": {
    "ITERATIONS": "400",
    "TRAIN_SEQ_LEN": "2048",
    "MATRIX_LR": "0.04"
  },
  "notes": "optional"
}
```

## Persistent Output Layout

Each run writes to a unique directory under `/runpod-volume/runs/`:

- `config.json`
- `metrics.json`
- `artifact_summary.json`
- `stdout.log`
- `logs/` from the trainer
- final model artifacts if produced

Workers must never update a shared file on the network volume.

## Local Controller State

The controller persists API responses locally under `.runpod/controller/`. This is the durable record for job IDs and terminal summaries; async Runpod job output should be treated as temporary.
