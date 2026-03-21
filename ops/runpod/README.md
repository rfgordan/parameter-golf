# Runpod Experiment Setup

This directory contains the Runpod Pod execution layer for this repository:

- `controller.py`: creates Pods, starts runs in `tmux`, polls status, and egresses results
- `worker.py`: runs exactly one Parameter Golf experiment and persists results
- `remote_bootstrap.sh`: remote Pod bootstrap that clones the repo and launches the run in `tmux`

## Ordering

1. Install and configure `runpodctl` locally with your Runpod API key.
2. Ensure your SSH public key is attached to your Runpod account.
3. Set `RUNPOD_TEMPLATE_ID` for the official Parameter Golf Pod template, or rely on `RUNPOD_IMAGE_NAME`.
4. Use `controller.py submit` to create one Pod per run.
5. Use `controller.py watch --watch` to monitor active runs and receive local-shell notifications.
6. Use `controller.py attach --run-id ...` to print a `tmux` attach command if you want to inspect a live run.
7. The controller copies the final remote run directory to `.runpod/results/` and then deletes the Pod.

The controller is optimized for quick actionable updates without noisy repeated output:

- SSH readiness is polled aggressively at first so the run can start as soon as the Pod is reachable.
- `watch --watch` only reprints the status table when the rendered state changes.
- Notifications and summaries include the `tmux` session name by default so it is obvious what to attach to.

## What You Need To Do

- Configure `runpodctl` with your Runpod API key.
- Add an SSH public key to your Runpod account so Pods can be reached over SSH.
- Set the Pod template ID or image name:
  - `RUNPOD_TEMPLATE_ID`
  - optional fallback `RUNPOD_IMAGE_NAME`
- Optionally override:
  - `RUNPOD_GPU_TYPE`
  - `PARAMETER_GOLF_NOTIFY_CMD`
  - `PARAMETER_GOLF_DATASET_VARIANT`
  - `PARAMETER_GOLF_TRAIN_SHARDS`

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

Each run writes to a unique remote directory on the Pod and is then copied locally under `.runpod/results/`:

- `config.json`
- `metrics.json`
- `artifact_summary.json`
- `stdout.log`
- `logs/` from the trainer
- final model artifacts if produced

Runs do not rely on a network volume in this version. Dataset download happens per Pod.

## Local Controller State

The controller persists run state locally under `.runpod/controller/`. This is the durable record for Pod IDs, SSH info, `tmux` session names, and terminal status.
