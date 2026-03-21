# Repo Notes

This repository is a Parameter Golf challenge baseline and experiment archive, not a conventional application.

## Important Files

- `train_gpt.py`: primary CUDA/PyTorch baseline trainer.
- `train_gpt_mlx.py`: Apple Silicon / MLX local trainer for quick iteration.
- `data/`: dataset and tokenizer download/export helpers for FineWeb challenge data.
- `records/`: archived submissions and experiment snapshots; competitive variants live here.
- `README.md`: challenge rules, leaderboard, setup, and submission format.

## Core Workflow

The main workflow is:

1. Download cached FineWeb shards and tokenizer assets from `data/`.
2. Train with `train_gpt.py` or `train_gpt_mlx.py`.
3. Evaluate with tokenizer-agnostic `val_bpb`.
4. Quantize and compress the trained weights to check the final artifact stays under the 16 MB budget.

## Model / Eval Notes

Both training scripts implement a compact GPT-style model and include:

- tokenizer-agnostic validation using `val_bpb`
- post-training quantization / compressed artifact measurement
- baseline training loops intended as starting points, not SOTA implementations

## Practical Guidance

- Treat top-level training scripts as baselines.
- Look in `records/` for the strongest or most specialized approaches.
- Ignore `main.py`; it is a placeholder and not part of the real workflow.
