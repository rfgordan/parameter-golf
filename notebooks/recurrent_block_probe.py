from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_gpt import CastedLinear, GPT, Hyperparameters, load_data_shard, load_model_state_dict, restore_low_dim_params_to_fp32


DEFAULT_RUN_DIR = (
    "/Users/robertgordan/Projects/parameter-golf/"
    ".runpod/results/20260324T232001Z_loopformer_recur_4_3x3_4_int8_h100_10min_20260324"
)


@contextlib.contextmanager
def temporary_env(overrides: dict[str, str]):
    old_env = os.environ.copy()
    try:
        os.environ.update(overrides)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def args_from_config(run_dir: Path) -> SimpleNamespace:
    config = json.loads((run_dir / "config.json").read_text())
    env_overrides = {k: str(v) for k, v in config.get("env_overrides", {}).items()}
    data_path = env_overrides.get("DATA_PATH", Hyperparameters.data_path)
    args = SimpleNamespace(
        data_path=data_path,
        train_files=os.path.join(data_path, "fineweb_train_*.bin"),
        val_files=os.path.join(data_path, "fineweb_val_*.bin"),
        tokenizer_path=env_overrides.get("TOKENIZER_PATH", Hyperparameters.tokenizer_path),
        vocab_size=int(env_overrides.get("VOCAB_SIZE", Hyperparameters.vocab_size)),
        num_encoder_layers=int(env_overrides.get("NUM_ENCODER_LAYERS", Hyperparameters.num_encoder_layers)),
        num_recurrent_layers=int(env_overrides.get("NUM_RECURRENT_LAYERS", Hyperparameters.num_recurrent_layers)),
        num_decoder_layers=int(env_overrides.get("NUM_DECODER_LAYERS", Hyperparameters.num_decoder_layers)),
        num_recurrent_loops=int(env_overrides.get("NUM_RECURRENT_LOOPS", Hyperparameters.num_recurrent_loops)),
        num_kv_heads=int(env_overrides.get("NUM_KV_HEADS", Hyperparameters.num_kv_heads)),
        model_dim=int(env_overrides.get("MODEL_DIM", Hyperparameters.model_dim)),
        num_heads=int(env_overrides.get("NUM_HEADS", Hyperparameters.num_heads)),
        mlp_mult=int(env_overrides.get("MLP_MULT", Hyperparameters.mlp_mult)),
        tie_embeddings=bool(int(env_overrides.get("TIE_EMBEDDINGS", "1" if Hyperparameters.tie_embeddings else "0"))),
        tied_embed_init_std=float(env_overrides.get("TIED_EMBED_INIT_STD", Hyperparameters.tied_embed_init_std)),
        logit_softcap=float(env_overrides.get("LOGIT_SOFTCAP", Hyperparameters.logit_softcap)),
        rope_base=float(env_overrides.get("ROPE_BASE", Hyperparameters.rope_base)),
        qk_gain_init=float(env_overrides.get("QK_GAIN_INIT", Hyperparameters.qk_gain_init)),
    )
    return args


def build_model(run_dir: Path, model_path: Path, device: torch.device) -> tuple[SimpleNamespace, GPT]:
    args = args_from_config(run_dir)
    model = GPT(
        vocab_size=args.vocab_size,
        num_encoder_layers=args.num_encoder_layers,
        num_recurrent_mid_layers=args.num_recurrent_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_recurrent_loops=args.num_recurrent_loops,
        shortcut_consistency_weight=0.0,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    for block in model.recurrent_blocks:
        block.adaLN_modulation.bfloat16()
    state_dict = load_model_state_dict(str(model_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return args, model


def load_probe_batch(args: Hyperparameters, batch_size: int, seq_len: int, offset: int) -> torch.Tensor:
    val_files = sorted(glob.glob(args.val_files))
    if not val_files:
        raise FileNotFoundError(f"No validation shards found for {args.val_files}")
    tokens = load_data_shard(Path(val_files[0])).to(torch.int64)
    needed = batch_size * seq_len + 1
    start = min(offset, max(tokens.numel() - needed, 0))
    chunk = tokens[start : start + needed]
    if chunk.numel() < needed:
        raise ValueError(f"Need {needed} tokens for probe, got {chunk.numel()}")
    return chunk[:-1].reshape(batch_size, seq_len)


def recurrent_states(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    force_time: float | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.inference_mode():
        x = model.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for i in range(model.num_encoder_layers):
            x = model.encoder_blocks[i](x, x0)
        encoder_out = x.float().cpu()
        states = [encoder_out]
        for loop_idx in range(model.num_recurrent_loops):
            t_value = force_time if force_time is not None else loop_idx / model.num_recurrent_loops
            ti = torch.full((x.size(0), 1), t_value, dtype=torch.float32, device=x.device)
            c = model.time_embedder(ti).to(dtype=x.dtype)
            for block_idx in range(model.num_recurrent_mid_layers):
                x = model.recurrent_blocks[block_idx](x, c)
            states.append(x.float().cpu())
    return encoder_out, states


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1, a.size(-1)), b.reshape(-1, b.size(-1)), dim=-1).mean())


def rel_norm(delta: torch.Tensor, ref: torch.Tensor) -> float:
    return float(delta.norm() / ref.norm().clamp_min(1e-12))


def summarize(encoder_out: torch.Tensor, states: list[torch.Tensor], zero_time_states: list[torch.Tensor]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for loop_idx in range(1, len(states)):
        prev = states[loop_idx - 1]
        cur = states[loop_idx]
        zero_cur = zero_time_states[loop_idx]
        rows.append(
            {
                "loop": float(loop_idx),
                "rel_update_norm": rel_norm(cur - prev, prev),
                "cos_prev": cosine_mean(cur, prev),
                "cos_encoder": cosine_mean(cur, encoder_out),
                "time_sensitivity_rel": rel_norm(cur - zero_cur, cur),
            }
        )
    return rows


def compare_raw_quant(raw_states: list[torch.Tensor], quant_states: list[torch.Tensor]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for loop_idx in range(len(raw_states)):
        raw = raw_states[loop_idx]
        quant = quant_states[loop_idx]
        rows.append(
            {
                "loop": float(loop_idx),
                "raw_quant_rel_diff": rel_norm(raw - quant, raw),
                "raw_quant_cos": cosine_mean(raw, quant),
            }
        )
    return rows


def save_plots(
    output_dir: Path,
    raw_summary: list[dict[str, float]],
    quant_summary: list[dict[str, float]],
    raw_quant_summary: list[dict[str, float]],
) -> None:
    loops = [int(row["loop"]) for row in raw_summary]

    def plot_pair(key: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(7.5, 4.8))
        plt.plot(loops, [row[key] for row in raw_summary], marker="o", linewidth=2, label="raw")
        plt.plot(loops, [row[key] for row in quant_summary], marker="o", linewidth=2, label="quantized")
        plt.xlabel("Recurrent loop")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    plot_pair("rel_update_norm", "Relative update norm", "rel_update_norm.png")
    plot_pair("cos_prev", "Cosine with previous loop state", "cos_prev.png")
    plot_pair("time_sensitivity_rel", "Relative time-embedding sensitivity", "time_sensitivity_rel.png")

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(
        [int(row["loop"]) for row in raw_quant_summary],
        [row["raw_quant_rel_diff"] for row in raw_quant_summary],
        marker="o",
        linewidth=2,
        label="raw vs quant",
    )
    plt.xlabel("State index (0 = encoder output)")
    plt.ylabel("Relative raw/quant state diff")
    plt.title("Quantization drift across recurrent rollout")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "raw_quant_rel_diff.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--offset", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir or Path(
        "experiment_summaries/assets/recurrent_block_probe_" + run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    _, raw_model = build_model(run_dir, run_dir / "final_model.pt", device)
    run_args, quant_model = build_model(run_dir, run_dir / "final_model.int8.ptz", device)
    probe_x = load_probe_batch(run_args, args.batch_size, args.seq_len, args.offset).to(device)

    raw_encoder, raw_states = recurrent_states(raw_model, probe_x)
    _, raw_zero_states = recurrent_states(raw_model, probe_x, force_time=0.0)
    quant_encoder, quant_states = recurrent_states(quant_model, probe_x)
    _, quant_zero_states = recurrent_states(quant_model, probe_x, force_time=0.0)

    raw_summary = summarize(raw_encoder, raw_states, raw_zero_states)
    quant_summary = summarize(quant_encoder, quant_states, quant_zero_states)
    raw_quant_summary = compare_raw_quant(raw_states, quant_states)
    save_plots(output_dir, raw_summary, quant_summary, raw_quant_summary)

    summary = {
        "run_dir": str(run_dir),
        "probe": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "offset": args.offset,
            "device": str(device),
        },
        "raw_summary": raw_summary,
        "quant_summary": quant_summary,
        "raw_quant_summary": raw_quant_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote recurrent probe outputs to {output_dir}")
    print("Raw recurrent rollout:")
    for row in raw_summary:
        print(row)
    print("Quantized recurrent rollout:")
    for row in quant_summary:
        print(row)
    print("Raw vs quantized state drift:")
    for row in raw_quant_summary:
        print(row)


if __name__ == "__main__":
    main()
