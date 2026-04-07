from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = Path(os.environ.get("PARAMETER_GOLF_RUNS_ROOT", ".runpod/runs"))
DEFAULT_TRAIN_SCRIPT = os.environ.get("PARAMETER_GOLF_TRAIN_SCRIPT", "train_gpt.py")
DEFAULT_TRAIN_COMMAND = os.environ.get(
    "PARAMETER_GOLF_TRAIN_COMMAND",
    f"{sys.executable} {REPO_ROOT / DEFAULT_TRAIN_SCRIPT}",
)

VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<iters>\d+) val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)")
TRAIN_RE = re.compile(r"step:(?P<step>\d+)/(?P<iters>\d+) train_loss:(?P<train_loss>\S+)")
ROUNDTRIP_RE = re.compile(
    r"final_int(?:8|6)_zlib_roundtrip_exact val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)"
)
TOTAL_SIZE_RE = re.compile(r"Total submission size int(?:8|6)\+zlib: (?P<bytes>\d+) bytes")
SERIALIZED_RE = re.compile(r"Serialized model int(?:8|6)\+zlib: (?P<bytes>\d+) bytes")


@dataclass
class RunPaths:
    run_dir: Path
    stdout_log: Path
    config_json: Path
    metrics_json: Path
    artifact_summary_json: Path


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "run"


def read_payload(path: str | None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("expected JSON payload on stdin or via --payload-file")
    return json.loads(raw)


def resolve_run_paths(run_name: str) -> RunPaths:
    fixed_run_dir = os.environ.get("PARAMETER_GOLF_FIXED_RUN_DIR")
    if fixed_run_dir:
        run_dir = Path(fixed_run_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = slugify(run_name)
        run_dir = DEFAULT_RUNS_ROOT / f"{stamp}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        stdout_log=run_dir / "stdout.log",
        config_json=run_dir / "config.json",
        metrics_json=run_dir / "metrics.json",
        artifact_summary_json=run_dir / "artifact_summary.json",
    )


def current_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_env(payload: dict[str, Any], run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    overrides = {str(k): str(v) for k, v in payload.get("env_overrides", {}).items()}
    env.update(overrides)
    env.setdefault("RUN_ID", slugify(str(payload["run_name"])))
    if "DATA_PATH" in env:
        env["DATA_PATH"] = str(Path(env["DATA_PATH"]).expanduser().resolve())
    if "TOKENIZER_PATH" in env:
        env["TOKENIZER_PATH"] = str(Path(env["TOKENIZER_PATH"]).expanduser().resolve())
    env["PYTHONUNBUFFERED"] = "1"
    env["PARAMETER_GOLF_RUN_DIR"] = str(run_dir)
    return env


def parse_metrics(log_text: str) -> dict[str, Any]:
    train_matches = list(TRAIN_RE.finditer(log_text))
    val_matches = list(VAL_RE.finditer(log_text))
    roundtrip = ROUNDTRIP_RE.search(log_text)
    total_size = TOTAL_SIZE_RE.search(log_text)
    serialized = SERIALIZED_RE.search(log_text)

    return {
        "last_train_step": int(train_matches[-1].group("step")) if train_matches else None,
        "last_train_loss": float(train_matches[-1].group("train_loss")) if train_matches else None,
        "last_val_step": int(val_matches[-1].group("step")) if val_matches else None,
        "last_val_loss": float(val_matches[-1].group("val_loss")) if val_matches else None,
        "last_val_bpb": float(val_matches[-1].group("val_bpb")) if val_matches else None,
        "final_roundtrip_val_loss": float(roundtrip.group("val_loss")) if roundtrip else None,
        "final_roundtrip_val_bpb": float(roundtrip.group("val_bpb")) if roundtrip else None,
        "artifact_total_submission_bytes": int(total_size.group("bytes")) if total_size else None,
        "artifact_compressed_model_bytes": int(serialized.group("bytes")) if serialized else None,
    }


def collect_artifacts(run_dir: Path) -> dict[str, Any]:
    artifact_paths = []
    for name in ("final_model.pt", "final_model.int8.ptz"):
        path = run_dir / name
        if path.is_file():
            artifact_paths.append({"path": str(path), "bytes": path.stat().st_size})
    trainer_logs_dir = run_dir / "logs"
    trainer_logs = sorted(str(path) for path in trainer_logs_dir.glob("*.txt")) if trainer_logs_dir.is_dir() else []
    return {"run_dir": str(run_dir), "artifacts": artifact_paths, "trainer_logs": trainer_logs}


def run_training(payload: dict[str, Any], paths: RunPaths) -> dict[str, Any]:
    run_name = str(payload["run_name"])
    git_ref = payload.get("git_ref")
    git_head = current_git_head()
    env = resolve_env(payload, paths.run_dir)
    if payload.get("command"):
        raise ValueError("worker payloads must not override the command; use env_overrides only")
    command = DEFAULT_TRAIN_COMMAND

    config = {
        "run_name": run_name,
        "git_ref": git_ref,
        "git_head": git_head,
        "command": command,
        "env_overrides": {str(k): str(v) for k, v in payload.get("env_overrides", {}).items()},
        "notes": payload.get("notes"),
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "run_dir": str(paths.run_dir),
    }
    write_json(paths.config_json, config)

    started = time.time()
    with paths.stdout_log.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "run_started", "created_at": config["created_at"]}) + "\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            cwd=paths.run_dir,
            env=env,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - started
    log_text = paths.stdout_log.read_text(encoding="utf-8")
    metrics = parse_metrics(log_text)
    env_overrides = {str(k): str(v) for k, v in payload.get("env_overrides", {}).items()}
    train_only = env_overrides.get("TRAIN_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    train_only_success = train_only and (paths.run_dir / "final_model.pt").is_file()
    roundtrip_success = metrics["final_roundtrip_val_bpb"] is not None
    status = "success" if proc.returncode == 0 and (roundtrip_success or train_only_success) else "failed"
    payload_out = {
        "status": status,
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "run_name": run_name,
        "git_ref": git_ref,
        "git_head": git_head,
        "run_dir": str(paths.run_dir),
        "stdout_log": str(paths.stdout_log),
        **metrics,
    }
    write_json(paths.metrics_json, payload_out)
    write_json(paths.artifact_summary_json, collect_artifacts(paths.run_dir))
    return payload_out


def handle_job(payload: dict[str, Any]) -> dict[str, Any]:
    if "run_name" not in payload:
        raise ValueError("payload must include run_name")
    paths = resolve_run_paths(str(payload["run_name"]))
    try:
        return run_training(payload, paths)
    except Exception as exc:
        failure = {
            "status": "failed",
            "run_name": str(payload.get("run_name", "unknown")),
            "error": f"{type(exc).__name__}: {exc}",
            "run_dir": str(paths.run_dir),
            "stdout_log": str(paths.stdout_log),
        }
        if not paths.stdout_log.exists():
            paths.stdout_log.write_text(f"{failure['error']}\n", encoding="utf-8")
        write_json(paths.metrics_json, failure)
        write_json(paths.artifact_summary_json, collect_artifacts(paths.run_dir))
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Parameter Golf experiment")
    parser.add_argument("--payload-file", help="Path to job payload JSON", default=None)
    args = parser.parse_args()
    payload = read_payload(args.payload_file)
    result = handle_job(payload)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
