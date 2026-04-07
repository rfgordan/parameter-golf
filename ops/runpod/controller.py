from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = Path(os.environ.get("PARAMETER_GOLF_CONTROLLER_ROOT", ".runpod/controller"))
RESULTS_ROOT = Path(os.environ.get("PARAMETER_GOLF_RESULTS_ROOT", ".runpod/results"))
DEFAULT_REPO_URL = os.environ.get("PARAMETER_GOLF_GIT_URL", "https://github.com/rfgordan/parameter-golf.git")
DEFAULT_GIT_REF = os.environ.get("PARAMETER_GOLF_GIT_REF", "main")
DEFAULT_GPU_TYPE = os.environ.get("RUNPOD_GPU_TYPE", "NVIDIA H100 80GB HBM3")
DEFAULT_TEMPLATE_ID = os.environ.get("RUNPOD_TEMPLATE_ID")
DEFAULT_IMAGE_NAME = os.environ.get("RUNPOD_IMAGE_NAME", "runpod/parameter-golf:latest")
DEFAULT_CONTAINER_DISK_GB = int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "100"))
DEFAULT_VOLUME_GB = int(os.environ.get("RUNPOD_VOLUME_GB", "30"))
DEFAULT_CLOUD_TYPE = os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE")
DEFAULT_POLL_INTERVAL = float(os.environ.get("PARAMETER_GOLF_POLL_INTERVAL", "5"))
DEFAULT_SSH_POLL_INTERVAL_FAST = float(os.environ.get("PARAMETER_GOLF_SSH_POLL_INTERVAL_FAST", "2"))
DEFAULT_SSH_POLL_INTERVAL_SLOW = float(os.environ.get("PARAMETER_GOLF_SSH_POLL_INTERVAL_SLOW", "10"))
DEFAULT_SSH_POLL_FAST_WINDOW = float(os.environ.get("PARAMETER_GOLF_SSH_POLL_FAST_WINDOW", "60"))
DEFAULT_DATASET_VARIANT = os.environ.get("PARAMETER_GOLF_DATASET_VARIANT", "sp1024")
DEFAULT_TRAIN_SHARDS = int(os.environ.get("PARAMETER_GOLF_TRAIN_SHARDS", "80"))
DEFAULT_REMOTE_ROOT_BASE = os.environ.get("PARAMETER_GOLF_REMOTE_ROOT_BASE", "/root/parameter-golf-runner")

TERMINAL_STATUSES = {
    "success",
    "bootstrap_failed",
    "ssh_failed",
    "train_failed",
    "egress_failed",
    "delete_failed",
    "inspection_hold",
}


@dataclass
class SSHSpec:
    host: str
    user: str | None
    port: int | None
    identity_file: str | None
    raw_command: str

    def destination(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "run"


def ensure_state() -> tuple[Path, Path, Path]:
    runs_dir = STATE_ROOT / "runs"
    tmp_dir = STATE_ROOT / "tmp"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return runs_dir, tmp_dir, RESULTS_ROOT


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def notify(event: str, record: dict[str, Any], extra: str | None = None) -> None:
    pod_id = ((record.get("pod") or {}).get("id")) or "-"
    run_name = record.get("run_name", "-")
    local_dir = record.get("local_result_dir", "")
    tmux_session = ((record.get("remote") or {}).get("tmux_session")) or ""
    message = f"\a[{event}] run={run_name} pod={pod_id}"
    if tmux_session:
        message += f" TMUX={tmux_session}"
    if extra:
        message += f" {extra}"
    if local_dir:
        message += f" local={local_dir}"
    print(message, file=sys.stderr)
    sys.stderr.flush()

    notify_cmd = os.environ.get("PARAMETER_GOLF_NOTIFY_CMD")
    if not notify_cmd:
        return
    payload = {"event": event, "record": record, "extra": extra, "at": utc_now()}
    try:
        subprocess.run(
            notify_cmd,
            shell=True,
            input=json.dumps(payload),
            text=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def require_local_tools() -> None:
    required = ["runpodctl", "ssh", "scp"]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing required local tools: {', '.join(missing)}")


def run_command(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=check,
    )


def parse_pod_id(output: str) -> str:
    quoted = re.search(r'pod\s+"(?P<id>[A-Za-z0-9_-]+)"', output)
    if quoted:
        return quoted.group("id")
    bare = re.search(r"\b([a-z0-9]{10,})\b", output)
    if bare:
        return bare.group(1)
    raise RuntimeError(f"unable to parse pod id from output: {output.strip()}")


def parse_ssh_info(output: str) -> SSHSpec:
    stripped = output.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        ssh_command = payload.get("ssh_command")
        ssh_key = payload.get("ssh_key") or {}
        if not ssh_command or not payload.get("ip"):
            raise RuntimeError(f"unable to parse ssh info from output: {output.strip()}")
        return SSHSpec(
            host=str(payload["ip"]),
            user="root",
            port=int(payload["port"]) if payload.get("port") else None,
            identity_file=ssh_key.get("path"),
            raw_command=str(ssh_command),
        )
    ssh_line = None
    for line in output.splitlines():
        candidate = line.strip()
        if candidate.startswith("ssh "):
            if " -p " in candidate:
                ssh_line = candidate
                break
            ssh_line = ssh_line or candidate
    if ssh_line is None:
        raise RuntimeError(f"unable to parse ssh command from output: {output.strip()}")
    parts = shlex.split(ssh_line)
    host = None
    user = None
    port = None
    identity = None
    i = 1
    while i < len(parts):
        part = parts[i]
        if part == "-i" and i + 1 < len(parts):
            identity = parts[i + 1]
            i += 2
            continue
        if part == "-p" and i + 1 < len(parts):
            port = int(parts[i + 1])
            i += 2
            continue
        if not part.startswith("-"):
            if "@" in part:
                user, host = part.split("@", 1)
            else:
                host = part
            i += 1
            continue
        i += 1
    if host is None:
        raise RuntimeError(f"unable to parse ssh host from output: {output.strip()}")
    return SSHSpec(host=host, user=user, port=port, identity_file=identity, raw_command=ssh_line)


def ssh_base_args(spec: SSHSpec) -> list[str]:
    args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if spec.identity_file:
        args.extend(["-i", spec.identity_file])
    if spec.port is not None:
        args.extend(["-p", str(spec.port)])
    args.append(spec.destination())
    return args


def scp_base_args(spec: SSHSpec) -> list[str]:
    args = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if spec.identity_file:
        args.extend(["-i", spec.identity_file])
    if spec.port is not None:
        args.extend(["-P", str(spec.port)])
    return args


def ssh_run(spec: SSHSpec, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(ssh_base_args(spec) + [command], check=check)


def scp_to_remote(spec: SSHSpec, local_path: Path, remote_path: str) -> None:
    run_command(scp_base_args(spec) + [str(local_path), f"{spec.destination()}:{remote_path}"])


def scp_from_remote(spec: SSHSpec, remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    run_command(scp_base_args(spec) + ["-r", f"{spec.destination()}:{remote_path}/.", str(local_dir)])


def load_runs(batch_file: Path) -> list[dict[str, Any]]:
    payload = read_json(batch_file)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("batch file must contain a JSON object or list of objects")


def pod_env_for_payload(payload: dict[str, Any]) -> dict[str, str]:
    pod_env = {str(k): str(v) for k, v in payload.get("pod_env", {}).items()}
    env_overrides = {str(k): str(v) for k, v in payload.get("env_overrides", {}).items()}
    wandb_enabled = env_overrides.get("WANDB_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if wandb_enabled:
        pod_env.setdefault("WANDB_API_KEY", "{{ RUNPOD_SECRET_WANDB_API_KEY }}")
    return pod_env


def prepare_payload(
    payload: dict[str, Any],
    remote_root: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    prepared = json.loads(json.dumps(payload))
    uploads: list[dict[str, str]] = []
    local_model_path = prepared.pop("local_model_path", None)
    if local_model_path:
        local_path = Path(str(local_model_path)).expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"local_model_path does not exist: {local_path}")
        remote_model_dir = f"{remote_root}/inputs"
        remote_model_path = f"{remote_model_dir}/{local_path.name}"
        env_overrides = prepared.setdefault("env_overrides", {})
        env_overrides.setdefault("INIT_MODEL_PATH", remote_model_path)
        env_overrides.setdefault("EVAL_ONLY", "1")
        uploads.append({"local_path": str(local_path), "remote_path": remote_model_path})
    return prepared, uploads


def record_path_for(run_id: str) -> Path:
    runs_dir, _, _ = ensure_state()
    return runs_dir / f"{run_id}.json"


def save_record(record: dict[str, Any]) -> Path:
    path = record_path_for(record["run_id"])
    write_json(path, record)
    return path


def append_event(record: dict[str, Any], status: str, note: str | None = None) -> None:
    record["status"] = status
    record.setdefault("events", []).append({"at": utc_now(), "status": status, "note": note})
    save_record(record)


def make_record(payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    _, tmp_dir, results_root = ensure_state()
    slug = slugify(str(payload["run_name"]))
    stamp = compact_stamp()
    run_id = f"{stamp}_{slug}"
    remote_root = f"{DEFAULT_REMOTE_ROOT_BASE}/{run_id}"
    remote_run_dir = f"{remote_root}/run"
    prepared_payload, uploads = prepare_payload(payload, remote_root)
    payload_path = tmp_dir / f"{run_id}.payload.json"
    local_result_dir = results_root / run_id
    record = {
        "run_id": run_id,
        "run_name": prepared_payload["run_name"],
        "payload": prepared_payload,
        "created_at": utc_now(),
        "status": "created",
        "events": [{"at": utc_now(), "status": "created"}],
        "pod": {
            "name": f"pg-{slug[:35]}-{stamp.lower()}",
            "gpu_type": prepared_payload.get("gpu_id") or DEFAULT_GPU_TYPE,
            "template_id": prepared_payload.get("template_id") or DEFAULT_TEMPLATE_ID,
            "image_name": prepared_payload.get("image_name") or DEFAULT_IMAGE_NAME,
            "cloud_type": str(prepared_payload.get("cloud_type") or DEFAULT_CLOUD_TYPE).upper(),
            "env": pod_env_for_payload(prepared_payload),
        },
        "remote": {
            "root": remote_root,
            "run_dir": remote_run_dir,
            "payload_path": f"{remote_root}/payload.json",
            "tmux_session": f"pg_{slug[:40]}_{stamp[-6:].lower()}",
        },
        "local_payload_path": str(payload_path),
        "local_result_dir": str(local_result_dir),
        "git_ref": prepared_payload.get("git_ref") or DEFAULT_GIT_REF,
        "repo_url": prepared_payload.get("repo_url") or DEFAULT_REPO_URL,
        "hold_open": bool(prepared_payload.get("hold_open", False)),
        "uploads": uploads,
    }
    write_json(payload_path, prepared_payload)
    save_record(record)
    return record, payload_path


def create_pod(record: dict[str, Any]) -> str:
    cmd = [
        "runpodctl",
        "pod",
        "create",
        "--output",
        "json",
        "--name",
        record["pod"]["name"],
        "--gpu-id",
        record["pod"]["gpu_type"],
        "--gpu-count",
        "1",
        "--container-disk-in-gb",
        str(DEFAULT_CONTAINER_DISK_GB),
        "--volume-in-gb",
        str(DEFAULT_VOLUME_GB),
        "--ports",
        "22/tcp",
        "--cloud-type",
        record["pod"]["cloud_type"],
        "--ssh",
    ]
    pod_env = record["pod"].get("env") or {}
    if pod_env:
        cmd.extend(["--env", json.dumps(pod_env, separators=(",", ":"))])
    if record["pod"]["template_id"]:
        cmd.extend(["--template-id", record["pod"]["template_id"]])
    cmd.extend(["--image", record["pod"]["image_name"]])
    result = run_command(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stdout + "\n" + result.stderr).strip())
    try:
        pod_id = json.loads(result.stdout)["id"]
    except Exception:
        pod_id = parse_pod_id(result.stdout + "\n" + result.stderr)
    record["pod"]["id"] = pod_id
    record["pod"]["create_output"] = (result.stdout + "\n" + result.stderr).strip()
    append_event(record, "pod_created")
    notify("pod_created", record)
    return pod_id


def wait_for_ssh(record: dict[str, Any], timeout_seconds: int = 900) -> SSHSpec:
    pod_id = record["pod"]["id"]
    started = time.time()
    deadline = started + timeout_seconds
    last_error = None
    while time.time() < deadline:
        result = run_command(["runpodctl", "ssh", "info", pod_id], check=False)
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode == 0:
            try:
                spec = parse_ssh_info(output)
            except Exception as exc:
                last_error = str(exc)
            else:
                record["ssh"] = {
                    "host": spec.host,
                    "user": spec.user,
                    "port": spec.port,
                    "identity_file": spec.identity_file,
                    "raw_command": spec.raw_command,
                }
                append_event(record, "ssh_ready")
                notify("ssh_ready", record, extra=f'attach={attach_command(record)}')
                return spec
        else:
            last_error = output
        elapsed = time.time() - started
        interval = DEFAULT_SSH_POLL_INTERVAL_FAST if elapsed < DEFAULT_SSH_POLL_FAST_WINDOW else DEFAULT_SSH_POLL_INTERVAL_SLOW
        time.sleep(interval)
    raise RuntimeError(f"timed out waiting for ssh info for pod {pod_id}: {last_error}")


def attach_command(record: dict[str, Any]) -> str:
    ssh = record.get("ssh") or {}
    spec = SSHSpec(
        host=ssh.get("host"),
        user=ssh.get("user"),
        port=ssh.get("port"),
        identity_file=ssh.get("identity_file"),
        raw_command=ssh.get("raw_command", ""),
    )
    cmd = ssh_base_args(spec) + ["-t", f"tmux attach -t {record['remote']['tmux_session']} || tmux ls"]
    return shlex.join(cmd)


def start_remote_run(record: dict[str, Any], payload_path: Path) -> None:
    spec = SSHSpec(
        host=record["ssh"]["host"],
        user=record["ssh"]["user"],
        port=record["ssh"]["port"],
        identity_file=record["ssh"]["identity_file"],
        raw_command=record["ssh"]["raw_command"],
    )
    ssh_run(spec, f"mkdir -p {shlex.quote(record['remote']['root'])}")
    scp_to_remote(spec, payload_path, record["remote"]["payload_path"])
    for upload in record.get("uploads", []):
        remote_parent = str(Path(upload["remote_path"]).parent)
        ssh_run(spec, f"mkdir -p {shlex.quote(remote_parent)}")
        scp_to_remote(spec, Path(upload["local_path"]), upload["remote_path"])
    remote_script = REPO_ROOT / "ops" / "runpod" / "remote_bootstrap.sh"
    remote_cmd = (
        f"PARAMETER_GOLF_DATASET_VARIANT={shlex.quote(str(record['payload'].get('dataset_variant', DEFAULT_DATASET_VARIANT)))} "
        f"PARAMETER_GOLF_TRAIN_SHARDS={shlex.quote(str(record['payload'].get('train_shards', DEFAULT_TRAIN_SHARDS)))} "
        f"bash -s -- "
        f"{shlex.quote(record['repo_url'])} "
        f"{shlex.quote(record['git_ref'])} "
        f"{shlex.quote(record['remote']['root'])} "
        f"{shlex.quote(record['remote']['payload_path'])} "
        f"{shlex.quote(record['remote']['run_dir'])} "
        f"{shlex.quote(record['remote']['tmux_session'])}"
    )
    result = subprocess.run(
        ssh_base_args(spec) + [remote_cmd],
        stdin=remote_script.open("r", encoding="utf-8"),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote bootstrap failed: {(result.stdout + result.stderr).strip()}")
    record["remote"]["bootstrap_output"] = (result.stdout + "\n" + result.stderr).strip()
    append_event(record, "training_started", note=attach_command(record))
    notify("training_started", record, extra=f'tmux={record["remote"]["tmux_session"]}')


def remote_exit_code(record: dict[str, Any]) -> tuple[bool, int | None, str]:
    spec = SSHSpec(
        host=record["ssh"]["host"],
        user=record["ssh"]["user"],
        port=record["ssh"]["port"],
        identity_file=record["ssh"]["identity_file"],
        raw_command=record["ssh"]["raw_command"],
    )
    exit_file = f"{record['remote']['run_dir']}/.runpod_exit_code"
    status_file = f"{record['remote']['run_dir']}/.runpod_status"
    result = ssh_run(
        spec,
        f"if [ -f {shlex.quote(exit_file)} ]; then cat {shlex.quote(exit_file)}; fi; "
        f"if [ -f {shlex.quote(status_file)} ]; then printf '\\n'; cat {shlex.quote(status_file)}; fi",
        check=False,
    )
    if result.returncode != 0:
        return False, None, ""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False, None, ""
    exit_code = int(lines[0]) if re.fullmatch(r"-?\d+", lines[0]) else None
    status = lines[1] if len(lines) > 1 else ""
    return exit_code is not None, exit_code, status


REQUIRED_EGRESS_FILES = ["metrics.json", "config.json", "stdout.log", "artifact_summary.json"]
MAX_EGRESS_ATTEMPTS = 3


def verify_egress(local_result_dir: Path) -> list[str]:
    """Check egressed results for completeness. Returns list of problems (empty = OK)."""
    problems: list[str] = []
    for name in REQUIRED_EGRESS_FILES:
        path = local_result_dir / name
        if not path.is_file():
            problems.append(f"missing required file: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"empty required file: {name}")
    summary_path = local_result_dir / "artifact_summary.json"
    if summary_path.is_file() and summary_path.stat().st_size > 0:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"corrupt artifact_summary.json: {exc}")
            return problems
        for artifact in summary.get("artifacts", []):
            remote_path = artifact.get("path", "")
            expected_bytes = artifact.get("bytes")
            name = Path(remote_path).name
            local_path = local_result_dir / name
            if not local_path.is_file():
                problems.append(f"missing artifact: {name}")
            elif expected_bytes is not None and local_path.stat().st_size != expected_bytes:
                problems.append(
                    f"size mismatch for {name}: expected {expected_bytes} got {local_path.stat().st_size}"
                )
    return problems


def local_egress_manifest(local_result_dir: Path) -> dict[str, int]:
    """Return {relative_name: size_bytes} for all files present in the local result dir."""
    manifest: dict[str, int] = {}
    if not local_result_dir.is_dir():
        return manifest
    for path in local_result_dir.rglob("*"):
        if path.is_file():
            manifest[str(path.relative_to(local_result_dir))] = path.stat().st_size
    return manifest


class EgressError(RuntimeError):
    """Raised when egress fails, carrying structured diagnostics for the agent."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        problems: list[str],
        local_manifest: dict[str, int],
        pod_reachable: bool | None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.problems = problems
        self.local_manifest = local_manifest
        self.pod_reachable = pod_reachable

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "attempts": self.attempts,
            "problems": self.problems,
            "local_manifest": self.local_manifest,
            "pod_reachable": self.pod_reachable,
        }


def egress_results(record: dict[str, Any]) -> None:
    spec = SSHSpec(
        host=record["ssh"]["host"],
        user=record["ssh"]["user"],
        port=record["ssh"]["port"],
        identity_file=record["ssh"]["identity_file"],
        raw_command=record["ssh"]["raw_command"],
    )
    local_result_dir = Path(record["local_result_dir"])
    local_result_dir.mkdir(parents=True, exist_ok=True)
    last_problems: list[str] = []
    for attempt in range(1, MAX_EGRESS_ATTEMPTS + 1):
        try:
            scp_from_remote(spec, record["remote"]["run_dir"], local_result_dir)
        except Exception as exc:
            last_problems = [f"scp failed: {exc}"]
            if attempt < MAX_EGRESS_ATTEMPTS:
                time.sleep(2 * attempt)
                continue
            break
        last_problems = verify_egress(local_result_dir)
        if not last_problems:
            return
        if attempt < MAX_EGRESS_ATTEMPTS:
            time.sleep(2 * attempt)
    pod_reachable: bool | None = None
    try:
        ssh_run(spec, "true", check=True)
        pod_reachable = True
    except Exception:
        pod_reachable = False
    raise EgressError(
        f"egress verification failed after {MAX_EGRESS_ATTEMPTS} attempts: "
        + "; ".join(last_problems),
        attempts=MAX_EGRESS_ATTEMPTS,
        problems=last_problems,
        local_manifest=local_egress_manifest(local_result_dir),
        pod_reachable=pod_reachable,
    )


def delete_pod(record: dict[str, Any]) -> None:
    result = run_command(["runpodctl", "pod", "delete", record["pod"]["id"]], check=False)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        append_event(record, "delete_failed", note=output)
        notify("delete_failed", record, extra=output)
        return
    record["pod"]["delete_output"] = output
    record["pod"]["deleted_at"] = utc_now()
    record.setdefault("events", []).append({"at": utc_now(), "status": "pod_deleted", "note": output})
    save_record(record)


def launch_run(payload: dict[str, Any]) -> Path:
    require_local_tools()
    record, payload_path = make_record(payload)
    try:
        create_pod(record)
        spec = wait_for_ssh(record)
        record["attach_command"] = attach_command(record)
        save_record(record)
        start_remote_run(record, payload_path)
    except Exception as exc:
        append_event(record, "bootstrap_failed", note=str(exc))
        notify("bootstrap_failed", record, extra=str(exc))
        pod_id = (record.get("pod") or {}).get("id")
        if pod_id:
            delete_pod(record)
        return record_path_for(record["run_id"])
    return record_path_for(record["run_id"])


def submit_runs(batch_file: Path, watch: bool) -> None:
    for payload in load_runs(batch_file):
        path = launch_run(payload)
        print(f"{payload['run_name']}: {path}")
    if watch:
        watch_runs(watch=True, interval=DEFAULT_POLL_INTERVAL)


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["run_id", "run_name", "pod_id", "status", "tmux_session", "gpu", "local_result_dir"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    lines = [
        " | ".join(header.ljust(widths[header]) for header in headers),
        "-|-".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def summarize_runs() -> None:
    runs_dir, _, _ = ensure_state()
    rows = []
    for path in sorted(runs_dir.glob("*.json")):
        record = read_json(path)
        rows.append(
            {
                "run_name": record.get("run_name", ""),
                "run_id": record.get("run_id", ""),
                "pod_id": ((record.get("pod") or {}).get("id")) or "",
                "status": record.get("status", ""),
                "tmux_session": ((record.get("remote") or {}).get("tmux_session")) or "",
                "gpu": ((record.get("pod") or {}).get("gpu_type")) or "",
                "local_result_dir": record.get("local_result_dir", ""),
            }
        )
    print(render_table(rows))


def watch_runs(*, watch: bool, interval: float) -> None:
    runs_dir, _, _ = ensure_state()
    last_render = None
    while True:
        pending = 0
        rows = []
        for path in sorted(runs_dir.glob("*.json")):
            record = read_json(path)
            status = record.get("status", "")
            if status in TERMINAL_STATUSES:
                rows.append(
                    {
                        "run_id": record.get("run_id", ""),
                        "run_name": record.get("run_name", ""),
                        "pod_id": ((record.get("pod") or {}).get("id")) or "",
                        "status": status,
                        "tmux_session": ((record.get("remote") or {}).get("tmux_session")) or "",
                        "gpu": ((record.get("pod") or {}).get("gpu_type")) or "",
                        "local_result_dir": record.get("local_result_dir", ""),
                    }
                )
                continue
            pending += 1
            try:
                completed, exit_code, remote_status = remote_exit_code(record)
            except Exception as exc:
                append_event(record, "ssh_failed", note=str(exc))
                notify("ssh_failed", record, extra=str(exc))
                completed = False
                exit_code = None
                remote_status = ""
            if completed:
                try:
                    egress_results(record)
                except EgressError as exc:
                    record["egress_diagnostics"] = exc.to_dict()
                    save_record(record)
                    append_event(record, "egress_failed", note=str(exc))
                    notify("egress_failed", record, extra=json.dumps(exc.to_dict()))
                except Exception as exc:
                    append_event(record, "egress_failed", note=str(exc))
                    notify("egress_failed", record, extra=str(exc))
                    rows.append(
                        {
                            "run_id": record.get("run_id", ""),
                            "run_name": record.get("run_name", ""),
                            "pod_id": ((record.get("pod") or {}).get("id")) or "",
                            "status": "egress_failed",
                            "tmux_session": ((record.get("remote") or {}).get("tmux_session")) or "",
                            "gpu": ((record.get("pod") or {}).get("gpu_type")) or "",
                            "local_result_dir": record.get("local_result_dir", ""),
                        }
                    )
                    continue
                final_status = "success" if exit_code == 0 else "train_failed"
                append_event(record, final_status, note=remote_status)
                notify(final_status, record)
                if not record.get("hold_open"):
                    delete_pod(record)
                else:
                    append_event(record, "inspection_hold")
                    notify("inspection_hold", record, extra=record.get("attach_command"))
            rows.append(
                {
                    "run_id": record.get("run_id", ""),
                    "run_name": record.get("run_name", ""),
                    "pod_id": ((record.get("pod") or {}).get("id")) or "",
                    "status": read_json(path).get("status", ""),
                    "tmux_session": ((record.get("remote") or {}).get("tmux_session")) or "",
                    "gpu": ((record.get("pod") or {}).get("gpu_type")) or "",
                    "local_result_dir": record.get("local_result_dir", ""),
                }
            )
        render = render_table(rows)
        if render != last_render:
            print(render)
            last_render = render
        if not watch or pending == 0:
            break
        time.sleep(interval)


def print_attach(run_id: str) -> None:
    record = read_json(record_path_for(run_id))
    print(record.get("attach_command") or attach_command(record))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Runpod Parameter Golf Pod runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Create pods and start runs")
    submit_parser.add_argument("--batch-file", required=True, type=Path)
    submit_parser.add_argument("--watch", action="store_true")

    watch_parser = subparsers.add_parser("watch", help="Watch active runs")
    watch_parser.add_argument("--watch", action="store_true")
    watch_parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL)

    attach_parser = subparsers.add_parser("attach", help="Print the tmux attach command for a run")
    attach_parser.add_argument("--run-id", required=True)

    subparsers.add_parser("summarize", help="Print a status table")

    args = parser.parse_args()
    if args.command == "submit":
        submit_runs(args.batch_file, watch=args.watch)
    elif args.command == "watch":
        watch_runs(watch=args.watch, interval=args.interval)
    elif args.command == "attach":
        print_attach(args.run_id)
    elif args.command == "summarize":
        summarize_runs()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
