from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATE_ROOT = Path(os.environ.get("PARAMETER_GOLF_CONTROLLER_ROOT", ".runpod/controller"))
DEFAULT_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_state() -> tuple[Path, Path]:
    jobs_dir = STATE_ROOT / "jobs"
    runs_dir = STATE_ROOT / "runs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir, runs_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def api_request(endpoint_id: str, suffix: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    url = f"{DEFAULT_API_BASE}/{endpoint_id}{suffix}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod API {exc.code}: {detail}") from exc


def save_run_request(payload: dict[str, Any], submitted: dict[str, Any]) -> Path:
    _, runs_dir = ensure_state()
    record = {
        "run_name": payload["run_name"],
        "payload": payload,
        "submitted_at": utc_now(),
        "submission": submitted,
    }
    path = runs_dir / f"{payload['run_name']}.json"
    write_json(path, record)
    return path


def load_runs(batch_file: Path) -> list[dict[str, Any]]:
    payload = read_json(batch_file)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("batch file must contain a JSON object or list of objects")


def submit_runs(endpoint_id: str, batch_file: Path) -> None:
    for payload in load_runs(batch_file):
        response = api_request(endpoint_id, "/run", {"input": payload})
        record_path = save_run_request(payload, response)
        print(f"{payload['run_name']}: {record_path}")


def refresh_status(endpoint_id: str, record_path: Path) -> dict[str, Any]:
    record = read_json(record_path)
    job_id = record["submission"]["id"]
    status = api_request(endpoint_id, f"/status/{job_id}")
    record["last_checked_at"] = utc_now()
    record["status"] = status
    write_json(record_path, record)
    return record


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["run_name", "job_id", "status", "val_bpb", "elapsed_seconds"]
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


def poll_runs(endpoint_id: str, watch: bool, interval: float) -> None:
    _, runs_dir = ensure_state()
    record_paths = sorted(runs_dir.glob("*.json"))
    if not record_paths:
        raise RuntimeError("no run records found under .runpod/controller/runs")
    while True:
        pending = 0
        rows = []
        for path in record_paths:
            record = refresh_status(endpoint_id, path)
            status = (record.get("status") or {}).get("status", "UNKNOWN")
            output = (record.get("status") or {}).get("output") or {}
            rows.append(
                {
                    "run_name": record["run_name"],
                    "job_id": record["submission"]["id"],
                    "status": status,
                    "val_bpb": output.get("final_roundtrip_val_bpb", ""),
                    "elapsed_seconds": output.get("elapsed_seconds", ""),
                }
            )
            if status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                pending += 1
        print(render_table(rows))
        if not watch or pending == 0:
            break
        time.sleep(interval)


def summarize_runs() -> None:
    _, runs_dir = ensure_state()
    rows = []
    for path in sorted(runs_dir.glob("*.json")):
        record = read_json(path)
        status = record.get("status") or {}
        output = status.get("output") or {}
        rows.append(
            {
                "run_name": record["run_name"],
                "job_id": record["submission"]["id"],
                "status": status.get("status", "UNKNOWN"),
                "val_bpb": output.get("final_roundtrip_val_bpb", ""),
                "elapsed_seconds": output.get("elapsed_seconds", ""),
            }
        )
    print(render_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Runpod Parameter Golf jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit one or more jobs")
    submit_parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    submit_parser.add_argument("--batch-file", required=True, type=Path)

    poll_parser = subparsers.add_parser("poll", help="Poll job status")
    poll_parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    poll_parser.add_argument("--watch", action="store_true")
    poll_parser.add_argument("--interval", type=float, default=10.0)

    subparsers.add_parser("summarize", help="Print a markdown-style summary table")

    args = parser.parse_args()
    if args.command in {"submit", "poll"} and not args.endpoint_id:
        raise RuntimeError("RUNPOD_ENDPOINT_ID or --endpoint-id is required")
    if args.command == "submit":
        submit_runs(args.endpoint_id, args.batch_file)
    elif args.command == "poll":
        poll_runs(args.endpoint_id, args.watch, args.interval)
    elif args.command == "summarize":
        summarize_runs()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
