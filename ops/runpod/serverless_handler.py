from __future__ import annotations

from runpod import serverless

from ops.runpod.worker import handle_job


def handler(job):
    return handle_job(job["input"])


if __name__ == "__main__":
    serverless.start({"handler": handler})
