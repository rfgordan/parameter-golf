#!/usr/bin/env bash
# Appends controller notifications as JSON lines to the event log.
# Intended for use as PARAMETER_GOLF_NOTIFY_CMD.
# Reads JSON payload from stdin.
set -euo pipefail

EVENT_LOG="${PARAMETER_GOLF_EVENT_LOG:-.runpod/controller/events.jsonl}"
mkdir -p "$(dirname "$EVENT_LOG")"
cat >> "$EVENT_LOG"
printf '\n' >> "$EVENT_LOG"
