#!/usr/bin/env bash
# Blocks until a new event appears in the event log, then prints all new events and exits.
# Run this as a background task — when it returns, new state is available.
#
# Usage: wait_for_event.sh [--timeout SECONDS]
set -euo pipefail

EVENT_LOG="${PARAMETER_GOLF_EVENT_LOG:-.runpod/controller/events.jsonl}"
POLL_INTERVAL="${PARAMETER_GOLF_EVENT_POLL_INTERVAL:-5}"
TIMEOUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$EVENT_LOG")"
touch "$EVENT_LOG"

INITIAL_LINES=$(wc -l < "$EVENT_LOG" | tr -d ' ')
STARTED=$(date +%s)

while true; do
    CURRENT_LINES=$(wc -l < "$EVENT_LOG" | tr -d ' ')
    if [ "$CURRENT_LINES" -gt "$INITIAL_LINES" ]; then
        tail -n +"$((INITIAL_LINES + 1))" "$EVENT_LOG"
        exit 0
    fi
    if [ -n "$TIMEOUT" ]; then
        ELAPSED=$(( $(date +%s) - STARTED ))
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo '{"event": "timeout", "elapsed": '"$ELAPSED"'}'
            exit 0
        fi
    fi
    sleep "$POLL_INTERVAL"
done
