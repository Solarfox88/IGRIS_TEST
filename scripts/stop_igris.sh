#!/usr/bin/env bash
# stop_igris.sh — Stop the IGRIS_GPT server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$REPO_DIR/logs/igris.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found. IGRIS_GPT does not appear to be running."
  exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping IGRIS_GPT (PID $PID)..."
  kill "$PID"
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process still alive. Sending SIGKILL..."
    kill -9 "$PID" 2>/dev/null || true
  fi
  echo "IGRIS_GPT stopped."
else
  echo "Process $PID is not running (stale PID file)."
fi

rm -f "$PID_FILE"
