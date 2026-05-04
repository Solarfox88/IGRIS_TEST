#!/usr/bin/env bash
set -euo pipefail

PID_FILE="igris.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[IGRIS] PID file not found; server may not be running."
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" >/dev/null 2>&1; then
  echo "[IGRIS] Stopping IGRIS server (PID $PID)..."
  kill "$PID"
  rm -f "$PID_FILE"
  echo "[IGRIS] Server stopped."
else
  echo "[IGRIS] Process $PID not found. Removing stale PID file."
  rm -f "$PID_FILE"
fi