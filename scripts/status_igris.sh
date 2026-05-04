#!/usr/bin/env bash
set -euo pipefail

PID_FILE="igris.pid"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" >/dev/null 2>&1; then
    echo "[IGRIS] Server running (PID $PID)"
  else
    echo "[IGRIS] PID file present but process $PID is not running."
  fi
else
  echo "[IGRIS] Server is not running."
fi

echo "[IGRIS] Checking API status..."
if command -v curl >/dev/null 2>&1; then
  curl -s http://127.0.0.1:7778/api/status || true
else
  echo "curl not installed"
fi

echo "[IGRIS] Tail of logs:"
tail -n 10 logs/igris.log 2>/dev/null || echo "No logs found."