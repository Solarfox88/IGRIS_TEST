#!/usr/bin/env bash
set -euo pipefail

PID_FILE="igris.pid"
LOG_FILE="logs/igris.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "[IGRIS] Server is already running (PID $(cat "$PID_FILE"))."
  exit 0
fi

echo "[IGRIS] Starting IGRIS server..."
source .venv/bin/activate
nohup uvicorn igris.web.server:create_app --factory --host 0.0.0.0 --port 7778 \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[IGRIS] Server started (PID $(cat "$PID_FILE")). Logs: $LOG_FILE"