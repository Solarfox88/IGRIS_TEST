#!/usr/bin/env bash
# stop_igris.sh — Stop the IGRIS_GPT server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$REPO_DIR/logs/igris.pid"
PORT="${IGRIS_PORT:-7778}"

stop_pid() {
  local pid="$1"
  echo "Stopping IGRIS_GPT (PID $pid)..."
  kill "$pid"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "Process still alive. Sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null || true
  fi
  echo "IGRIS_GPT stopped."
}

LISTENER_PID=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN -n -P 2>/dev/null | head -n1 || true)

if [ ! -f "$PID_FILE" ]; then
  if [ -n "$LISTENER_PID" ]; then
    echo "No PID file found, but listener PID $LISTENER_PID is active on port $PORT."
    stop_pid "$LISTENER_PID"
  else
    echo "No PID file found. IGRIS_GPT does not appear to be running."
  fi
  exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
  stop_pid "$PID"
else
  echo "Process $PID is not running (stale PID file)."
  if [ -n "$LISTENER_PID" ]; then
    echo "Stopping listener PID $LISTENER_PID on port $PORT..."
    stop_pid "$LISTENER_PID"
  fi
fi

rm -f "$PID_FILE"
