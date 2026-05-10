#!/usr/bin/env bash
# start_igris.sh — Start the IGRIS_GPT server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/.venv"
PID_FILE="$REPO_DIR/logs/igris.pid"
LOG_FILE="$REPO_DIR/logs/igris.log"
HOST="${IGRIS_HOST:-0.0.0.0}"
PORT="${IGRIS_PORT:-7778}"

# Ensure directories
mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/.igris/tasks"
mkdir -p "$REPO_DIR/.igris/reports"
mkdir -p "$REPO_DIR/.igris/timeline"
mkdir -p "$REPO_DIR/.igris/memory"

# Check for existing instance
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -a -p "$OLD_PID" >/dev/null 2>&1; then
      echo "IGRIS_GPT is already running (PID $OLD_PID)."
      echo "Use: bash scripts/stop_igris.sh  to stop it first."
      exit 1
    fi
    echo "PID file points to non-listener process ($OLD_PID). Removing stale PID file..."
    rm -f "$PID_FILE"
  else
    echo "Stale PID file found. Removing..."
    rm -f "$PID_FILE"
  fi
fi

# Activate venv
if [ -d "$VENV" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ -f "$REPO_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/.venv/bin/activate"
fi

# Load .env if present
if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

echo "=== Starting IGRIS_GPT ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log:  $LOG_FILE"
echo "PID:  $PID_FILE"

cd "$REPO_DIR"
nohup python -m uvicorn igris.web.server:app \
  --host "$HOST" \
  --port "$PORT" \
  --factory \
  --log-level info \
  >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

sleep 2

LISTENER_PID=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN -n -P 2>/dev/null | head -n1 || true)
if [ -n "$LISTENER_PID" ]; then
  SERVER_PID="$LISTENER_PID"
  echo "$SERVER_PID" > "$PID_FILE"
fi

if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo ""
  echo "IGRIS_GPT started (PID $SERVER_PID)"
  echo "URL: http://$HOST:$PORT"
  echo ""
  echo "Commands:"
  echo "  bash scripts/status_igris.sh   — check status"
  echo "  bash scripts/stop_igris.sh     — stop server"
  echo "  tail -f $LOG_FILE              — follow logs"
else
  echo "ERROR: Server failed to start. Check $LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
fi
