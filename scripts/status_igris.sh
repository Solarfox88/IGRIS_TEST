#!/usr/bin/env bash
# status_igris.sh — Check IGRIS_GPT server status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$REPO_DIR/logs/igris.pid"
LOG_FILE="$REPO_DIR/logs/igris.log"
PORT="${IGRIS_PORT:-7778}"

echo "=== IGRIS_GPT Status ==="

# PID check
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "PID:    $PID (running)"
  else
    echo "PID:    $PID (NOT running — stale PID file)"
  fi
else
  echo "PID:    not found (server not started via scripts)"
fi

echo "Port:   $PORT"

# Health check
echo ""
echo "--- Health ---"
if curl -sf "http://127.0.0.1:$PORT/api/health" 2>/dev/null; then
  echo ""
else
  echo "Health endpoint unreachable (server may not be running)"
fi

echo ""
echo "--- Readiness ---"
if curl -sf "http://127.0.0.1:$PORT/api/readiness" 2>/dev/null; then
  echo ""
else
  echo "Readiness endpoint unreachable"
fi

# Last log lines
echo ""
echo "--- Last 10 log lines ---"
if [ -f "$LOG_FILE" ]; then
  tail -10 "$LOG_FILE"
else
  echo "(no log file found)"
fi
