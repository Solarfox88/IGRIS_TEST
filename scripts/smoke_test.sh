#!/usr/bin/env bash
# smoke_test.sh — Quick verification that IGRIS_GPT is working
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${IGRIS_PORT:-7778}"
PASS=0
FAIL=0

check() {
  local desc="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    PASS=$((PASS+1))
    echo "PASS: $desc"
  else
    FAIL=$((FAIL+1))
    echo "FAIL: $desc"
  fi
}

echo "=== IGRIS_GPT Smoke Test ==="

# Python import
check "Python import igris.web.server" python -c "from igris.web.server import create_app"
check "Python create_app()" python -c "from igris.web.server import create_app; app = create_app(); print(app.title)"
check "Python import safety" python -c "from igris.core.safety import redact_secrets"
check "Python import task_engine" python -c "from igris.core.task_engine import TaskEngine"

# Tests (optional, only if dev deps available)
if python -c "import pytest" 2>/dev/null; then
  echo ""
  echo "--- Running pytest ---"
  cd "$REPO_DIR"
  if python -m pytest -q; then
    PASS=$((PASS+1))
    echo "PASS: pytest"
  else
    FAIL=$((FAIL+1))
    echo "FAIL: pytest"
  fi
else
  echo "SKIP: pytest (not installed)"
fi

# Server health (only if running)
echo ""
echo "--- Server checks (if running) ---"
if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  check "GET /api/health" curl -sf "http://127.0.0.1:$PORT/api/health"
  check "GET /api/readiness" curl -sf "http://127.0.0.1:$PORT/api/readiness"
  check "GET /.well-known/agent-card.json" curl -sf "http://127.0.0.1:$PORT/.well-known/agent-card.json"
  check "GET /api/tasks" curl -sf "http://127.0.0.1:$PORT/api/tasks"
  check "GET /api/a2a/capabilities" curl -sf "http://127.0.0.1:$PORT/api/a2a/capabilities"
else
  echo "SKIP: Server not running on port $PORT"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit "$FAIL"
