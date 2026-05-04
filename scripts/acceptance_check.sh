#!/usr/bin/env bash
# IGRIS_GPT Acceptance Check — automated subset of HUMAN_ACCEPTANCE_TEST.md
# Runs checks that can be automated without a browser.
# Exit code 0 = all checks pass.

set -euo pipefail

PASS=0
FAIL=0
TOTAL=0

check() {
    local name="$1"
    local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== IGRIS_GPT Acceptance Check ==="
echo ""

# 1. pytest
echo "--- Check 1: pytest ---"
if python -m pytest -q --tb=no -q 2>/dev/null | tail -1 | grep -q "passed"; then
    check "pytest green" 0
else
    check "pytest green" 1
fi

# 2. Start server (background)
echo "--- Check 2: Start server ---"
python -c "from igris.web.server import create_app, run_app; import threading; app=create_app(); t=threading.Thread(target=run_app, args=(app,), daemon=True); t.start()" &
SERVER_PID=$!
sleep 3

# 3. Health check
echo "--- Check 3: Health checks ---"
HC=$(curl -sf http://localhost:7778/api/health 2>/dev/null || echo "FAIL")
if echo "$HC" | grep -q "status"; then
    check "GET /api/health" 0
else
    check "GET /api/health" 1
fi

RC=$(curl -sf http://localhost:7778/api/readiness 2>/dev/null || echo "FAIL")
if echo "$RC" | grep -q "ready\|status"; then
    check "GET /api/readiness" 0
else
    check "GET /api/readiness" 1
fi

# 4. Chat
echo "--- Check 4: Chat ---"
CHAT=$(curl -sf -X POST http://localhost:7778/api/chat/stream \
    -H "Content-Type: application/json" \
    -d '{"message": "hello"}' 2>/dev/null || echo "FAIL")
if echo "$CHAT" | grep -q "data:"; then
    check "Chat stream response" 0
else
    check "Chat stream response" 1
fi

# 5. Mission lifecycle
echo "--- Check 5: Mission lifecycle ---"
MISSION=$(curl -sf -X POST http://localhost:7778/api/missions \
    -H "Content-Type: application/json" \
    -d '{"title": "Acceptance test", "description": "1. Check system\n2. Verify output"}' 2>/dev/null || echo "FAIL")
if echo "$MISSION" | grep -q "id"; then
    check "Create mission" 0
    MID=$(echo "$MISSION" | python -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")
else
    check "Create mission" 1
    MID=""
fi

if [ -n "$MID" ]; then
    PLAN=$(curl -sf -X POST "http://localhost:7778/api/missions/$MID/plan?mode=deterministic" 2>/dev/null || echo "FAIL")
    if echo "$PLAN" | grep -q "deterministic"; then
        check "Plan deterministic" 0
    else
        check "Plan deterministic" 1
    fi

    EXPLAIN=$(curl -sf "http://localhost:7778/api/missions/$MID/plan/explain" 2>/dev/null || echo "FAIL")
    if echo "$EXPLAIN" | grep -q "explanation"; then
        check "Plan explain" 0
    else
        check "Plan explain" 1
    fi

    MAT=$(curl -sf -X POST "http://localhost:7778/api/missions/$MID/materialize-tasks" 2>/dev/null || echo "FAIL")
    if echo "$MAT" | grep -q "task_ids\|active"; then
        check "Materialize tasks" 0
    else
        check "Materialize tasks" 1
    fi
fi

# 6. Loop step
echo "--- Check 6: Loop step ---"
LOOP=$(curl -sf -X POST http://localhost:7778/api/loop/step 2>/dev/null || echo "FAIL")
if echo "$LOOP" | grep -q "step_number\|selected_task\|stop_reason"; then
    check "Loop step" 0
else
    check "Loop step" 1
fi

# 7. Diagnostics
echo "--- Check 7: Diagnostics ---"
DIAG=$(curl -sf http://localhost:7778/api/diagnostics 2>/dev/null || echo "FAIL")
if echo "$DIAG" | grep -q "starvation\|diagnostics\|issues"; then
    check "Diagnostics" 0
else
    check "Diagnostics" 1
fi

# 8. Memory analysis
echo "--- Check 8: Memory analysis ---"
MEM=$(curl -sf -X POST http://localhost:7778/api/memory/analyze 2>/dev/null || echo "FAIL")
if echo "$MEM" | grep -q "advisory_only"; then
    check "Memory analysis" 0
else
    check "Memory analysis" 1
fi

LESSONS=$(curl -sf http://localhost:7778/api/memory/lessons 2>/dev/null || echo "FAIL")
if echo "$LESSONS" | grep -q "advisory_only"; then
    check "Memory lessons" 0
else
    check "Memory lessons" 1
fi

# 9. Vast.ai gates
echo "--- Check 9: Vast.ai gates ---"
EST=$(curl -sf -X POST http://localhost:7778/api/vastai/estimate \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek-r1:32b", "hours": 1}' 2>/dev/null || echo "FAIL")
if echo "$EST" | grep -q "cost\|total"; then
    check "Vast.ai estimate" 0
else
    check "Vast.ai estimate" 1
fi

PROV=$(curl -s -X POST http://localhost:7778/api/vastai/provision \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek-r1:32b"}' 2>/dev/null || echo "FAIL")
if echo "$PROV" | grep -q "error\|approval\|required"; then
    check "Vast.ai provision rejected without approval" 0
else
    check "Vast.ai provision rejected without approval" 1
fi

# 10. No secrets in responses
echo "--- Check 10: No secrets ---"
CONFIG=$(curl -sf http://localhost:7778/api/vastai/config 2>/dev/null || echo "FAIL")
if echo "$CONFIG" | grep -qE "sk-[a-zA-Z0-9]{20}|ghp_[a-zA-Z0-9]{30}"; then
    check "No secrets in vastai config" 1
else
    check "No secrets in vastai config" 0
fi

# 11. Git status
echo "--- Check 11: Git status ---"
GIT_DIRTY=$(git status --porcelain 2>/dev/null | grep -v "^??" | head -5)
if [ -z "$GIT_DIRTY" ]; then
    check "Git status clean (tracked files)" 0
else
    check "Git status clean (tracked files)" 1
fi

# Cleanup
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Results ==="
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
else
    echo "All acceptance checks passed."
    exit 0
fi
