# Human Acceptance Test — IGRIS_GPT v0.4

Step-by-step checklist to verify IGRIS_GPT is installed and fully operational on a fresh machine.

**Expected time:** 15-20 minutes
**Prerequisites:** Python 3.12+, git, curl

---

## 1. Clone and Install

```bash
git clone https://github.com/Solarfox88/IGRIS_GPT.git
cd IGRIS_GPT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Pass criteria:** no errors during install.

---

## 2. Run Tests

```bash
python -m pytest -q
```

**Pass criteria:** 800+ tests pass, 0 failures.

---

## 3. Start Server

```bash
bash scripts/start_igris.sh
# Wait a few seconds for startup
```

**Pass criteria:** server starts without errors.

---

## 4. Health Check

```bash
curl -s http://localhost:7778/api/health | python -m json.tool
curl -s http://localhost:7778/api/readiness | python -m json.tool
curl -s http://localhost:7778/api/status | python -m json.tool
```

**Pass criteria:** all return 200 with JSON response.

---

## 5. UI Load

Open `http://localhost:7778` in a browser.

**Pass criteria:**
- Page loads without errors
- 14+ tabs visible (Tasks, Chat, Git, Mission, Loop, Safety, Cost, Reports, Timeline, Memory, Diagnostics, Patches, Agent, State)
- No JS console errors blocking functionality

---

## 6. Chat — Local/Fallback

```bash
curl -s -X POST http://localhost:7778/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is IGRIS_GPT?"}' 
```

**Pass criteria:**
- Response is SSE stream (text/event-stream)
- Contains `data:` lines with content chunks and metadata
- Final chunk includes `provider`, `model`, `fallback_used`, `latency_ms`
- If Ollama running: `provider` = "ollama", `model` = "phi4-mini"
- If Ollama not running: `provider` = "deterministic", `fallback_used` = true
- No crash either way

---

## 7. Create Mission

```bash
curl -s -X POST http://localhost:7778/api/missions \
  -H "Content-Type: application/json" \
  -d '{"title": "Fix login bug", "description": "1. Analyze auth flow\n2. Fix validation\n3. Add tests"}' | python -m json.tool
```

Save the returned `id` as `$MISSION_ID`.

**Pass criteria:** returns mission object with id, title, status="created".

---

## 8. Plan — Deterministic Mode

```bash
curl -s -X POST "http://localhost:7778/api/missions/$MISSION_ID/plan?mode=deterministic" | python -m json.tool
```

**Pass criteria:**
- `planning.mode` = "deterministic"
- `planning.fallback_used` = false
- `mission.steps` has 3 steps
- Each step has `success_criteria` and `risk`

---

## 9. Plan — LLM Safe Schema (with fallback)

```bash
curl -s -X POST "http://localhost:7778/api/missions/$MISSION_ID/plan?mode=auto" | python -m json.tool
```

**Pass criteria:**
- Returns plan (LLM or deterministic fallback)
- `planning.validation.valid` = true
- No secrets in response
- Steps have `success_criteria` and `risk`

---

## 10. Plan Explanation

```bash
curl -s "http://localhost:7778/api/missions/$MISSION_ID/plan/explain" | python -m json.tool
```

**Pass criteria:**
- `step_count` >= 1
- `explanation` present
- `max_risk` in (low, medium, high)
- No secrets in response

---

## 11. Materialize Tasks

```bash
curl -s -X POST "http://localhost:7778/api/missions/$MISSION_ID/materialize-tasks" | python -m json.tool
```

**Pass criteria:**
- `status` = "active"
- `task_ids` has entries
- Tasks visible at `/api/tasks`

---

## 12. Run Loop — 1 Step

```bash
curl -s -X POST http://localhost:7778/api/loop/step | python -m json.tool
```

**Pass criteria:**
- Returns step result with `selected_task` or `stop_reason`
- No crash
- No uncontrolled execution

---

## 13. Create Patch Proposal

```bash
curl -s -X POST http://localhost:7778/api/patches/propose \
  -H "Content-Type: application/json" \
  -d '{"file_path": "docs/test_patch.md", "original": "", "proposed": "# Test\nThis is a test patch.", "description": "Add test doc"}' | python -m json.tool
```

**Pass criteria:** returns proposal with id, status, diff preview.

---

## 14. Validate Patch

```bash
# Use the proposal ID from step 13
curl -s -X POST "http://localhost:7778/api/patches/$PATCH_ID/validate" | python -m json.tool
```

**Pass criteria:**
- Validation result with safety checks
- No secrets in proposal
- No path traversal
- No binary file

---

## 15. Decision Report

```bash
curl -s http://localhost:7778/api/decision-reports | python -m json.tool
```

**Pass criteria:**
- Returns list of reports (may be empty if loop hasn't run enough)
- Reports contain `selected_task`, `rejected_candidates`, `safety_decisions`

---

## 16. Diagnostics

```bash
curl -s http://localhost:7778/api/diagnostics | python -m json.tool
curl -s http://localhost:7778/api/diagnostics/summary | python -m json.tool
```

**Pass criteria:**
- Returns diagnostic data (starvation, blocked, family health)
- No crash even with empty data

---

## 17. Memory Analysis

```bash
curl -s -X POST http://localhost:7778/api/memory/analyze | python -m json.tool
curl -s http://localhost:7778/api/memory/analysis | python -m json.tool
curl -s http://localhost:7778/api/memory/lessons | python -m json.tool
```

**Pass criteria:**
- All return `advisory_only: true`
- No secrets in output
- Deterministic analysis present even without LLM

---

## 18. GitHub PR — Prepare (Gated)

```bash
# Without approval — should be rejected
curl -s -X POST http://localhost:7778/api/github/pr/prepare \
  -H "Content-Type: application/json" \
  -d '{"branch": "test-branch"}' | python -m json.tool

# Verify no push happens without approval
curl -s -X POST http://localhost:7778/api/github/pr/create \
  -H "Content-Type: application/json" \
  -d '{"branch": "test-branch"}' | python -m json.tool
```

**Pass criteria:**
- PR creation without `I_APPROVE_GITHUB_WRITE` approval is rejected
- No push to any remote
- No auto-merge endpoint

---

## 19. Vast.ai — Estimate and Provision Refusal

```bash
# Estimate (no approval needed)
curl -s -X POST http://localhost:7778/api/vastai/estimate \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1:32b", "hours": 1}' | python -m json.tool

# Provision without approval — should be rejected
curl -s -X POST http://localhost:7778/api/vastai/provision \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1:32b"}' | python -m json.tool

# Provision with wrong approval — should be rejected
curl -s -X POST http://localhost:7778/api/vastai/provision \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1:32b", "approval": "wrong_token"}' | python -m json.tool
```

**Pass criteria:**
- Estimate returns cost breakdown
- Provision without approval returns error
- Provision with wrong approval returns error
- No instance created
- No API key in any response

---

## 20. Git Status Clean

```bash
cd IGRIS_GPT
git status
```

**Pass criteria:**
- Only `.igris/`, `logs/`, `.venv/` as untracked (all gitignored)
- No modified tracked files
- No secrets committed

---

## 21. Stop Server

```bash
bash scripts/stop_igris.sh
```

**Pass criteria:** server stops cleanly.

---

## Summary

| # | Check | Expected |
|---|-------|----------|
| 1 | Clone + install | No errors |
| 2 | pytest | 800+ pass |
| 3 | Start server | Clean start |
| 4 | Health check | 200 OK |
| 5 | UI load | 14+ tabs |
| 6 | Chat | Response with metadata |
| 7 | Create mission | Mission created |
| 8 | Plan deterministic | 3 steps |
| 9 | Plan LLM/auto | Valid plan with fallback |
| 10 | Plan explain | Explanation with risk |
| 11 | Materialize tasks | Tasks created |
| 12 | Loop 1 step | Step result |
| 13 | Patch proposal | Proposal created |
| 14 | Validate patch | Safety checks pass |
| 15 | Decision report | Report data |
| 16 | Diagnostics | Diagnostic data |
| 17 | Memory analysis | Advisory results |
| 18 | GitHub PR gated | Rejected without approval |
| 19 | Vast.ai gated | Rejected without approval |
| 20 | Git status clean | No leaked files |
| 21 | Stop server | Clean stop |

If all 21 checks pass, IGRIS_GPT v0.4 is operational.
