# IGRIS_GPT Operational Baseline — v0.5

Current state of the system and operational procedures.

Tag: `v0.5-real-world-candidate`

## What Works

### Core Infrastructure
- **FastAPI server** on port 7778 with 80+ API endpoints
- **Web console** with 14+ functional tabs
- **Persistent storage** under `.igris/` (tasks, reports, timeline, memory, missions, state, decision reports)
- **Ubuntu install scripts** (install, start, stop, restart, status, smoke test)
- **Systemd service example** for production deployment
- **Clean install verified** — clone, venv, pip install, pytest, start, health check

### Chat Engine
- **Ollama integration** — uses local phi4-mini when available
- **OpenAI fallback** — if API key configured
- **Deterministic fallback** — contextual responses without any LLM
- **SSE streaming** — text/event-stream endpoint
- **Tier selector** — auto/local/fallback per session
- **Context enrichment** — mission, tasks, memory, git, reports, patches, cost
- Response includes `provider`, `model`, `fallback_used`, `latency_ms`, `routing_reason`

### Mission Planning
- **Deterministic planner** — keyword-based, always available
- **LLM-based planner** — JSON schema validated, safe capabilities only
- **Mode selection** — deterministic | llm | auto
- **Plan explanation** — step-by-step with risk/family analysis
- **Fallback** — invalid LLM plans fall back to deterministic

### Task Management
- Create, list, complete, block tasks via API and UI
- Tasks persist as JSON files under `.igris/tasks/`
- Task family classification, priority, risk assessment
- Anti-loop detection with family saturation
- **Explainable selection** — selected task, score, why, rejected reasons

### Memory & Analysis
- **Decision/failure/saturation/remediation events**
- **LLM memory analysis** — pattern detection, root causes, remediations, lessons
- **Deterministic analysis fallback** — always produces results
- **Teacher integration** — memory constraints in payload
- **Advisory only** — analysis never executes actions

### Safety
- Terminal accepts only `command_id` from allowlist (no free shell)
- **Strict command policy** — second-layer validation after command resolution
- File preview blocks `.env`, path traversal, binary files
- Secret redaction in all output (OpenAI, GitHub, AWS, Vast.ai patterns)
- Agent card contains no secrets
- **Safety policy** with destructive pattern detection

### Operational Diagnostics
- Task starvation detection
- Observation loop detection
- Blocked task accumulation warnings
- Family failure health monitoring
- Recovery escalation tracking

### ProjectState & Cooldown
- Unified project state snapshot
- Family metrics (completed/failed/blocked counts)
- Cooldown per family
- Recovery escalation tracking
- State-driven diagnostics and teacher integration

### Decision Reports
- JSON decision report per loop cycle
- Selected/rejected tasks with reasons
- Safety decisions and memory constraints
- Teacher recommendations and next action
- Persisted under `.igris/reports/decisions/`

### Git & GitHub Workflow
- Diff viewer with syntax highlighting
- Branch management with name sanitization
- Safety check for secrets and runtime artifacts
- Commit proposals (gated)
- PR summary generation
- **GitHub PR workflow** — gated commit, push, PR creation
- Approval token: `I_APPROVE_GITHUB_WRITE`
- No push to main, no force push, no auto-merge

### Vast.ai GPU Management (Gated/Mock)
- Config, status, estimate, offer search endpoints
- Gated provision/destroy/set-mode
- Approval token: `I_APPROVE_VASTAI_COSTS`
- Budget gate ($0.50/hr default)
- Anti-duplicate instance guard
- All operations mock — no real costs

### Autonomous Loop
- Bounded execution steps
- Task selection with memory/diagnostic constraints
- Outcome routing to recommendations
- Stop conditions (max steps, no tasks, critical diagnostics)

### A2A Protocol
- Agent card at `/.well-known/agent-card.json`
- Task creation, querying, and message exchange
- Artifacts storage with long-running task support

### Cost and Routing
- Tracks routing decisions with provider, model, latency, cost
- Cost summary with local vs. fallback call counts
- Ollama/OpenAI availability check

### Benchmarks
- 5 operational workflow benchmarks documented
- Deterministic E2E verification
- docs-only, bugfix, test recovery, multi-file, full loop smoke

## How to Operate

### Install
```bash
git clone https://github.com/Solarfox88/IGRIS_GPT.git
cd IGRIS_GPT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Test
```bash
python -m pytest -q
# 928+ tests must pass
```

### Start
```bash
bash scripts/start_igris.sh
# or manually:
python -c "from igris.web.server import create_app, run_app; run_app(create_app())"
```

### Stop
```bash
bash scripts/stop_igris.sh
```

### Health Check
```bash
curl -s http://localhost:7778/api/health
curl -s http://localhost:7778/api/readiness
curl -s http://localhost:7778/api/status
```

## What's Still Placeholder

| Feature | Status | Notes |
|---|---|---|
| Vast.ai real API calls | Gated mock | Framework ready, HTTP transport not connected |
| LLM patch generation | Proposal-only (v0.5) | Schema-validated, deterministic fallback |
| WebSocket live updates | Not implemented | UI uses polling (15s auto-refresh) |
| Vector search memory | Not implemented | Memory is JSON append |
| Multi-repo management | Not implemented | Single project root |
| Real external benchmarks | Sandbox (v0.5) | 5 scenarios on sandbox project, PR dry-run |

See `docs/PREPARED_NOT_IMPLEMENTED.md` for full details.

## Runtime Directories

Created at first run, never committed:
- `.igris/tasks/` — persistent task storage
- `.igris/reports/` — execution reports
- `.igris/reports/decisions/` — decision reports
- `.igris/timeline/` — agent events
- `.igris/memory/` — memory events
- `.igris/missions/` — mission plans
- `.igris/state/` — project state
- `logs/` — application logs

## Key Files

- `igris/web/server.py` — all API endpoints
- `igris/core/chat_engine.py` — LLM integration
- `igris/core/task_engine.py` — task persistence
- `igris/core/mission_planner.py` — deterministic planner
- `igris/core/llm_planner.py` — LLM-based planner
- `igris/core/decision_memory.py` — memory events
- `igris/core/memory_analysis.py` — memory analysis
- `igris/core/teacher.py` — teacher governance
- `igris/core/outcome_router.py` — outcome routing
- `igris/core/safety.py` — safety module
- `igris/core/diagnostics.py` — operational diagnostics
- `igris/core/project_state.py` — project state
- `igris/core/decision_reports.py` — decision reports
- `igris/core/autonomous_loop.py` — loop engine
- `igris/core/llm_patch_generator.py` — LLM patch generation
- `igris/layers/advisory/vastai_manager.py` — Vast.ai manager
- `igris/web/templates/index.html` — UI HTML
- `igris/web/static/js/app.js` — UI JavaScript
