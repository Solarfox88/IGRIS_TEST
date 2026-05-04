# IGRIS_GPT

**A2A-ready AI Engineering Agent** — local-first, safety-first, repo-aware, cost-aware.

IGRIS_GPT is a personal AI engineering agent designed as a controllable,
self-hosted alternative to cloud coding assistants. It runs a FastAPI server
with a tabbed web console, a safe terminal (command-id only), persistent task
engine, A2A protocol support, Ollama chat integration, teacher remediation,
anti-loop heuristics and cost-aware routing.

---

## Ubuntu Quick Install

```bash
git clone https://github.com/Solarfox88/IGRIS_GPT.git
cd IGRIS_GPT
bash scripts/install_ubuntu.sh
cp .env.example .env          # edit with your settings
bash scripts/setup_ollama.sh  # optional: local LLM
bash scripts/start_igris.sh
```

Open: **http://localhost:7778** (or `http://SERVER_IP:7778` for remote)

### Lifecycle Commands

```bash
bash scripts/status_igris.sh   # check status, health, readiness
bash scripts/stop_igris.sh     # stop server
bash scripts/restart_igris.sh  # restart
bash scripts/smoke_test.sh     # quick validation
```

---

## Local Install (Windows / Linux / macOS)

```bash
git clone https://github.com/Solarfox88/IGRIS_GPT.git
cd IGRIS_GPT
python -m venv .venv
```

Activate:
- **Linux/macOS**: `source .venv/bin/activate`
- **Windows**: `.venv\Scripts\activate`

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest -q                # run tests
cp .env.example .env               # edit with your settings
```

Start the server:

```bash
python -c "from igris.web.server import create_app, run_app; run_app(create_app())"
```

Open: **http://localhost:7778**

---

## Configuration

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `IGRIS_HOST` | `0.0.0.0` | Server bind address |
| `IGRIS_PORT` | `7778` | Server port |
| `PROJECT_ROOT` | `.` | Root directory IGRIS manages |
| `WORKSPACE_ROOT` | `.` | Workspace directory |
| `LOCAL_LLM_PROVIDER` | `ollama` | Local LLM provider |
| `LOCAL_LLM_MODEL` | `phi4-mini` | Model name |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:11434` | Ollama URL |
| `FALLBACK_LLM_PROVIDER` | `openai` | Fallback provider |
| `FALLBACK_LLM_MODEL` | `gpt-4o-mini` | Fallback model |
| `OPENAI_API_KEY` | *(empty)* | OpenAI key (optional) |
| `AUTO_COMMIT` | `false` | Auto-commit changes |
| `AUTO_PUSH` | `false` | Auto-push changes |

See `config/config.sample.json` for full configuration reference.

---

## Chat Engine

IGRIS_GPT uses a multi-tier chat engine:

1. **Ollama** (local, free) — default if running
2. **OpenAI** (fallback) — if `OPENAI_API_KEY` is set
3. **Deterministic fallback** — contextual responses without any LLM

Set up Ollama: `bash scripts/setup_ollama.sh`

The system never crashes if no LLM is available — it gracefully degrades to
deterministic responses that help navigate IGRIS capabilities.

---

## Tests

```bash
python -m pytest -q     # 393 tests
```

---

## Security

- **No free shell** — terminal accepts only pre-defined `command_id` values
- **No .env preview** — file browser blocks `.env` and secret-named files
- **Secret redaction** — output is scanned for OpenAI/GitHub/AWS keys and redacted
- **Path traversal blocked** — file browser rejects `..` and symlinks outside root
- **No arbitrary command execution** from UI or API

See [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md).

## Safe Terminal

The terminal accepts only commands from a fixed allowlist identified by
`command_id`. Available commands: `git_status`, `git_log`, `run_tests`,
`list_files`.

## File Browser

Read-only file browser with:
- Tree view of project files
- Text preview with secret redaction
- Blocks: path traversal, `.env`, binary files, sensitive filenames

## Task Engine

Persistent task storage under `.igris/tasks/` (git-ignored).

- Create, list, complete, block tasks via `/api/tasks`
- Timeline events under `.igris/timeline/`
- Tasks carry `family`, `priority`, `risk`, `semantic_fingerprint`

See [docs/TASK_ENGINE.md](docs/TASK_ENGINE.md).

## Teacher Remediation

The teacher module validates agent assignments and proposes remediation:

- `POST /api/teacher/remediate` — get remediation proposals
- Detects family saturation, duplicate tasks, observation loops
- Can auto-create remediation tasks with `create: true`

See [docs/TEACHER_GOVERNANCE.md](docs/TEACHER_GOVERNANCE.md).

## A2A Readiness

IGRIS_GPT implements the Agent-to-Agent protocol:

- `GET /.well-known/agent-card.json` — agent card with skills
- `POST /api/a2a/tasks` — create tasks from external agents
- `GET /api/a2a/tasks/{id}` — query task status
- `POST /api/a2a/tasks/{id}/messages` — append messages
- `GET /api/a2a/capabilities` — list capabilities

See [docs/A2A_READY_ARCHITECTURE.md](docs/A2A_READY_ARCHITECTURE.md).

## Cost Routing

Routes tasks to the cheapest suitable provider:
1. Local Ollama (free)
2. OpenAI (fallback)
3. VAST.ai (placeholder)

`/api/routing/history` and `/api/cost/summary` expose routing data with
`latency_ms`, `fallback_used`, and `estimated_cost`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Provider and model info |
| `/api/health` | GET | Health check |
| `/api/readiness` | GET | Readiness checks (incl. Ollama) |
| `/api/project/context` | GET | Project snapshot |
| `/api/git/status` | GET | Git branch/dirty/changed |
| `/api/files/tree` | GET | File tree |
| `/api/files/preview` | GET | File content preview |
| `/api/terminal/commands` | GET | List available commands |
| `/api/terminal/run` | POST | Execute command by ID |
| `/api/tests/run` | POST | Run pytest |
| `/api/tasks` | GET/POST | List/create tasks |
| `/api/tasks/{id}` | GET | Get task details |
| `/api/tasks/{id}/complete` | POST | Complete a task |
| `/api/tasks/{id}/block` | POST | Block a task |
| `/api/reports/recent` | GET | Recent execution reports |
| `/api/reports/{id}` | GET | Single report |
| `/api/agent/timeline` | GET | Agent timeline events |
| `/api/safety/status` | GET | Safety/anti-loop status |
| `/api/routing/history` | GET | Routing decisions |
| `/api/routing/explain` | GET | Routing explanation |
| `/api/cost/summary` | GET | Cost summary |
| `/api/sessions` | POST | Create chat session |
| `/api/sessions/{id}/messages` | POST | Send chat message |
| `/api/teacher/remediate` | POST | Teacher remediation |
| `/api/outcome/recent` | GET | Recent outcome recommendations |
| `/api/a2a/tasks` | POST | Create A2A task |
| `/api/a2a/tasks/{id}` | GET | Get A2A task |
| `/api/a2a/tasks/{id}/messages` | POST | A2A messages |
| `/api/a2a/capabilities` | GET | Agent capabilities |
| `/.well-known/agent-card.json` | GET | A2A agent card |
| `/api/logs` | GET | Application logs |
| `/api/memory/recent` | GET | Recent memory events |
| `/api/patches` | GET | List patch proposals |
| `/api/patches/propose` | POST | Create patch proposal |
| `/api/patches/{id}` | GET | Patch proposal detail + diff |
| `/api/patches/{id}/validate` | POST | Safety validation |
| `/api/patches/{id}/apply` | POST | Apply validated patch |
| `/api/patches/{id}/reject` | POST | Reject proposal |
| `/api/git/diff` | GET | Working tree diff (secret-redacted) |
| `/api/git/diff/stat` | GET | Diffstat summary |
| `/api/git/branches` | GET | List local branches |
| `/api/git/branch` | POST | Create branch (sanitized) |
| `/api/git/safety-check` | GET | Pre-commit safety analysis |
| `/api/git/commit-proposal` | POST | Commit proposal (no actual commit) |
| `/api/git/pr-summary` | GET | PR summary vs base branch |
| `/api/missions` | GET | List missions |
| `/api/missions` | POST | Create mission |
| `/api/missions/{id}` | GET | Mission detail |
| `/api/missions/{id}/plan` | POST | Generate plan for mission |
| `/api/missions/{id}/materialize-tasks` | POST | Create tasks from plan |
| `/api/missions/{id}/graph` | GET | Mission task dependency graph |
| `/api/memory/decisions` | GET | Recent decision events |
| `/api/memory/failures` | GET | Recent failure events |
| `/api/memory/saturation` | GET | Saturated families + constraints |
| `/api/memory/events` | POST | Record decision/failure/saturation/remediation |
| `/api/loop/step` | POST | Execute single loop step |
| `/api/loop/run` | POST | Run N loop steps (max_steps required) |
| `/api/loop/status` | GET | Current loop status |
| `/api/loop/recent` | GET | Recent loop step results |
| `/api/tasks/{id}/validate` | POST | Validate task against success criteria |
| `/api/tasks/{id}/validations` | GET | List validations for a task |
| `/api/validations/{id}` | GET | Get specific validation result |
| `/api/tasks/{id}/complete` | POST | Complete task (requires validation or manual reason) |
| `/api/a2a/store/tasks` | POST | Create A2A task |
| `/api/a2a/store/tasks` | GET | List A2A tasks |
| `/api/a2a/store/tasks/{id}` | GET | Get A2A task detail |
| `/api/a2a/store/tasks/{id}/status` | POST | Update A2A task status |
| `/api/a2a/tasks/{id}/artifacts` | GET | List task artifacts |
| `/api/a2a/tasks/{id}/artifacts` | POST | Add artifact to task |
| `/api/a2a/tasks/{id}/cancel` | POST | Cancel A2A task |
| `/api/a2a/tasks/{id}/events` | GET | Get task events |
| `/api/routing/availability` | GET | Check provider availability |
| `/api/routing/estimate` | POST | Estimate route + cost |
| `/api/cost/budget` | GET | Current budget status |
| `/api/cost/budget` | POST | Update budget config |
| `/api/safety/policy` | GET | Safety policy config and status |
| `/api/safety/policy/check` | POST | Check command against safety policy |
| `/api/tasks/selection/explain` | GET | Explainable task selection with scores |
| `/api/diagnostics` | GET | Full operational diagnostics report |
| `/api/diagnostics/summary` | GET | Quick diagnostics summary |

## Web Console

14-tab agentic console:
- **Mission Control** — health, readiness, project context (auto-refresh)
- **Terminal** — safe command execution by ID
- **Files** — file tree and preview
- **Git** — diff viewer, branch management, safety check, commit proposals, PR summary
- **Tests** — run pytest with output
- **Logs** — application log viewer
- **Agent** — timeline events with type/severity (auto-refresh)
- **Tasks** — create/complete/block tasks + teacher remediation
- **Safety** — anti-loop status + execution reports (unified refresh)
- **Cost** — provider availability cards, budget bar, route estimate, cost summary (auto-refresh)
- **A2A** — agent card, capabilities, A2A store tasks
- **Memory** — decision/failure memory, constraints, record events
- **Loop** — autonomous execution loop controls (1/3/5 steps), status, recent steps (auto-refresh)
- **Patches** — propose, validate, diff preview, apply/reject code changes

## What Works (v0.2)

- Full FastAPI backend with 55+ endpoints
- Ollama chat engine with deterministic fallback
- Persistent task engine and execution reports
- Safety module: path access, secret detection, output truncation
- A2A protocol: agent card, task lifecycle, messages, artifacts, cancel, events
- Teacher governance with remediation proposals
- Outcome router with recommendations
- Anti-loop heuristics with family saturation
- Cost-aware routing with latency tracking, budget management
- Patch proposals: propose, validate, diff preview, apply/reject
- Controlled git workflow: diff, branches, safety check, commit proposals, PR summary
- Mission planner: create missions, generate plans, materialize tasks, dependency graph
- Decision/failure memory: record, query, constraints, teacher/task_selection integration
- Autonomous execution loop: safety-first semi-autonomous step execution
- Validation layer: success criteria, definition of done, manual override
- A2A store: extended status lifecycle, artifacts, cancel, events
- Provider availability checks, route estimation, per-session budget tracking
- Mobile-responsive UI with 14 tabs and auto-refresh
- 393 passing tests
- Ubuntu install scripts with lifecycle management
- Systemd service example

## What's Placeholder

- VAST.ai integration (routing logic present, no real API calls or provisioning)
- LLM-based mission planning (currently deterministic keyword-based)
- LLM-based memory analysis
- WebSocket live updates
- Memory cleanup/archival

## Systemd Service

See [docs/SYSTEMD_SERVICE.md](docs/SYSTEMD_SERVICE.md) for production deployment.

## Documentation

- [SECURITY_MODEL.md](docs/SECURITY_MODEL.md)
- [TASK_ENGINE.md](docs/TASK_ENGINE.md)
- [TEACHER_GOVERNANCE.md](docs/TEACHER_GOVERNANCE.md)
- [A2A_READY_ARCHITECTURE.md](docs/A2A_READY_ARCHITECTURE.md)
- [A2A_ARTIFACTS_LONG_RUNNING.md](docs/A2A_ARTIFACTS_LONG_RUNNING.md)
- [AGENT_CONTRACT.md](docs/AGENT_CONTRACT.md)
- [SYSTEMD_SERVICE.md](docs/SYSTEMD_SERVICE.md)
- [INSTALLATION_VERIFICATION.md](docs/INSTALLATION_VERIFICATION.md)
- [OPERATIONAL_BASELINE.md](docs/OPERATIONAL_BASELINE.md)
- [PATCH_PROPOSAL_WORKFLOW.md](docs/PATCH_PROPOSAL_WORKFLOW.md)
- [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)
- [MISSION_PLANNER.md](docs/MISSION_PLANNER.md)
- [DECISION_MEMORY.md](docs/DECISION_MEMORY.md)
- [AUTONOMOUS_LOOP.md](docs/AUTONOMOUS_LOOP.md)
- [VALIDATION_MODEL.md](docs/VALIDATION_MODEL.md)
- [COST_ROUTING.md](docs/COST_ROUTING.md)
- [OPERATIONAL_DIAGNOSTICS.md](docs/OPERATIONAL_DIAGNOSTICS.md)
- [STRICT_SAFETY_EXPLAINABLE_SELECTION.md](docs/STRICT_SAFETY_EXPLAINABLE_SELECTION.md)

## License

Private — Solarfox88
