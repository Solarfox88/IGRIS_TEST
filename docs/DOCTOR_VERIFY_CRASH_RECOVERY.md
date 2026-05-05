# Doctor, Verify & Crash Recovery

Epic #39 — Operational Baseline for Autonomous Missions

## Overview

This epic makes IGRIS_GPT stable, diagnosticable, and ready for long-running
autonomous missions. It provides three core capabilities:

1. **Doctor** — comprehensive environment diagnostics
2. **Verify** — quick installation smoke check
3. **Crash Recovery** — structured crash handling with reports, timeline events,
   and last-known-good-state tracking

## Doctor (`igris doctor`)

Runs 14 environment checks covering:

| Category | What it checks |
|----------|---------------|
| `python` | Python version >= 3.10 |
| `venv` | Virtual environment active |
| `deps` | Critical packages importable (fastapi, uvicorn, pydantic, jinja2, httpx) |
| `server` | FastAPI server reachable at configured host:port |
| `ollama` | Ollama reachable, configured model available |
| `openai` | OpenAI API key present (never exposed) |
| `git` | git installed and runnable |
| `docker` | Docker installed and daemon reachable (optional) |
| `ssh` | SSH client available (optional) |
| `ports` | Configured port status |
| `workspace` | Project root exists, .igris dir present |
| `permissions` | Write access to project root |
| `config` | .env file exists, config.json valid |

### API

- `GET /api/doctor` — JSON report with all checks, overall status, summary
- `GET /api/doctor/markdown` — Markdown-formatted report

### Report schema

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "overall": "ok | warning | error",
  "summary": {"ok": 12, "warning": 1, "error": 0, "skipped": 1},
  "checks": [
    {
      "name": "python_version",
      "category": "python",
      "status": "ok",
      "detail": "Python 3.12.0",
      "fix_suggestion": "...",  // only if applicable
      "meta": {"version": "3.12.0"}
    }
  ],
  "total_checks": 14
}
```

Each check with status `error` or `warning` includes a `fix_suggestion`.

## Verify (`igris verify`)

Quick smoke check of installation essentials:

- Project root exists
- Critical files present (pyproject.toml, igris/__init__.py, server.py, config.py)
- .igris directory writable
- Config loadable
- Dependencies importable

### API

- `GET /api/verify` — JSON with pass/fail per check and overall `ok` flag

## Crash Recovery

Every unhandled exception during mission/loop execution is routed through
`handle_crash()`, which:

1. **Classifies** the failure into a category
2. **Redacts** the stacktrace (removes secrets, API keys, tokens)
3. **Builds** a structured `CrashReport`
4. **Persists** it as JSON and Markdown in `.igris/recovery/crashes/`
5. **Returns** the report for timeline event creation

### Failure categories

| Category | Description | Suggested remediation |
|----------|-------------|----------------------|
| `import_error` | Missing dependency | `pip install -e ".[dev]"` |
| `connection_error` | Service unreachable | Check network/services |
| `timeout_error` | Operation timeout | Increase timeout |
| `permission_error` | Insufficient perms | Check ownership |
| `file_not_found` | Missing file/path | Verify paths |
| `config_error` | Bad configuration | Check .env/config |
| `json_error` | Malformed JSON | Fix syntax |
| `llm_error` | LLM provider failure | Check Ollama/API key |
| `git_error` | Git operation failed | Check git status |
| `test_failure` | Test failure | Run pytest locally |
| `validation_error` | Schema validation | Check input data |
| `unknown` | Unclassified | Inspect stacktrace |

### Last Known Good State

- `save_good_state(state)` — Persist current state before risky operations
- `load_good_state()` — Retrieve last safe checkpoint
- Stored in `.igris/recovery/last_known_good_state.json`

### API

- `GET /api/crash-reports` — List recent crash reports (newest first)
- `GET /api/crash-reports/{crash_id}` — Get specific crash report
- `GET /api/crash-reports/last-good-state` — Get last known good state
- `POST /api/crash-reports/save-good-state` — Save current state as checkpoint

### Crash report schema

```json
{
  "id": "crash-abc123def456",
  "timestamp": "2024-01-01T00:00:00Z",
  "failure_category": "connection_error",
  "failure_description": "Connection refused",
  "exception_type": "ConnectionError",
  "redacted_stacktrace": "...",
  "mission_id": "m1",
  "task_id": "t1",
  "action_id": null,
  "trace_id": "trace-abc12345",
  "context": {},
  "last_known_good_state": {...},
  "suggested_remediation": "Check network...",
  "severity": "error"
}
```

## Config Validation

Validates five configuration sections:

1. **env** — .env presence, expected environment variables
2. **config_json** — config/config.sample.json validity
3. **provider** — LLM provider names, API key presence
4. **budget** — Vast.ai cost limits
5. **safety_policy** — AUTO_PUSH/AUTO_COMMIT/VASTAI flags

### API

- `GET /api/config/validate` — Full validation report

### Safety

- `.env` contents are **never read** — only existence is checked
- Secret values are **never exposed** in any response
- All output is redacted via `safety.redact_secrets()`
- AUTO_PUSH=true is flagged as error (violates safety policy)

## File Layout

```
igris/core/doctor.py           — Doctor checks and verify
igris/core/crash_recovery.py   — Crash handler, reports, good state
igris/core/config_validator.py — Config validation
tests/test_doctor.py           — 48 tests
tests/test_crash_recovery.py   — 38 tests
tests/test_config_validator.py — 25 tests
```

## Timeline events

All doctor/verify/crash operations create timeline events:

- `type: "doctor"` — Doctor run result
- `type: "verify"` — Verify pass/fail
- `type: "crash"` — Crash occurrence
- `type: "recovery"` — Good state saved

Each event includes severity and is traceable via mission/task/trace IDs.
