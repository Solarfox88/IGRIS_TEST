# Tool Runtime — Epic #41

Modular, governed tool execution for real local/server operations.

## Tool Families

| Tool | Actions | Risk Level |
|------|---------|------------|
| `shell` | execute (governed by allowlist) | varies |
| `filesystem` | read, write, diff | low-medium |
| `git` | status, diff, log, branch, commit (gated), push (gated) | low-high |
| `docker` | ps, logs, compose_config, compose_up/down (gated), health | low-high |
| `nginx` | config_test, reload (gated) | low-high |
| `systemd` | status, logs, restart (gated) | low-high |
| `http` | health check (status, SSL, response time) | low |
| `test` | pytest runner | low |
| `ssh_host` | registry with policies | - |

## Safety Features

- **No free shell** — only allowlisted commands via `shell_execute()`
- **Path guard** — filesystem operations restricted to project root
- **Secret guard** — blocks read/write of `.env`, credential files
- **Secret content detection** — blocks writing secret-like patterns
- **Auto-backup** — files backed up before overwrite (rollback)
- **Risk gating** — high/critical actions blocked in safe mode
- **Output redaction** — all output passes through `redact_secrets()`
- **Timeout** — every subprocess has configurable timeout
- **Environment redaction** — env vars with KEY/TOKEN/SECRET/PASSWORD redacted

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/tools` | List available tools |
| `POST` | `/api/tools/shell/execute` | Run governed command |
| `POST` | `/api/tools/fs/read` | Read file safely |
| `POST` | `/api/tools/fs/write` | Write file safely |
| `POST` | `/api/tools/fs/diff` | Preview diff |
| `GET` | `/api/tools/git/status` | Git status |
| `GET` | `/api/tools/git/diff` | Git diff |
| `GET` | `/api/tools/git/log` | Git log |
| `GET` | `/api/tools/git/branch` | Git branches |
| `POST` | `/api/tools/git/commit` | Gated commit |
| `POST` | `/api/tools/docker/ps` | Docker ps |
| `POST` | `/api/tools/http/check` | HTTP health check |
| `POST` | `/api/tools/test/run` | Run tests |
| `GET` | `/api/tools/hosts` | List SSH hosts |
| `POST` | `/api/tools/hosts/register` | Register SSH host |

## Push Safety

- Push requires `approval_token: "I_APPROVE_GITHUB_WRITE"`
- Push to `main`/`master` is always forbidden
- Force push is never available

## File Layout

```
igris/core/tool_runtime.py    — Runtime logic
tests/test_tool_runtime.py    — 53 tests
docs/TOOL_RUNTIME.md          — This file
```
