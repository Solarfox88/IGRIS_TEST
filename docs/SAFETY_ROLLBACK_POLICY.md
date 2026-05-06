# Safety, Rollback & Autonomy Policy — Epic #42

## Overview

Enables real operational freedom without destructivity. Every risky action
passes through risk classification → approval check → rollback verification
→ safety event logging.

## Risk Levels

| Level | Description | Examples |
|-------|-------------|----------|
| `low` | Read-only, no side effects | status, diff, grep, test, analyze |
| `medium` | Workspace writes, local installs | write file, pip install, git commit |
| `high` | Remote/service operations | deploy, push, nginx reload, docker down |
| `critical` | Destructive/irreversible | delete, db migrate, force push, DNS changes |

## Approval Modes

| Mode | Low | Medium | High | Critical |
|------|-----|--------|------|----------|
| `safe` | auto | auto | blocked | blocked |
| `operator` | auto | auto | auto if rollback | blocked |
| `trusted` | auto | auto | auto on authorized hosts | blocked |
| `manual-critical` | auto | auto | confirmation | confirmation |

Critical actions always require an explicit `approval_token`.

## API Endpoints

### Risk Classification
- `POST /api/safety/classify-risk` — Classify action risk
- `POST /api/safety/check-approval` — Check approval under policy
- `POST /api/safety/guard-secret` — Block secret file access

### Rollback
- `POST /api/rollback/backup-file` — Backup file before modification
- `POST /api/rollback/save-state` — Save state snapshot
- `GET /api/rollback/entries` — List rollback entries
- `GET /api/rollback/entries/{id}` — Get specific entry
- `POST /api/rollback/entries/{id}/verify` — Verify rollback applicable
- `POST /api/rollback/entries/{id}/apply` — Apply file rollback

### Safety Events
- `GET /api/safety/events` — List events (filter by type/mission/severity)
- `GET /api/safety/events/{id}` — Get specific event
- `GET /api/safety/summary` — Event statistics

## Secret Guard

Blocks access to files matching: `.env*`, `.secret`, `credentials*`,
`service_account*`, `id_rsa`, `id_ed25519`, `*.pem`, `*.key`.

## Rollback Types

| Type | Description |
|------|-------------|
| `file_backup` | Copy of file before modification (auto-restorable) |
| `config_backup` | Config file snapshot (nginx, docker-compose) |
| `diff_snapshot` | Git diff before commit (manual review) |
| `state_snapshot` | Arbitrary JSON state (manual review) |

## Safety Event Types

`action_blocked`, `action_approved`, `risk_decision`, `rollback_required`,
`rollback_applied`, `escalation`, `secret_detected`, `policy_violation`,
`approval_requested`, `approval_granted`, `approval_denied`

## File Layout

```
igris/core/risk_classifier.py    — Risk levels, approval modes, secret guard
igris/core/rollback_manager.py   — Backup/restore/rollback
igris/core/safety_event_log.py   — Safety event tracking
tests/test_safety_rollback.py    — 80 tests
```
