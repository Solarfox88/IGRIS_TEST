# Strict Safety Policy + Explainable Task Selection

## Strict Safety Policy

`igris/core/safe_policy.py` provides a second-level safety check after the
command_id allowlist. It validates:

1. **Allowlist check**: command_id must be in ALLOWED_COMMANDS
2. **Blocked list**: explicitly blocked commands (push, delete, shell_exec, sudo)
3. **Destructive keywords**: rejects commands containing dangerous patterns
4. **Rate limiting**: max 20 executions per 60s, max 5 same-command burst
5. **Context validation**: detects path traversal in context

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/safety/policy` | GET | Current policy config and status |
| `/api/safety/policy/check` | POST | Check if a command_id would be allowed |

### POST /api/safety/policy/check

```json
{"command_id": "git_status", "context": {"project_root": "/path"}}
```

Response:
```json
{
  "allowed": true,
  "command_id": "git_status",
  "reason": "All safety checks passed",
  "checks_passed": ["allowlist", "blocked_list", "destructive_check", "rate_limit", "context_validation"],
  "checks_failed": []
}
```

## Explainable Task Selection

`igris/core/task_selection_explain.py` wraps the existing task selection with
detailed explanations of why each candidate was selected, rejected, or skipped.

Each candidate receives a score based on:
- Base priority (priority × 10)
- Risk penalty (high: -50, medium: -10)
- Blocked family (-100)
- Saturated family (-80)
- Failure history penalty (-5 per failure, -30 at 3+)
- Semantic duplicate (-60)
- Non-pending status (-200)

### API Endpoint

| Endpoint | Method | Description |
|---|---|---|
| `/api/tasks/selection/explain` | GET | Full selection explanation |

### GET /api/tasks/selection/explain

Response:
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "selected": {"id": 1, "title": "...", ...},
  "candidates": [
    {
      "task_id": 1,
      "title": "run tests",
      "family": "test",
      "priority": 5,
      "risk": "low",
      "status": "pending",
      "selected": true,
      "score": 50.0,
      "why": "Selected via fallback",
      "rejected_reasons": []
    }
  ],
  "saturated_families": [],
  "blocked_families": [],
  "recent_failure_count": 0,
  "recent_decision_count": 0,
  "selection_source": "fallback",
  "summary": "Selected task #1 'run tests' (family=test, source=fallback). 1 candidates evaluated, 0 families saturated, 0 blocked."
}
```

## Safety

- All output is secret-redacted
- No destructive actions
- Read-only analysis
- Rate limits prevent abuse
