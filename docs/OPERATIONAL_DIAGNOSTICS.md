# Operational Diagnostics

IGRIS_GPT includes an operational diagnostics system that detects unhealthy
patterns in task execution, memory, and loop behavior.

## Diagnostic Categories

### Task Starvation
Detects pending tasks that have been waiting too long (>5 minutes) without
being selected for execution. Also flags large backlogs (>10 pending tasks).

### Observation Loop
Detects when the same task families appear repeatedly in recent timeline events
without making progress. Triggers when a single family accounts for >50% of
the last 20 events.

### Blocked Accumulation
Detects accumulation of blocked tasks. Warning at 3+ blocked, critical at 5+.
Reports top blocking reasons.

### Family Failure Health
Analyzes failure rates per family from decision memory. Flags families with
>50% failure rate across 3+ attempts.

### Recovery Escalation
Detects when recovery attempts aren't resolving issues. Flags 10+ recent
failures and 3+ saturated families as signs of ineffective recovery.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/diagnostics` | GET | Full diagnostic report with all findings |
| `/api/diagnostics/summary` | GET | Quick summary for dashboard display |

### GET /api/diagnostics

Returns full diagnostic report:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "findings": [
    {
      "category": "starvation",
      "severity": "warning",
      "title": "3 task(s) starving",
      "detail": "3 pending task(s) have been waiting...",
      "affected_items": ["1", "2", "3"],
      "recommendation": "Check task selection logic..."
    }
  ],
  "summary": {
    "total_tasks": 10,
    "pending": 3,
    "running": 1,
    "completed": 4,
    "blocked": 2,
    "categories": {"starvation": 1},
    "severities": {"warning": 1},
    "healthy": false
  },
  "finding_count": 1,
  "has_critical": false,
  "has_warning": true
}
```

### GET /api/diagnostics/summary

Returns quick summary:

```json
{
  "healthy": true,
  "finding_count": 0,
  "has_critical": false,
  "has_warning": false,
  "categories": {},
  "task_stats": {
    "total": 5,
    "pending": 1,
    "running": 0,
    "completed": 4,
    "blocked": 0
  }
}
```

## Severity Levels

- **info**: Informational, no action required
- **warning**: Should be investigated
- **critical**: Requires immediate attention

## Safety

- All diagnostic output is secret-redacted
- No destructive actions are taken
- Read-only analysis of existing data
