# Validation Layer + Definition of Done

## Overview

The validation layer ensures tasks are not completed just because a command returned exit code 0. Tasks must satisfy their `success_criteria` to be considered truly complete.

## What It Does

- Validates task completion against success_criteria
- Maps criteria to automated checks (test reports, file existence)
- Falls back to manual verification for generic criteria
- Persists validation results under `.igris/validations/`
- Prevents auto-completion without criteria or validation

## What It Does NOT Do

- No automatic task completion without validation
- No LLM-based evaluation (deterministic checks)
- No modification of task data during validation
- No execution of commands during validation

## Validation Rules

1. **Tasks without criteria** cannot be auto-completed → status `needs_review`
2. **Manual completion** requires `manual_completion_reason`
3. **Test criteria** check against recent execution reports
4. **File criteria** check file existence on disk or in files_changed
5. **Generic criteria** require manual verification
6. **Failed validation** leaves task as `needs_review` or `blocked`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/tasks/{id}/validate` | Validate task against criteria |
| `GET` | `/api/tasks/{id}/validations` | List validations for task |
| `GET` | `/api/validations/{id}` | Get specific validation result |
| `POST` | `/api/tasks/{id}/complete` | Complete task (requires validation or manual reason) |

### POST /api/tasks/{id}/validate Body

```json
{
  "files_changed": ["src/main.py", "docs/readme.md"],
  "manual_completion_reason": "optional override reason"
}
```

### ValidationResult Response

```json
{
  "valid": true,
  "task_id": 1,
  "overall_status": "completed",
  "criteria_results": [
    {"criterion": "All tests pass", "met": true, "evidence": "Recent test report shows success"}
  ],
  "reason": "All 1 criteria met",
  "manual_completion_reason": "",
  "validated_at": "2024-01-01T00:00:00Z",
  "validation_id": "abc123def456"
}
```

## Overall Status Mapping

| Validation Status | Task Status |
|-------------------|-------------|
| `completed` | `completed` |
| `needs_review` | `pending` |
| `blocked` | `blocked` |

## Criterion Types

| Keyword Match | Check Type | Automated |
|---------------|------------|-----------|
| test, pytest, pass, green | Test report check | Yes |
| file, create, exist, add | File existence check | Yes |
| (other) | Manual verification | No |

## Persistence

Validation results stored in `.igris/validations/` as JSON files. Each validation has a unique `validation_id`.

## Safety

- Secret redaction on all text fields (reason, manual_completion_reason)
- No automatic execution during validation
- Timeline event for every validation attempt
