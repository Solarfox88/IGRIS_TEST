# ProjectState + Saturation Cooldown

## Overview

`igris/core/project_state.py` tracks per-family execution metrics, saturation
cooldowns, recovery escalation, and recent task fingerprints. It integrates
with `decision_memory` without replacing it.

## Family Metrics

Each family tracks:
- `total_attempts`, `successes`, `failures`
- `failure_rate` (computed)
- `consecutive_failures`
- `recovery_level` (0=normal, 1=caution, 2=elevated, 3=critical)
- `cooldown_until` (timestamp when cooldown expires)
- Last attempt/success/failure timestamps

## Recovery Escalation

| Level | Label | Trigger | Cooldown |
|---|---|---|---|
| 0 | normal | default | 0s |
| 1 | caution | 2 consecutive failures | 60s |
| 2 | elevated | 4 consecutive failures | 5min |
| 3 | critical | 6 consecutive failures | 15min |

Successes reduce recovery level by 1. Cooldown resets can also be done manually.

## Fingerprints

Recent task fingerprints (up to 50) are tracked to detect duplicate attempts.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/project-state` | GET | Full project state with all family metrics |
| `/api/project-state/recovery` | GET | Recovery summary integrated with decision memory |
| `/api/project-state/family/{family}` | GET | Check family availability |
| `/api/project-state/family/{family}/reset-cooldown` | POST | Reset cooldown for a family |
| `/api/project-state/fingerprints` | GET | Recent task fingerprints |

## Integration

- **Decision Memory**: Saturated families from decision_memory are included in
  availability checks and recovery summary
- **Teacher**: Recovery summary provides constraints for teacher payload
- **Loop**: Cooling-down families should be skipped in task selection
- **Diagnostics**: Family failure health is complemented by project state metrics
