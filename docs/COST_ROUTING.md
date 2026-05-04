# Cost Router + VAST.ai Prep

## Overview

Enhanced provider routing with availability checks, cost estimation, and budget management. No automatic cost-incurring actions.

## Providers

| Provider | Cost/Call | Auto-Provision |
|----------|-----------|----------------|
| Ollama (local) | $0.00 | N/A (local) |
| OpenAI (fallback) | ~$0.003 | No |
| Vast.ai | ~$0.01 | **No** |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/routing/availability` | Check provider availability |
| `GET` | `/api/routing/explain` | Explain last routing decision |
| `GET` | `/api/routing/history` | Full routing history |
| `POST` | `/api/routing/estimate` | Estimate route + cost |
| `GET` | `/api/cost/summary` | Cost summary with budget |
| `GET` | `/api/cost/budget` | Current budget status |
| `POST` | `/api/cost/budget` | Update budget config |

### GET /api/routing/availability

Returns availability status for each provider without exposing API keys:

```json
{
  "ollama": {"available": true, "model": "phi4-mini", "cost_per_call": 0.0},
  "openai": {"available": false, "key_present": false, "cost_per_call": 0.003},
  "vastai": {"available": false, "key_present": false, "auto_provision": false}
}
```

### POST /api/routing/estimate

```json
{"task_type": "chat", "complexity": "low"}
```

Response:
```json
{
  "recommended_provider": "local",
  "model": "phi4-mini",
  "reason": "Local provider available and sufficient",
  "estimated_cost": 0.0,
  "budget_remaining": 10.0,
  "would_exceed_budget": false,
  "availability": {"ollama": true, "openai": false, "vastai": false}
}
```

### POST /api/cost/budget

```json
{"max_session_cost": 5.0, "warn_threshold": 0.9}
```

## Budget System

- Per-session cost tracking (not persistent across restarts)
- Default budget: $10.00 per session
- Warning at 80% usage (configurable)
- Budget status included in cost summary

## Safety

- API keys never exposed in responses (only `key_present: bool`)
- No automatic Vast.ai provisioning
- No cost-incurring action by default
- Budget warnings prevent runaway costs
- Vast.ai `auto_provision` always `false`
