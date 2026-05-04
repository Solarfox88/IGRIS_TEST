# Context-Enriched Chat

## Overview

Enriches chat messages with full project context: missions, tasks,
reports, memory constraints, git state, patch proposals, validation
state, and cost/routing information.

The chat can propose patches or tasks but does NOT execute commands
or apply changes directly — all actions go through the proper workflow.

## Context Sections

| Section | Content |
|---|---|
| missions | Active/completed missions, titles |
| tasks | Task counts by status (pending, running, completed, blocked) |
| memory | Avoid families, saturated families, failure counts |
| git | Branch, dirty status, changed files, HEAD |
| patches | Proposal counts by status |
| cost | Current provider and model |
| project_state | Cooling-down, critical, elevated families |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat/context` | GET | Full project context (all sections) |
| `/api/chat/context/summary` | GET | Concise context summary |
| `/api/chat/stream` | POST | SSE chat with optional `enrich: true` |

### POST /api/chat/stream (enriched)

```json
{
  "message": "what tasks are pending?",
  "enrich": true,
  "session_id": "1"
}
```

When `enrich: true`, the system prompt includes full project context
so the LLM (or fallback) can provide informed responses.

### GET /api/chat/context/summary

Returns a flat dict with key metrics:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "missions_active": 1,
  "tasks_pending": 3,
  "tasks_blocked": 0,
  "memory_avoid_families": ["deploy"],
  "git_branch": "main",
  "git_dirty": false,
  "patches_pending": 1,
  "provider": "ollama",
  "cooling_down": []
}
```

## Safety

- Chat does NOT execute commands
- Chat does NOT apply patches or create files
- Chat can suggest/propose — user must use proper workflow to act
- All context is secret-redacted
