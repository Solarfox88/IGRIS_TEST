# LLM-Based Planning — Safe Schema Mode

IGRIS_GPT supports LLM-based mission planning with strict JSON schema validation. Invalid or unsafe plans automatically fall back to the deterministic planner.

## Planning Modes

| Mode | Behavior |
|------|----------|
| `deterministic` | Keyword-based, no LLM (default) |
| `llm` | Try LLM first, fall back to deterministic on failure |
| `auto` | Use LLM if available, otherwise deterministic |

## API

### POST /api/missions/{id}/plan?mode=deterministic|llm|auto

Plan a mission with the specified mode.

```json
// Response
{
  "mission": { /* mission data with steps */ },
  "planning": {
    "mode": "deterministic",
    "fallback_used": false,
    "fallback_reason": "",
    "validation": { "valid": true, "step_count": 3 },
    "provider": "deterministic",
    "model": "keyword-based",
    "latency_ms": 0
  }
}
```

### GET /api/missions/{id}/plan/explain

Explain the current plan for a mission.

```json
{
  "mission_id": "...",
  "title": "Fix login bug",
  "step_count": 3,
  "families": ["analyze", "code", "test"],
  "max_risk": "low",
  "steps": [...],
  "explanation": "Plan has 3 steps across 3 families. Maximum risk: low."
}
```

## Schema Validation

LLM output must be valid JSON matching this schema:

```json
{
  "steps": [
    {
      "title": "string (required)",
      "description": "string (required)",
      "family": "analyze|code|test|docs|config|refactor|deploy|review|debug|other (required)",
      "success_criteria": ["at least one criterion (required)"],
      "risk": "low|medium|high (required)",
      "safe_capabilities": ["read", "write", "patch_propose", ...],
      "dependencies": ["step_id"]
    }
  ]
}
```

### Required Fields

Every step must have: `title`, `description`, `family`, `success_criteria`, `risk`.

### Safe Capabilities (allowed)

`read`, `write`, `patch_propose`, `patch_validate`, `patch_apply`, `test_run`, `lint_run`, `analyze`, `search`, `diff_view`

### Unsafe Capabilities (rejected)

`shell_exec`, `auto_push`, `force_push`, `delete_repo`, `auto_merge`, `write_env`, `write_secrets`, `sudo`

Any step with an unsafe capability causes the entire plan to be rejected, triggering deterministic fallback.

## Fallback Scenarios

The deterministic planner is used when:
1. LLM is unreachable (Ollama down, no OpenAI key)
2. LLM returns invalid JSON
3. LLM response missing required fields
4. Steps contain unsafe capabilities
5. Steps missing success_criteria
6. Secret-like content detected in plan fields
7. Mode explicitly set to `deterministic`

## Safety Guarantees

- **No auto-execution** — plans describe what to do, never execute automatically
- **No auto-push/merge** — deploy steps are proposals only
- **Schema validation** — every field checked before accepting
- **Secret detection** — API keys, tokens rejected in plan text
- **Success criteria enforced** — every step must define how to verify completion
- **Risk assessment** — every step must declare risk level
