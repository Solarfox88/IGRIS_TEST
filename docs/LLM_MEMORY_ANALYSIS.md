# LLM Memory Analysis

IGRIS_GPT can analyze its operational memory to extract actionable insights from decision/failure events.

## Principles

- **Advisory only** — analysis never executes actions
- **Deterministic fallback** — always produces results without LLM
- **Secret redaction** — all output is sanitized
- **Bounded output** — results are capped regardless of event volume

## Endpoints

### POST /api/memory/analyze

Full analysis with failure patterns, root causes, remediations, lessons.

```json
{
  "deterministic": {
    "failure_patterns": [...],
    "root_causes": [...],
    "remediations": [...],
    "lessons": [...],
    "constraints": {...}
  },
  "llm_enhanced": false,
  "latency_ms": 12,
  "advisory_only": true
}
```

### GET /api/memory/analysis

Compact summary for dashboard/chat.

```json
{
  "pattern_count": 2,
  "critical_issues": 0,
  "high_issues": 1,
  "saturated_families": ["config"],
  "avoid_families": ["config", "code"],
  "lesson_count": 3,
  "recommendation": "Avoid families: config, code",
  "llm_enhanced": false,
  "advisory_only": true
}
```

### GET /api/memory/lessons

Extracted lessons learned.

```json
{
  "lessons": [
    {"type": "recovery", "family": "code", "lesson": "Family 'code' recovered after remediation"},
    {"type": "persistent_block", "family": "config", "lesson": "Family 'config' saturated without remediation"}
  ],
  "count": 2,
  "types": ["recovery", "persistent_block"],
  "advisory_only": true
}
```

## Analysis Components

### Failure Patterns
Detects families with repeated failures (threshold: 2+ failures).

### Root Causes
Infers likely causes from failure reasons:
- Timeout → resource/connectivity issues
- Permission denied → access problems
- Not found → missing dependencies
- Syntax/parse errors → generation issues
- Test/assert failures → regression issues

### Remediation Strategies
Suggests actions based on severity:
- **Critical** (5+ failures): Block family, fix infrastructure, add pre-flight checks
- **High** (3+ failures): Add cooldown, review regressions, add validation gates
- **Moderate** (2 failures): Monitor, review logs

### Lessons Learned
Extracts patterns:
- **Recovery**: Families that recovered after remediation
- **Strength**: Families with 80%+ success rate
- **Weakness**: Families with 30% or lower success rate
- **Persistent block**: Saturated families without remediation

## LLM Enhancement

When an LLM (Ollama/OpenAI) is available, analysis is enhanced with:
- Pattern descriptions
- Root cause evidence
- Prioritized recommendations
- Contextual insights

Falls back to deterministic analysis when LLM is unavailable.
