# LLM Patch Generation (Proposal-Only)

LLM-powered patch generation that creates proposals — never auto-applies.

## Architecture

```
Task/Mission -> LLM Prompt -> JSON Output -> Schema Validation -> Patch Proposal -> Validation -> Diff Preview -> Gated Apply
```

## Endpoints

### `POST /api/patches/generate`
Generate a patch proposal from a description.

```json
{
  "title": "Fix divide-by-zero bug",
  "description": "Add zero check in divide function",
  "context": "optional additional context"
}
```

### `POST /api/tasks/{task_id}/generate-patch`
Generate a patch from an existing task.

## Safety Rules

- **Proposal-only**: generated patches are never auto-applied
- **Schema validated**: LLM output must match expected JSON schema
- **Path validation**: no `.env`, `.git`, `.igris`, binary files, path traversal
- **Content validation**: no secrets, max 50KB per file, max 5 files
- **Secret redaction**: all output content is redacted
- **Deterministic fallback**: when LLM unavailable, returns safe placeholder

## Blocked Paths

- `.env`, `.git`, `.igris`, `node_modules`, `__pycache__`
- `.pem`, `.key`, `.exe`, `.bin`, `.dll`, `.so`, binary/image extensions

## Output Schema

```json
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "action": "create | modify",
      "after": "full file content",
      "reason": "why this change"
    }
  ],
  "description": "patch description",
  "risk": "low | medium | high",
  "generated_by": "llm | deterministic",
  "proposal_only": true,
  "latency_ms": 150
}
```

## Deterministic Fallback

When LLM is unavailable or returns invalid output:
- Returns empty files list
- `generated_by: "deterministic"`
- Includes `fallback_reason`
- No crash, no error

## Integration with Existing Workflow

Generated patches feed into the existing patch proposal pipeline:
1. `POST /api/patches/generate` -> draft
2. `POST /api/patches/propose` -> create proposal from draft
3. `POST /api/patches/{id}/validate` -> safety check
4. `GET /api/patches/{id}` -> diff preview
5. `POST /api/patches/{id}/apply` -> gated apply (existing workflow)
