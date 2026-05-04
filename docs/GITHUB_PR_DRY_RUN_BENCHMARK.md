# GitHub PR Dry Run Benchmark

Verifies the full GitHub PR workflow without remote side effects.

## Workflow Steps

```
Safety Check -> Commit Proposal -> PR Prepare -> Gated PR Create -> PR Status
```

## What's Tested

### Branch Validation
- Protected branches (main, master) blocked from push
- Branch names must match allowlist: `devin/*`, `feature/*`, `fix/*`, `bugfix/*`, `hotfix/*`, `sprint/*`, `release/*`, `chore/*`, `docs/*`

### Safety Check (`GET /api/git/safety-check`)
- Scans staged files for secrets, runtime artifacts, blocked files
- Must pass before commit

### Commit Proposal (`POST /api/git/commit-proposal`)
- Dry-run commit — shows what would be committed
- No actual commit created

### Gated Commit (`POST /api/git/commit`)
- Requires approval: `I_APPROVE_GITHUB_WRITE`
- Without approval: blocked
- Without commit message: 400 error
- Safety check must pass

### PR Prepare (`POST /api/github/pr/prepare`)
- Generates PR body from diffstat, branch, base
- No remote operations
- Returns: title, body, branch, base, diffstat, commit_count

### PR Create (`POST /api/github/pr/create`)
- Requires approval: `I_APPROVE_GITHUB_WRITE`
- Without approval: blocked
- Without title: 400 error
- Returns: success, pr_url, pr_number, gated flag

### PR Status (`GET /api/github/pr/status`)
- Current PR status info
- No secrets in response

## Endpoints NOT Present (by design)

- `POST /api/github/pr/merge` — 404 (no auto-merge)
- `POST /api/github/merge` — 404
- `POST /api/git/force-push` — 404

## Safety Gates

| Operation | Approval Required | Token |
|-----------|-------------------|-------|
| Commit | Yes | `I_APPROVE_GITHUB_WRITE` |
| Push | Yes | `I_APPROVE_GITHUB_WRITE` |
| PR Create | Yes | `I_APPROVE_GITHUB_WRITE` |
| PR Merge | N/A | No endpoint exists |

## Full Dry-Run Example

```bash
# 1. Safety check
curl http://localhost:58000/api/git/safety-check

# 2. Commit proposal
curl -X POST http://localhost:58000/api/git/commit-proposal \
  -H "Content-Type: application/json" \
  -d '{"message": "feat: my change"}'

# 3. PR prepare
curl -X POST http://localhost:58000/api/github/pr/prepare \
  -H "Content-Type: application/json" \
  -d '{"base": "main", "title": "My PR"}'

# 4. PR create (will be blocked without approval)
curl -X POST http://localhost:58000/api/github/pr/create \
  -H "Content-Type: application/json" \
  -d '{"title": "My PR", "body": "...", "base": "main"}'
# Response: {"success": false, "gated": true, ...}

# 5. PR create with approval
curl -X POST http://localhost:58000/api/github/pr/create \
  -H "Content-Type: application/json" \
  -d '{"title": "My PR", "body": "...", "base": "main", "approval": "I_APPROVE_GITHUB_WRITE"}'
```
