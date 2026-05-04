# Controlled Git Workflow

## Overview

IGRIS_GPT provides a safety-first Git workflow layer. All operations include secret detection, runtime artifact filtering, and sensitive filename blocking. **No push endpoint is exposed.**

## What It Does

- View working tree and staged diffs (with secret redaction)
- List and create branches (with name sanitization)
- Run pre-commit safety checks (secrets, artifacts, sensitive files)
- Generate commit proposals (without actually committing)
- Generate PR summaries comparing branches
- Detect runtime artifacts in changes
- Detect secret-like content in diffs

## What It Does NOT Do

- **No push** — no endpoint for pushing to remote
- **No force operations** — no reset, clean, or destructive git commands
- **No auto-commit** — commits require explicit safety gate
- **No merge/rebase** — not exposed in this version

## Safety Rules

| Check | Description |
|-------|-------------|
| Secret redaction | Diffs are scanned and secrets redacted before display |
| Runtime artifacts | `__pycache__`, `.pytest_cache`, `.venv`, `logs/`, `.igris/`, etc. blocked from staging |
| Sensitive filenames | `.env`, `credentials.json`, `id_rsa`, files containing `key`/`token`/`secret` blocked |
| Branch sanitization | Special characters removed, consecutive dashes collapsed |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/git/status` | Branch, remote, dirty status, changed files |
| `GET` | `/api/git/diff` | Working tree diff (add `?staged=true` for staged) |
| `GET` | `/api/git/diff/stat` | Diffstat summary |
| `GET` | `/api/git/branches` | List local branches |
| `POST` | `/api/git/branch` | Create new branch (sanitized name) |
| `GET` | `/api/git/safety-check` | Pre-commit safety analysis |
| `POST` | `/api/git/commit-proposal` | Generate commit proposal (no actual commit) |
| `GET` | `/api/git/pr-summary` | Compare current branch to base (default: main) |

## UI

The **Git** tab in the agentic console provides:

- Git status with branch, remote, dirty status
- Branch list and create branch form
- Working tree diff with syntax highlighting
- Staged diff viewer
- Safety check button showing warnings
- Commit proposal form
- PR summary generator

## Recommended Workflow

1. **Check status** — View branch, dirty state, changed files
2. **Review diff** — Load working tree or staged diff
3. **Run safety check** — Verify no secrets or artifacts
4. **Create commit proposal** — Preview what would be committed
5. **Stage and commit manually** — Use terminal or external tools
6. **Generate PR summary** — Get commit list and diffstat for PR description
