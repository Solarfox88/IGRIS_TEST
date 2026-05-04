# Autonomous Execution Loop MVP

## Overview

The autonomous loop implements a safety-first semi-autonomous execution cycle:

```
task selection → action decision → safe execution → report → outcome routing → memory → next step
```

All actions are bounded by `max_steps`, safety checks, and family saturation. No auto-commit, no auto-push, no unsafe patch apply.

## What It Does

- **Select next task**: Uses task selection with memory constraints
- **Decide action**: Maps task family to safe command_id or patch proposal
- **Execute safely**: Only via whitelisted command_ids (run_tests, git_status, list_files)
- **Report results**: Creates execution reports and timeline events
- **Route outcomes**: Determines next action based on result (remediation if needed)
- **Record memory**: Stores decisions, failures, saturation in decision memory

## What It Does NOT Do

- No auto-commit or auto-push
- No auto-patch apply (proposes via UI only)
- No shell execution (only command_id)
- No infinite loops (max_steps capped at 100)
- No destructive actions
- No execution of high-risk tasks

## Safety Guards

1. **max_steps required**: Capped at 100, minimum 1
2. **High-risk skip**: Tasks with risk="high" are skipped automatically
3. **Family saturation**: Saturated families blocked via decision memory
4. **Consecutive failure stop**: 3 consecutive failures stops the loop
5. **Skip accumulation stop**: 3+ skipped/blocked steps stops the loop
6. **No auto-commit/push**: Git tasks only run `git_status`
7. **No auto-patch**: Code/fix tasks propose patches but don't apply
8. **Secret redaction**: All step results redacted before return

## Action Mapping

| Task Family | Action | Command |
|-------------|--------|---------|
| test | execute_command | run_tests |
| analyze (with "file"/"list") | execute_command | list_files |
| git, config (with "status"/"git") | execute_command | git_status |
| code, fix, refactor, docs | propose_patch | (manual review) |
| other | skip | — |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/loop/step` | Execute single step |
| `POST` | `/api/loop/run` | Run N steps (`{"max_steps": N}`) |
| `GET` | `/api/loop/status` | Current loop status |
| `GET` | `/api/loop/recent` | Recent step results |

### POST /api/loop/run Body

```json
{
  "max_steps": 5
}
```

## UI

The **Loop** tab provides:
- Run 1/3/5 Steps buttons
- Status display (running, steps completed, stop reason)
- Recent steps list with outcome badges

## Workflow

1. Create tasks via Mission Planner or Tasks tab
2. Go to Loop tab
3. Click "Run 1 Step" or "Run N Steps"
4. Loop selects next task, executes safe command
5. Results shown with outcome badges
6. Failed outcomes create remediation tasks
7. Memory constraints prevent loops

## Integration

- **Task Selection**: Selects pending tasks respecting memory constraints
- **Decision Memory**: Records all decisions and failures
- **Outcome Router**: Routes results to next recommended action
- **Teacher**: Creates remediation tasks on failures
- **Timeline**: All steps logged as timeline events
