# GOAP-like Planner — Epic #43

Goal-Oriented Action Planning based on state, preconditions, effects,
risk, cost, and success criteria.

## Key Concepts

- **WorldState** — observable environment properties (repo clean, tests pass, etc.)
- **GOAPAction** — action with preconditions, effects, risk, cost, success criteria
- **GOAPPlan** — ordered sequence of actions to reach a goal
- **GOAPPlanner** — generates/validates plans, supports replanning

## Action Families

`observation`, `synthesis`, `repo_diff_discovery`, `patch_strategy`,
`branch_pr_plan`, `review_gate`, `candidate_materialization`,
`mastery_cycle`, `mastery_gate`, `school_report`, `grading_diagnosis`,
`stabilization_audit`, `devops_deploy`, `server_diagnosis`,
`test_repair`, `code_patch`, `documentation`, `security_audit`, `other`

## Planning Algorithm

Forward chaining from current state:
1. Get eligible actions (preconditions met, not blocked, not saturated)
2. Score each by goal progress, cost, risk, family saturation
3. Select best action, apply effects to state
4. Repeat until goal satisfied or no eligible actions
5. If empty → fallback to standard sequential plan

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/goap/state` | Current world state |
| `POST` | `/api/goap/plan` | Generate plan for goal |
| `GET` | `/api/goap/plans` | List plans |
| `GET` | `/api/goap/plans/{id}` | Get plan |
| `GET` | `/api/goap/plans/{id}/explain` | Explain plan |
| `GET` | `/api/goap/plans/{id}/next` | Explain next action |
| `POST` | `/api/goap/eligible-actions` | Eligible actions for state |
| `POST` | `/api/goap/validate-llm-plan` | Validate LLM plan output |
| `POST` | `/api/goap/replan` | Replan after failure |

## Replanning

After failure: blocks failed action, increments family saturation,
generates new plan. After 3 repetitions of same family, family is
saturated and excluded.

## LLM Plan Validation

LLM output must be JSON with `actions` array. Each action requires:
`title`, `risk` (low/medium/high/critical), `success_criteria` (non-empty).
Invalid plans are rejected → deterministic fallback.

## File Layout

```
igris/core/goap_planner.py    — Planner logic
tests/test_goap_planner.py    — 44 tests
docs/GOAP_PLANNER.md          — This file
```
