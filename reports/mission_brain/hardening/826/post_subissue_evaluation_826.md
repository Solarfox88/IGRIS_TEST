# Post-Subissue Evaluation — #826 Completion Policy / Manual Review Alignment

Status: **passed**

## What Changed
- Added deterministic completion-policy parity guard in final response builder.
- `completed` is now blocked when parity blockers are present, including:
  - quality evidence-policy blockers (`missing_evidence`, `shallow_evidence`, `insufficient_multistep_evidence`, `incomplete_checklist_evidence`)
  - satisfaction diagnostics present
  - not-ready-for-completion flag
- Resulting status falls back to `partial` instead of over-claiming `completed`.

## Tests Executed
- `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_mission_execution_and_gates.py tests/test_satisfaction_gate_semantics.py` → **23 passed**
- `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_mission_orchestrator.py tests/test_mission_brain_schema_report.py tests/test_mission_understand_and_plan.py tests/test_mission_validation_runner.py` → **8 passed**

## Replay / Delta
- Replay artifact: `reports/mission_brain/hardening/826/hardening_826_replay.json`
- Adoption false-completed sources (`#791 M2`, `#792 M4`) remain non-completed (`partial`).
- Manual-policy parity case:
  - `quality_passed=true`, `satisfaction_passed=true`, diagnostics present
  - final status is forced to `partial`
  - completion policy blocker marker present in judgment reason
- `false_completed_count=0`
- `critical_false_completed_count=0`

## What Is Now Measurable
- Explicit completion-policy blocker trace is present in final judgment reason.
- Manual-review parity guard can be validated by replay and unit tests.

## #827 Propagation Decision
- Decision: **#827 confirmed**.
- Scope refinement for #827:
  1. replay adoption false-completed set with post-#824/#825/#826 logic;
  2. produce clear before/after delta across requested metrics;
  3. verify no regression on previously correct completed paths.
