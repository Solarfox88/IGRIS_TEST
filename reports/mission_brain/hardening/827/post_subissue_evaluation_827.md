# Post-Subissue Evaluation — #827 Replay Adoption False-Completed Cases

Status: **passed**

## What Was Executed
- Consolidated replay/delta run created:
  - `scripts/run_mission_brain_hardening_827_replay_delta.py`
  - output: `reports/mission_brain/hardening/827/hardening_827_replay_delta.json`
- Regression pack executed on mission-brain tests.

## Tests Executed
- `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_mission_execution_and_gates.py tests/test_satisfaction_gate_semantics.py tests/test_mission_orchestrator.py tests/test_mission_brain_schema_report.py tests/test_mission_understand_and_plan.py tests/test_mission_validation_runner.py` → **31 passed**

## Replay / Delta Results
- Baseline adoption false_completed_count: **2**
- Replay false_completed_count: **0**
- Delta false_completed_count: **-2**
- critical_false_completed_count remains **0**

### Quality / Satisfaction / Alignment delta proxy
- baseline quality_gate_accuracy: 0.300
- replay quality_gate_accuracy proxy: 1.000
- baseline satisfaction_gate_accuracy: 0.200
- replay satisfaction_gate_accuracy proxy: 1.000
- baseline manual_review_alignment_rate: 0.800
- replay manual_review_alignment_rate proxy: 1.000

### No regression on legitimate completed
- Single-step sufficient-evidence legit case remains `completed`.
- completion policy blocker not triggered in legit completed case.

## Decision for #828
- Decision: **#828 confirmed**.
- Consolidation scope:
  1. use #827 delta as primary evidence for hardening efficacy;
  2. keep final decision constrained to allowed outcomes (`keep wrapper`, `candidate for controlled deeper integration`, `remediate again`);
  3. enforce #828 rule: no `candidate for controlled deeper integration` if `false_completed_count > 0`.
