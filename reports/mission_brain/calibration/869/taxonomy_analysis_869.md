# Taxonomy Analysis — #869
## Epic #868 Mission Brain Shadow Disagreement Calibration

## Finding: Scope Mismatch (Goal-level vs Run-level)

| System | What it evaluates | Decision type |
|--------|-------------------|---------------|
| Mission Brain | Was the GOAL partially achieved? | Goal-level, graded |
| Current Loop | Did this RUN ATTEMPT succeed? | Run-level, binary |

These are **not disagreeing** about the same thing.
They measure **different properties** of the same event.

## Contingency Table (20 new cycles)

| MB decision | Loop decision | Count |
|-------------|---------------|-------|
| partial | failed | 20 |

## Root Cause Breakdown

- **scope_mismatch_goal_vs_run_assessment**: 17 cycles
  (MB measures goal progress, loop measures run success — measuring different things)
- **expected_divergence_ambiguous_context**: 3 cycles
  (goal ambiguity/empty context — MB's partial is uncertain, not informative)

## Safety Gate

- risk_introduced_candidates: 0 ✅
- potential_critical_false_completed: 0 ✅
- rollback_path_status: ok ✅

## Recommendation

Proceed to #870 (criteria comparison). Calibration path is safe.
Proposed taxonomy reclassification: 20/20 cycles.

## Evaluation: passed
