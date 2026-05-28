# Consolidated Calibration Report — #873
## EPIC #868 Mission Brain Shadow Disagreement Calibration — COMPLETE

## Final Decision

### **CALIBRATION_COMPLETE**

All safety gates passed across all 5 subissues. Taxonomy calibration is complete for the 20 new cycles (the scoped dataset). The two new mismatch classes (scope_mismatch_goal_vs_run_assessment, expected_divergence_ambiguous_context) accurately describe the divergence. MB is NOT dangerously wrong — no invariant_failed cases, no risk increase. Shadow monitoring continues unchanged. Default loop behavior unchanged. No integration, no rollout, no enable-by-default.

## Gate Chain

| Subissue | Title | Evaluation |
|----------|-------|------------|
| #869 | Taxonomy Analysis | ✅ passed |
| #870 | Criteria Comparison | ✅ passed |
| #871 | Calibrated Taxonomy | ✅ 46 tests passing |
| #872 | 30-Cycle Replay | ✅ 34 tests passing |
| #873 | Consolidated Report | ✅ this document |

## Key Metrics

- **Total cycles:** 30 (10 baseline + 20 new)
- **agreement_rate:** 0.0 *(structural — expected)*
- **scope_mismatch_count:** 17 (MB and loop measure different scopes)
- **ambiguous_context_count:** 3 (contested — needs 'unknown' response)
- **invariant_failed_count:** 0 ✅ (MB never dangerously wrong)
- **risky_more_optimistic_count:** 0 ✅
- **risk_introduced_candidates:** 0 ✅
- **potential_critical_false_completed:** 0 ✅

## Key Findings

### F1: Scope mismatch is the primary divergence driver

17/20 new cycles classified as scope_mismatch_goal_vs_run_assessment. MB evaluates GOAL-level partial progress; loop evaluates RUN-level binary outcome. These measure different properties — not a real disagreement.
*Impact: low_risk*

### F2: 3 contested cycles identified (ambiguous_goal, empty_context, conflicting_signals)

For 3 goal classes, MB's 'partial' reflects goal ambiguity/missing context, not real partial progress. Calibration recommendation: MB should return 'unknown' or 'insufficient_context' for these cases.
*Impact: low_risk*

### F3: No invariant_failed cases — MB never dangerously wrong

invariant_failed_count=0. MB's 'partial' never creates false 'completed' signal.
*Impact: positive*

### F4: agreement_rate=0.0 is structural, not a bug

All 30 cycles had blocked/failed runs. Loop always returns 'failed' (binary run-level). MB always returns 'partial' (graded goal-level). The 100% disagreement rate reflects this structural scope difference, not MB malfunction.
*Impact: informational*

### F5: 10 baseline cycles lack goal_class — partial taxonomy coverage

legacy_unclassified_count=10. Baseline cycles (847+849) were generated before goal_class field was added. Full 30-cycle taxonomy coverage requires backfilling goal_class.
*Impact: minor_gap*

## Recommendations

### R1: Keep shadow monitoring active — no changes to loop behavior

No safety issues found. Shadow monitoring provides value in classifying goal-level partial progress.

### R2: Calibrate MB to return 'unknown' for ambiguous_goal, empty_context, conflicting_signals

3 contested cases: MB's 'partial' is uninformative when goal specification is insufficient.

### R3: Backfill goal_class for 10 baseline cycles in future sprint

Achieves full 30-cycle calibrated taxonomy coverage.

### R4: Add goal_class to all future shadow cycle records

Required for calibrated taxonomy to work fully. New cycles already have it.

## Guardrails

- shadow_mode_only: ✅
- default_behavior_unchanged: ✅
- no_enable_by_default: ✅
- no_mandatory_gate: ✅
- no_rollout: ✅
- no_integration_without_approval: ✅

## Evaluation: passed | Epic status: complete
