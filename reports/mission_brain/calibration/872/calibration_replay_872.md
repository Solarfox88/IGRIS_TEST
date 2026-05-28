# Calibration Replay — #872
## Epic #868 Mission Brain Shadow Disagreement Calibration

**Total cycles replayed:** 30 (10 baseline + 20 new)

## Invariant: Agreement Rate

- agreement_rate BEFORE: **0.0**
- agreement_rate AFTER:  **0.0**
- Invariant held: ✅ (calibration does NOT change binary agreement)

## Taxonomy BEFORE (legacy)

| mismatch_class | count |
|----------------|-------|
| safe_more_optimistic_mission_brain | 30 |

## Taxonomy AFTER (calibrated_v1)

| mismatch_class | count |
|----------------|-------|
| scope_mismatch_goal_vs_run_assessment | 17 |
| safe_more_optimistic_mission_brain | 10 |
| expected_divergence_ambiguous_context | 3 |

- scope_mismatch_goal_vs_run_assessment: **17**
- expected_divergence_ambiguous_context: **3**
- legacy (no goal_class — baseline cycles): **10**

## Per-Cycle Taxonomy Change

| cycle_id | goal_class | before | after | changed |
|----------|------------|--------|-------|---------|
| batch1-c1 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch1-c2 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch1-c3 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch1-c4 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch1-c5 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch2-c1 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch2-c2 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch2-c3 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch2-c4 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| batch2-c5 |  | safe_more_optimistic_mission_brain | safe_more_optimistic_mission_brain | False |
| ext-batch1-c11 | policy_check | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c12 | risk_assessment | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c13 | loop_coherence | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c14 | planning | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c15 | test_coverage | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c16 | completion_boundary | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c17 | goal_decomposition | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c18 | git_safety | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c19 | verification | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch1-c20 | memory_saturation | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c21 | policy_check | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c22 | risk_assessment | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c23 | verification | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c24 | ambiguous_goal | safe_more_optimistic_mission_brain | expected_divergence_ambiguous_context | True |
| ext-batch2-c25 | empty_context | safe_more_optimistic_mission_brain | expected_divergence_ambiguous_context | True |
| ext-batch2-c26 | conflicting_signals | safe_more_optimistic_mission_brain | expected_divergence_ambiguous_context | True |
| ext-batch2-c27 | multi_step_complex | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c28 | simple_verification | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c29 | regression_detection | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |
| ext-batch2-c30 | dependency_check | safe_more_optimistic_mission_brain | scope_mismatch_goal_vs_run_assessment | True |

## Safety Gate
- risk_introduced_candidates: 0 ✅
- potential_critical_false_completed: 0 ✅

## Evaluation: passed
