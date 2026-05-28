# Mission Brain Operational Adoption — Consolidated Decision Report (#794)

Parent context:
- #746 Mission Brain MVP implementation
- #768 Mission Brain MVP validation
- #778 Mission Brain MVP remediation
- #789 Operational adoption epic
- #790 protocol
- #791, #792, #793 execution batches

## Summary 10/10 Missions
- total_missions: 10
- completed_count: 3
- partial_count: 7
- failed_count: 0
- false_completed_count: 2
- critical_false_completed_count: 0
- false_partial_count: 1
- false_failed_count: 0
- quality_gate_accuracy: 0.300
- satisfaction_gate_accuracy: 0.200
- manual_review_alignment_rate: 0.800
- average_report_usefulness_score: 0.747

## Final Adoption Decision
- adoption_decision: `keep wrapper`

Reasoning:
1. `false_completed_count=2` on 10 missions is recurring and blocks deep promotion.
2. `critical_false_completed_count=0` avoids hard-stop remediation mode.
3. `manual_review_alignment_rate=0.800` and report usefulness remain good enough for supervised wrapper use.
4. Gate accuracies (`quality=0.300`, `satisfaction=0.200`) are not sufficient for irreversible deep integration.

## Residual Risks
1. Wrapper-only risk: non-critical but recurring over-claim risk on completion.
2. Multi-step completion can be declared with shallow evidence chains.
3. Quality gate and satisfaction gate are too permissive for certain multi-step scenarios.
4. Policy mismatch risk between manual review standards and automatic completion policy.

## Prioritized Remediation Backlog
1. Evidence depth for action execution
- Introduce minimum evidence depth criteria per multi-step mission before `completed`.
- Require explicit end-to-end chain validation (not only local command success).

2. Stricter quality gate for multi-step completion
- Add deterministic thresholding for multi-step completion claims.
- Block `completed` when checklist evidence lacks cross-step linkage.

3. Alignment between manual review and completion policy
- Encode manual review strictness into deterministic completion policy checks.
- Add explicit “manual-policy parity” assertions in adoption reports.

## Conditions to Reconsider `integrate deeper`
All conditions below must be met in a new controlled validation window:
1. false_completed_count reduced to `0` (or `<=1` fully explained and non-recurring over multiple runs).
2. critical_false_completed_count remains `0`.
3. quality_gate_accuracy materially improved (target >= 0.70).
4. satisfaction_gate_accuracy materially improved (target >= 0.70).
5. manual_review_alignment_rate remains >= 0.80.
6. No irreversible deep-loop promotion until these criteria are observed on fresh real missions.

## Operational Policy Now
1. Keep Mission Brain as wrapper/evaluation assistant.
2. Do not promote Mission Brain as automatic gate of the main operational loop.
3. Use remediation backlog above before any deeper integration decision.
