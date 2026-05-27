# Mission Brain Operational Adoption Protocol

Epic context:
- #746 Mission Brain MVP implementation
- #768 Mission Brain MVP validation
- #778 Mission Brain MVP remediation
- #789 Mission Brain Operational Adoption / Controlled Integration

## Purpose
Define a controlled, comparable, and auditable protocol for operational adoption of Mission Brain on at least 10 real (non-simulated) missions.

This phase does **not** introduce deep irreversible integration into the main loop.

## Scope of This Protocol
1. Mission selection criteria (10 real missions)
2. Per-mission report template
3. Aggregate metrics schema
4. Classification rules: completed / partial / failed
5. Critical false completed definition
6. Manual review criteria
7. Consolidated adoption report format
8. Final decision rules:
   - integrate deeper
   - keep wrapper
   - remediate again

## Mission Selection Criteria (10 Real Missions)
Required coverage:
- 2 simple requests
- 2 multi-step requests
- 1 multi-file change
- 1 bug diagnosis
- 1 architecture request
- 1 ambiguous mission
- 1 mission with predictable technical failure risk
- 1 mission with false-completed / intent-mismatch risk

Mission candidate quality constraints:
- Must come from real operational backlog or user requests.
- Must include observable outcomes (not purely subjective).
- Must be executable with current wrapper mode (no irreversible core-loop integration).

## Per-Mission Data Requirements
For each mission collect:
- user_input
- available_context
- mission_brain_report_path
- declared_status (completed/partial/failed)
- observable_outcome
- manual_reviewer_judgment
- discrepancy_present (boolean)
- discrepancy_cause
- recommended_follow_up
- runtime_overhead_note

## Classification Rules
- `completed`: required outcomes achieved, no blocking quality/satisfaction gaps, manual review agrees.
- `partial`: some value delivered, unresolved gaps remain, manual review can explain limitations.
- `failed`: objective mission outcome not achieved or result unusable for next step.

## False Classification Rules
- `false_completed`: declared completed but observable outcome/manual review indicates incomplete or incorrect mission resolution.
- `critical_false_completed`: false_completed that could trigger wrong operational decision, hidden production risk, or loss of rollback/debug traceability.
- `false_partial`: declared partial when mission is effectively completed or failed.
- `false_failed`: declared failed when mission outcome is actually acceptable/complete.

## Manual Review Criteria
Manual review must score:
1. outcome correctness vs input intent
2. evidence sufficiency
3. next-step usability of report
4. risk visibility (especially hidden failures)

Review outcome:
- aligned
- partially_aligned
- misaligned

## Consolidated Adoption Report Format
The consolidated report must include:
- mission list with evidence links
- declared vs observed matrix
- false classification analysis
- quality_gate_accuracy
- satisfaction_gate_accuracy
- manual review alignment summary
- operational slowdown summary
- adoption decision with rationale

## Mandatory Aggregate Metrics
- total_missions
- completed_count
- partial_count
- failed_count
- false_completed_count
- critical_false_completed_count
- false_partial_count
- false_failed_count
- satisfaction_gate_accuracy
- quality_gate_accuracy
- manual_review_alignment_rate
- average_report_usefulness_score
- adoption_decision

## Final Decision Rules
`integrate deeper` only if all are true:
- critical_false_completed_count == 0
- false_completed_count == 0 or non-critical and explicitly explained
- partial outcomes are useful and explainable
- failed outcomes map to real failures
- reports are decision-useful
- no severe operational regressions
- at least 10 real missions evaluated with evidence

`remediate again` if any are true:
- critical_false_completed_count >= 1
- gate judgments are not explainable
- reports do not support operational decisions
- Mission Brain masks real failures

`keep wrapper` when:
- no critical false completed
- system is useful
- but confidence is insufficient for deep integration

## Governance
Each subissue in #789 must include:
- Post-subissue evaluation
- Next-subissue propagation decision
- explicit status: passed / partial / failed / blocked

