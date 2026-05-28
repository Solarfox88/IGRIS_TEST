from __future__ import annotations

from typing import Dict, List

from igris.agent.mission.mission_schema import Mission


def _completion_policy_blockers(
    quality: Dict[str, object],
    satisfaction: Dict[str, object],
) -> List[str]:
    blockers: List[str] = []
    quality_reasons = set(str(r) for r in (quality.get("reasons") or []))
    if quality_reasons.intersection(
        {
            "missing_evidence",
            "shallow_evidence",
            "insufficient_multistep_evidence",
            "incomplete_checklist_evidence",
        }
    ):
        blockers.append("quality_evidence_policy_block")
    diagnostics = satisfaction.get("diagnostics") or []
    if diagnostics:
        blockers.append("satisfaction_diagnostics_present")
    if not bool(satisfaction.get("ready_for_completion", False)):
        blockers.append("not_ready_for_completion")
    return blockers


def build_final_response(
    mission: Mission,
    quality: Dict[str, object],
    satisfaction: Dict[str, object],
) -> Mission:
    completion_blockers = _completion_policy_blockers(quality, satisfaction)

    if (
        bool(quality.get("passed"))
        and bool(satisfaction.get("passed"))
        and not completion_blockers
    ):
        status = "completed"
    elif bool(quality.get("passed")) or bool(satisfaction.get("passed")):
        status = "partial"
    else:
        status = "failed"
    mission.status = status
    mission.final_response = (
        f"Mission {mission.id} status={status}; "
        f"quality_passed={quality.get('passed')}; "
        f"satisfaction_passed={satisfaction.get('passed')}."
    )
    mission.final_judgment.technical_status = "passed" if quality.get("passed") else "failed"
    mission.final_judgment.strategic_status = "passed" if satisfaction.get("passed") else "failed"
    if completion_blockers:
        mission.final_judgment.reason = mission.final_response + " completion_policy_blockers=" + ",".join(completion_blockers)
    else:
        mission.final_judgment.reason = mission.final_response
    return mission
