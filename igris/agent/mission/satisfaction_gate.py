from __future__ import annotations

from typing import Dict, List

from igris.agent.mission.mission_schema import Mission


def evaluate_satisfaction_gate(mission: Mission) -> Dict[str, object]:
    gaps: List[str] = []
    if not mission.intent_summary or mission.intent_summary == "unknown":
        gaps.append("Intent summary missing")
    if not mission.final_response.strip():
        gaps.append("Final response missing")

    normalized_intent = mission.intent_summary.lower()
    normalized_response = mission.final_response.lower()
    if normalized_intent and normalized_response:
        for token in ("architecture", "diagnosis", "verification", "code_change", "planning"):
            if token in normalized_intent and token not in normalized_response:
                gaps.append(f"Final response does not reflect intent type '{token}'")

    passed = len(gaps) == 0 and mission.quality_gate_passed
    mission.satisfaction_gate_passed = passed
    return {"passed": passed, "gaps": gaps}

