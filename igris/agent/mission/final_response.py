from __future__ import annotations

from typing import Dict

from igris.agent.mission.mission_schema import Mission


def build_final_response(
    mission: Mission,
    quality: Dict[str, object],
    satisfaction: Dict[str, object],
) -> Mission:
    if bool(satisfaction.get("passed")):
        status = "completed"
    elif bool(quality.get("passed")):
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
    mission.final_judgment.reason = mission.final_response
    return mission

