from __future__ import annotations

from typing import Dict, List

from igris.agent.mission.mission_schema import Mission


def verify_actions(mission: Mission) -> Dict[str, object]:
    failures: List[Dict[str, str]] = []
    for result in mission.execution_results:
        if not result.success:
            failures.append(
                {
                    "action_id": result.action_id,
                    "failure_type": "technical_failure",
                    "reason": result.stderr or "unknown error",
                }
            )
    return {
        "passed": len(failures) == 0,
        "technical_failures": failures,
        "strategic_failures": [],
    }

