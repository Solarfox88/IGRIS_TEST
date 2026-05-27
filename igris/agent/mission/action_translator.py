from __future__ import annotations

from typing import List

from igris.agent.mission.mission_schema import Mission, MissionAction


def translate_checklist_to_actions(mission: Mission) -> Mission:
    actions: List[MissionAction] = []
    for idx, item in enumerate(mission.checklist, start=1):
        actions.append(
            MissionAction(
                id=f"ACT-{idx:03d}",
                description=f"Produce evidence for checklist {item.id}: {item.description}",
                linked_checklist_ids=[item.id],
                expected_outcome=f"Checklist {item.id} marked with evidence",
                unsafe=False,
            )
        )
    mission.actions = actions
    return mission

