from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

from igris.agent.mission.mission_schema import Mission


@dataclass
class MissionLoopState:
    family: str
    semantic_key: str
    count: int
    saturated: bool
    escalation_required: bool


def classify_mission_family(mission: Mission) -> str:
    summary = (mission.intent_summary or "").lower()
    if "architecture" in summary:
        return "architecture"
    if "diagnosis" in summary:
        return "diagnosis"
    if "verification" in summary:
        return "verification"
    if "code_change" in summary:
        return "code_change"
    if "planning" in summary:
        return "planning"
    return "mixed"


def semantic_key_for_mission(mission: Mission) -> str:
    basis = " ".join((mission.user_input or "").lower().split())
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return digest[:12]


def evaluate_loop_state(
    mission: Mission,
    counters: Dict[str, int],
    *,
    saturation_threshold: int = 3,
    satisfaction_failures: int = 0,
    escalation_threshold: int = 2,
) -> MissionLoopState:
    family = classify_mission_family(mission)
    key = semantic_key_for_mission(mission)
    count = int(counters.get(family, 0))
    saturated = count >= saturation_threshold
    escalation_required = saturated or satisfaction_failures >= escalation_threshold
    return MissionLoopState(
        family=family,
        semantic_key=key,
        count=count,
        saturated=saturated,
        escalation_required=escalation_required,
    )

