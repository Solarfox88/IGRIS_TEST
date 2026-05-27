from __future__ import annotations

from typing import Dict, Optional

from igris.agent.mission.action_translator import translate_checklist_to_actions
from igris.agent.mission.action_verifier import verify_actions
from igris.agent.mission.execution_adapter import execute_mission_actions
from igris.agent.mission.final_response import build_final_response
from igris.agent.mission.mission_report import save_mission_report
from igris.agent.mission.mission_schema import Mission
from igris.agent.mission.quality_gate import evaluate_quality_gate
from igris.agent.mission.satisfaction_gate import evaluate_satisfaction_gate
from igris.agent.mission.understand_and_plan import understand_and_plan


def run_mission_pipeline(
    *,
    user_input: str,
    project: str = "igrisgpt",
    repo_view: Optional[Dict[str, object]] = None,
    command_map: Optional[Dict[str, str]] = None,
    dry_run: bool = True,
    project_root: str = ".",
) -> Mission:
    mission = understand_and_plan(user_input=user_input, project=project, repo_view=repo_view)
    mission = translate_checklist_to_actions(mission)
    mission = execute_mission_actions(mission, command_map or {}, dry_run=dry_run)
    verify_actions(mission)
    quality = evaluate_quality_gate(mission)
    if not mission.final_response:
        mission.final_response = mission.intent_summary
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    save_mission_report(mission, project_root=project_root)
    return mission

