from igris.agent.mission import (
    Mission,
    build_final_response,
    evaluate_quality_gate,
    evaluate_satisfaction_gate,
    execute_mission_actions,
    translate_checklist_to_actions,
    understand_and_plan,
    verify_actions,
)


def _build_mission():
    return understand_and_plan(
        user_input=(
            "Modifica il modulo missione, aggiorna la logica di pianificazione, "
            "aggiungi test di regressione e report finale con evidenze complete"
        ),
        project="igrisgpt",
        repo_view={"paths": ["igris/core/x.py"]},
    )


def test_action_translation_and_execution_dry_run():
    mission = translate_checklist_to_actions(_build_mission())
    assert mission.actions
    command_map = {action.id: f"echo ok-{action.id}" for action in mission.actions}
    mission = execute_mission_actions(mission, command_map, dry_run=True)
    assert mission.execution_results
    assert all(result.success for result in mission.execution_results)


def test_execution_blocks_blind_retry_without_differentiator():
    mission = translate_checklist_to_actions(_build_mission())
    command_map = {action.id: "echo same" for action in mission.actions}
    mission = execute_mission_actions(
        mission,
        command_map,
        dry_run=False,
        previous_commands={"echo same"},
        differentiator="",
    )
    assert any(not res.success for res in mission.execution_results)
    assert any("blind retry" in res.stderr for res in mission.execution_results)


def test_quality_and_satisfaction_gate_pass_path():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: f"echo ok-{action.id}" for action in mission.actions},
        dry_run=True,
    )
    quality = evaluate_quality_gate(mission)
    mission.final_response = "architecture verification completed with evidence"
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is True
    assert satisfaction["passed"] is True
    assert mission.status == "completed"


def test_strategic_fail_even_when_technical_passes():
    mission = translate_checklist_to_actions(
        understand_and_plan(
            user_input="Progetta architecture della mission pipeline",
            project="igrisgpt",
        )
    )
    mission = execute_mission_actions(
        mission,
        {action.id: f"echo ok-{action.id}" for action in mission.actions},
        dry_run=True,
    )
    quality = evaluate_quality_gate(mission)
    mission.final_response = "Tests passed."
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is True
    assert satisfaction["passed"] is False
    assert mission.status in {"partial", "failed"}


def test_action_verifier_reports_failures():
    mission = _build_mission()
    mission.execution_results = []
    report = verify_actions(mission)
    assert report["passed"] is True
