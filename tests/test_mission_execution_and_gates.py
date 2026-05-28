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
    assert all(result.evidence.startswith("dry-run") for result in mission.execution_results)
    assert all(result.evidence_depth == "shallow_evidence" for result in mission.execution_results)
    assert all("dry_run_evidence" in result.evidence_tags for result in mission.execution_results)


def test_single_step_sufficient_evidence_can_pass_quality_gate():
    mission = understand_and_plan(
        user_input="Verifica rapidamente lo stato missione",
        project="igrisgpt",
    )
    mission = translate_checklist_to_actions(mission)
    mission.requirements = mission.requirements[:1]
    mission = execute_mission_actions(
        mission,
        {action.id: f"printf ok-{action.id} > /tmp/{action.id}.txt" for action in mission.actions},
        dry_run=False,
    )
    quality = evaluate_quality_gate(mission)
    assert len(mission.checklist) == 1
    assert quality["passed"] is True


def test_single_step_shallow_evidence_forces_quality_fail_and_partial():
    mission = understand_and_plan(
        user_input="Verifica rapidamente lo stato missione",
        project="igrisgpt",
    )
    mission = translate_checklist_to_actions(mission)
    mission = execute_mission_actions(
        mission,
        {action.id: f"echo shallow-{action.id}" for action in mission.actions},
        dry_run=True,
    )
    quality = evaluate_quality_gate(mission)
    mission.final_response = "verification completed with evidence"
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is False
    assert "shallow_evidence" in quality.get("reasons", [])
    assert mission.status == "partial"


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
    assert all(res.evidence == "blocked-blind-retry" for res in mission.execution_results)


def test_execution_allows_retry_with_differentiator():
    mission = translate_checklist_to_actions(_build_mission())
    command_map = {action.id: "echo same" for action in mission.actions}
    mission = execute_mission_actions(
        mission,
        command_map,
        dry_run=True,
        previous_commands={"echo same"},
        differentiator="new-context",
    )
    assert all(res.success for res in mission.execution_results)
    assert all(res.evidence == "dry-run-differentiated" for res in mission.execution_results)


def test_execution_missing_command_is_explicit_failure():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(mission, {}, dry_run=True)
    assert all(not res.success for res in mission.execution_results)
    assert all(res.evidence == "missing-command" for res in mission.execution_results)
    assert all("missing command mapping" in res.stderr for res in mission.execution_results)
    assert all(res.evidence_depth == "missing_evidence" for res in mission.execution_results)
    assert all("missing_evidence" in res.evidence_tags for res in mission.execution_results)


def test_execution_blocks_unsafe_command_by_policy():
    mission = translate_checklist_to_actions(_build_mission())
    command_map = {action.id: "rm -rf /tmp/not-safe" for action in mission.actions}
    mission = execute_mission_actions(mission, command_map, dry_run=False)
    assert all(not res.success for res in mission.execution_results)
    assert all(res.evidence == "blocked-unsafe-command" for res in mission.execution_results)
    assert all("blocked unsafe command" in res.stderr for res in mission.execution_results)


def test_execution_non_dry_run_tracks_returncode():
    mission = translate_checklist_to_actions(_build_mission())
    command_map = {action.id: f"printf ok-{action.id}" for action in mission.actions}
    mission = execute_mission_actions(mission, command_map, dry_run=False)
    assert all(res.returncode == 0 for res in mission.execution_results)
    assert all(res.evidence == "process-executed" for res in mission.execution_results)
    assert all("command_executed" in res.evidence_tags for res in mission.execution_results)


def test_quality_and_satisfaction_gate_pass_path():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(mission, {action.id: f"printf ok-{action.id} > /tmp/{action.id}.txt" for action in mission.actions}, dry_run=False)
    quality = evaluate_quality_gate(mission)
    mission.final_response = "architecture verification completed with evidence in x.py; why unknown and clarified."
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is True
    assert satisfaction["passed"] is True
    assert satisfaction["ready_for_completion"] is True
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
        {action.id: f"printf ok-{action.id} > /tmp/{action.id}.txt" for action in mission.actions},
        dry_run=False,
    )
    quality = evaluate_quality_gate(mission)
    mission.final_response = "Tests passed."
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is True
    assert satisfaction["passed"] is False
    assert mission.status in {"partial", "failed"}


def test_satisfaction_strategic_pass_can_be_partial_if_quality_fails():
    mission = understand_and_plan(
        user_input="Verifica rapidamente la pipeline missione",
        project="igrisgpt",
        repo_view={"paths": ["igris/agent/mission"]},
    )
    mission.quality_gate_passed = False
    mission.final_response = "verification completed with evidence; why unknown"
    satisfaction = evaluate_satisfaction_gate(mission)
    out = build_final_response(mission, {"passed": False}, satisfaction)
    assert satisfaction["passed"] is True
    assert satisfaction["quality_prerequisite_met"] is False
    assert satisfaction["ready_for_completion"] is False
    assert out.status == "partial"


def test_action_verifier_reports_failures():
    mission = _build_mission()
    mission.execution_results = []
    report = verify_actions(mission)
    assert report["passed"] is True
    assert report["evidence_summary"]["insufficient_evidence_actions"] == []


def test_quality_and_verifier_consume_blocked_execution_results():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: "rm -rf /tmp/not-safe" for action in mission.actions},
        dry_run=False,
    )
    verify = verify_actions(mission)
    quality = evaluate_quality_gate(mission)
    assert verify["passed"] is False
    assert quality["passed"] is False


def test_test_command_generates_test_evidence_tags():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: "python3 -m pytest --version >/dev/null 2>&1 || true" for action in mission.actions},
        dry_run=False,
        differentiator="per-action-test-evidence",
    )
    assert all("test_executed" in res.evidence_tags for res in mission.execution_results)
    assert all("test_passed" in res.evidence_tags for res in mission.execution_results)
    assert all(res.evidence_depth == "sufficient_evidence" for res in mission.execution_results)


def test_file_and_report_update_evidence_tags():
    mission = translate_checklist_to_actions(_build_mission())
    command_map = {
        action.id: f"printf report-{action.id} > /tmp/{action.id}.report.json"
        for action in mission.actions
    }
    mission = execute_mission_actions(mission, command_map, dry_run=False)
    assert all("artifact_changed" in res.evidence_tags for res in mission.execution_results)
    assert all("file_updated" in res.evidence_tags for res in mission.execution_results)
    assert all("report_updated" in res.evidence_tags for res in mission.execution_results)
    assert all(res.evidence_depth == "sufficient_evidence" for res in mission.execution_results)


def test_multi_step_shallow_evidence_not_completable():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: f"echo shallow-{action.id}" for action in mission.actions},
        dry_run=True,
    )
    quality = evaluate_quality_gate(mission)
    mission.final_response = "verification completed with evidence in x.py"
    satisfaction = evaluate_satisfaction_gate(mission)
    mission = build_final_response(mission, quality, satisfaction)
    assert quality["passed"] is False
    assert any("insufficient action evidence depth" in g.lower() for g in quality["gaps"])
    assert mission.status == "partial"


def test_multi_step_all_sufficient_evidence_passes_quality():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: f"printf ok-{action.id} > /tmp/{action.id}.txt" for action in mission.actions},
        dry_run=False,
    )
    quality = evaluate_quality_gate(mission)
    assert quality["passed"] is True
    assert quality["reasons"] == []


def test_multi_step_missing_evidence_fails_quality():
    mission = translate_checklist_to_actions(_build_mission())
    partial_map = {mission.actions[0].id: "printf only-one > /tmp/only-one.txt"}
    mission = execute_mission_actions(mission, partial_map, dry_run=False)
    quality = evaluate_quality_gate(mission)
    assert quality["passed"] is False
    assert "missing_evidence" in quality.get("reasons", [])


def test_checklist_item_without_sufficient_action_evidence_fails_quality():
    mission = translate_checklist_to_actions(_build_mission())
    mission = execute_mission_actions(
        mission,
        {action.id: f"echo shallow-{action.id}" for action in mission.actions},
        dry_run=True,
    )
    quality = evaluate_quality_gate(mission)
    assert quality["passed"] is False
    assert "incomplete_checklist_evidence" in quality.get("reasons", [])
