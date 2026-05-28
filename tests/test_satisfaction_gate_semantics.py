from igris.agent.mission import (
    build_final_response,
    evaluate_satisfaction_gate,
    understand_and_plan,
)


def test_satisfaction_gate_handles_unknown_why_with_explicit_acknowledgement():
    mission = understand_and_plan(
        user_input="Verifica rapidamente la pipeline missione",
        project="igrisgpt",
        repo_view={"paths": ["igris/agent/mission"]},
    )
    mission.quality_gate_passed = True
    mission.final_response = (
        "verification completed on mission files; why unknown and pending clarification."
    )
    out = evaluate_satisfaction_gate(mission)
    assert out["passed"] is True
    assert out["ready_for_completion"] is True


def test_satisfaction_gate_rejects_false_positive_style_response():
    mission = understand_and_plan(
        user_input="Progetta architecture della mission pipeline",
        project="igrisgpt",
        repo_view={"paths": ["igris/agent/mission/mission_orchestrator.py"]},
    )
    mission.quality_gate_passed = True
    mission.final_response = "Tests passed."
    out = evaluate_satisfaction_gate(mission)
    assert out["passed"] is False
    assert any("intent type" in gap for gap in out["gaps"])


def test_satisfaction_gate_detects_diagnostics_without_hard_fail():
    mission = understand_and_plan(
        user_input="Diagnostica bug in igris/core/mission_planner.py",
        project="igrisgpt",
        repo_view={"paths": ["igris/core/mission_planner.py"]},
    )
    mission.quality_gate_passed = True
    mission.final_response = "Diagnosis completed with root cause and fix evidence."
    out = evaluate_satisfaction_gate(mission)
    assert out["passed"] is True
    assert isinstance(out["diagnostics"], list)


def test_completion_policy_blocks_completed_when_diagnostics_present():
    mission = understand_and_plan(
        user_input="Diagnostica bug in igris/core/mission_planner.py",
        project="igrisgpt",
        repo_view={"paths": ["igris/core/mission_planner.py"]},
    )
    mission.quality_gate_passed = True
    mission.final_response = "Diagnosis completed with root cause and fix evidence."
    sat = evaluate_satisfaction_gate(mission)
    out = build_final_response(mission, {"passed": True, "reasons": []}, sat)
    assert sat["passed"] is True
    assert sat["diagnostics"]
    assert out.status == "partial"
