from igris.agent.mission import (
    Mission,
    classify_mission_family,
    evaluate_loop_state,
    semantic_group_for_mission,
    semantic_key_for_mission,
    understand_and_plan,
)


def test_semantic_key_same_for_equivalent_inputs():
    m1 = Mission(user_input="Fix bug in mission planner")
    m2 = Mission(user_input="  fix   bug in mission planner ")
    assert semantic_key_for_mission(m1) == semantic_key_for_mission(m2)


def test_semantic_group_equivalent_phrases_match():
    m1 = Mission(user_input="Fix bug in mission planner quickly")
    m2 = Mission(user_input="Resolve error on planner mission now")
    assert semantic_group_for_mission(m1) == semantic_group_for_mission(m2)
    assert semantic_key_for_mission(m1) == semantic_key_for_mission(m2)


def test_similar_surface_but_different_intent_do_not_collide():
    m1 = understand_and_plan(
        user_input="Verify mission planner behavior",
        project="igrisgpt",
    )
    m2 = understand_and_plan(
        user_input="Fix mission planner behavior",
        project="igrisgpt",
    )
    assert semantic_group_for_mission(m1) != semantic_group_for_mission(m2)
    assert semantic_key_for_mission(m1) != semantic_key_for_mission(m2)


def test_family_saturation_forces_escalation():
    mission = understand_and_plan(
        user_input="Progetta architecture per il mission engine",
        project="igrisgpt",
    )
    state = evaluate_loop_state(mission, {"architecture": 3}, saturation_threshold=3)
    assert state.family == "architecture"
    assert state.saturated is True
    assert state.escalation_required is True


def test_satisfaction_failures_force_escalation():
    mission = Mission(user_input="verify module", intent_summary="[verification] verify module")
    state = evaluate_loop_state(mission, {"verification": 0}, satisfaction_failures=2, escalation_threshold=2)
    assert classify_mission_family(mission) == "verification"
    assert state.escalation_required is True


def test_no_regression_on_non_equivalent_similar_tokens():
    m1 = Mission(user_input="Plan mission roadmap")
    m2 = Mission(user_input="Test mission roadmap")
    assert semantic_key_for_mission(m1) != semantic_key_for_mission(m2)
