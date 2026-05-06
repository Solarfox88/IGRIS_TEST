"""Tests for Epic #43 — GOAP-like Planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from igris.core.goap_planner import (
    ACTION_FAMILIES,
    GOAPAction,
    GOAPPlan,
    GOAPPlanner,
    WorldState,
)


# ===========================================================================
# WorldState
# ===========================================================================


class TestWorldState:
    def test_default_state(self):
        s = WorldState.default_state()
        assert s.get(WorldState.BUDGET_OK) is True
        assert s.get(WorldState.TESTS_PASS) == "unknown"

    def test_set_and_get(self):
        s = WorldState()
        s.set("x", 42)
        assert s.get("x") == 42
        assert s.get("y") is None

    def test_satisfies(self):
        s = WorldState(properties={"a": True, "b": 1})
        assert s.satisfies({"a": True}) is True
        assert s.satisfies({"a": True, "b": 1}) is True
        assert s.satisfies({"a": False}) is False

    def test_apply_effects(self):
        s = WorldState(properties={"a": 1})
        s2 = s.apply_effects({"b": 2})
        assert s2.get("a") == 1
        assert s2.get("b") == 2
        assert s.get("b") is None  # Original unchanged

    def test_to_dict_from_dict(self):
        s = WorldState(properties={"x": True})
        d = s.to_dict()
        s2 = WorldState.from_dict(d)
        assert s2.get("x") is True


# ===========================================================================
# GOAPAction
# ===========================================================================


class TestGOAPAction:
    def test_to_dict(self):
        a = GOAPAction(title="Test", family="observation", risk="low")
        d = a.to_dict()
        assert d["title"] == "Test"
        assert d["risk"] == "low"
        assert d["id"].startswith("act-")

    def test_from_dict(self):
        d = {"title": "X", "family": "code_patch", "risk": "medium", "cost": 2.0}
        a = GOAPAction.from_dict(d)
        assert a.title == "X"
        assert a.cost == 2.0

    def test_cooldown_family_defaults(self):
        a = GOAPAction(family="test_repair")
        d = a.to_dict()
        assert d["cooldown_family"] == "test_repair"

    def test_preconditions_effects(self):
        a = GOAPAction(
            preconditions={"repo_analyzed": True},
            effects={"tests_pass": True},
        )
        s = WorldState(properties={"repo_analyzed": True})
        assert s.satisfies(a.preconditions)
        s2 = s.apply_effects(a.effects)
        assert s2.get("tests_pass") is True


# ===========================================================================
# GOAPPlan
# ===========================================================================


class TestGOAPPlan:
    def test_to_dict(self):
        p = GOAPPlan(goal={"done": True}, actions=[GOAPAction(title="A")])
        d = p.to_dict()
        assert d["action_count"] == 1
        assert d["goal"] == {"done": True}

    def test_from_dict(self):
        p = GOAPPlan(
            goal={"x": True},
            actions=[GOAPAction(title="A"), GOAPAction(title="B")],
            total_cost=5.0,
        )
        d = p.to_dict()
        p2 = GOAPPlan.from_dict(d)
        assert len(p2.actions) == 2
        assert p2.total_cost == 5.0


# ===========================================================================
# GOAPPlanner — Plan generation
# ===========================================================================


class TestGOAPPlannerGenerate:
    def test_generate_plan_with_goal(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        goal = {"tests_pass_after": True}
        state = WorldState(properties={})
        plan = planner.generate_plan(goal=goal, state=state)
        assert len(plan.actions) > 0
        assert plan.status == "created"

    def test_generate_plan_empty_goal(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        state = WorldState(properties={})
        plan = planner.generate_plan(goal={}, state=state)
        # Empty goal is immediately satisfied, may have fallback
        assert plan.status == "created"

    def test_generate_plan_fallback(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path), action_library=[])
        plan = planner.generate_plan(goal={"x": True}, state=WorldState())
        assert plan.status == "created"

    def test_plan_cost_calculated(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        goal = {"commit_ready": True}
        state = WorldState(properties={})
        plan = planner.generate_plan(goal=goal, state=state)
        assert plan.total_cost > 0


class TestGOAPPlannerEligible:
    def test_eligible_from_empty_state(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        state = WorldState(properties={})
        eligible = planner.get_eligible_actions(state)
        assert len(eligible) > 0
        # Only actions with no preconditions should be eligible
        for a in eligible:
            assert state.satisfies(a.preconditions)

    def test_blocked_actions_excluded(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        planner._blocked_actions.add("analyze-repo")
        state = WorldState(properties={})
        eligible = planner.get_eligible_actions(state)
        assert all(a.id != "analyze-repo" for a in eligible)

    def test_saturated_family_excluded(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        planner._saturated_families["observation"] = 3
        state = WorldState(properties={})
        eligible = planner.get_eligible_actions(state)
        assert all(a.family != "observation" for a in eligible)


# ===========================================================================
# GOAPPlanner — Replanning
# ===========================================================================


class TestGOAPPlannerReplan:
    def test_replan_blocks_failed_action(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        goal = {"tests_pass_after": True}
        state = WorldState(properties={})
        plan = planner.generate_plan(goal=goal, state=state)
        first_action_id = plan.actions[0].id if plan.actions else "none"

        new_plan = planner.replan_after_failure(plan, first_action_id, "test failed")
        assert new_plan.replan_count == 1
        assert first_action_id in planner._blocked_actions

    def test_replan_increments_saturation(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        goal = {"tests_pass_after": True}
        state = WorldState(properties={})
        plan = planner.generate_plan(goal=goal, state=state)
        if plan.actions:
            action = plan.actions[0]
            planner.replan_after_failure(plan, action.id)
            family = action.cooldown_family or action.family
            assert planner._saturated_families.get(family, 0) >= 1


# ===========================================================================
# GOAPPlanner — LLM validation
# ===========================================================================


class TestGOAPPlannerValidate:
    def test_valid_plan(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = {
            "actions": [
                {"title": "A", "family": "observation", "risk": "low",
                 "success_criteria": ["done"]},
            ]
        }
        plan = planner.validate_llm_plan(raw)
        assert plan is not None
        assert len(plan.actions) == 1

    def test_invalid_no_actions(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        assert planner.validate_llm_plan({}) is None

    def test_invalid_no_title(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = {"actions": [{"family": "other"}]}
        assert planner.validate_llm_plan(raw) is None

    def test_invalid_no_success_criteria(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = {"actions": [{"title": "X", "risk": "low"}]}
        assert planner.validate_llm_plan(raw) is None

    def test_invalid_risk(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = {"actions": [{"title": "X", "risk": "extreme", "success_criteria": ["x"]}]}
        assert planner.validate_llm_plan(raw) is None

    def test_invalid_json_string(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        assert planner.validate_llm_plan("not json") is None

    def test_valid_json_string(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = json.dumps({
            "actions": [{"title": "A", "risk": "low", "success_criteria": ["ok"]}]
        })
        plan = planner.validate_llm_plan(raw)
        assert plan is not None

    def test_unknown_family_defaults(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        raw = {"actions": [{"title": "X", "family": "alien", "risk": "low", "success_criteria": ["x"]}]}
        plan = planner.validate_llm_plan(raw)
        assert plan is not None
        assert plan.actions[0].family == "other"


# ===========================================================================
# GOAPPlanner — Explain
# ===========================================================================


class TestGOAPPlannerExplain:
    def test_explain_plan(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        plan = planner.generate_plan(goal={"repo_analyzed": True}, state=WorldState())
        explanation = planner.explain_plan(plan)
        assert "steps" in explanation
        assert "total_steps" in explanation

    def test_explain_next_action(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        plan = planner.generate_plan(goal={"repo_analyzed": True}, state=WorldState())
        info = planner.explain_next_action(plan)
        if plan.actions:
            assert "action" in info
        else:
            assert "message" in info

    def test_explain_next_when_complete(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        plan = GOAPPlan(actions=[])
        plan.current_step = 0
        info = planner.explain_next_action(plan)
        assert "message" in info


# ===========================================================================
# GOAPPlanner — Persistence
# ===========================================================================


class TestGOAPPlannerPersistence:
    def test_save_and_load(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        plan = planner.generate_plan(goal={"repo_analyzed": True}, state=WorldState())
        planner.save_plan(plan)
        loaded = planner.load_plan(plan.id)
        assert loaded is not None
        assert loaded.id == plan.id

    def test_load_nonexistent(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        assert planner.load_plan("plan-nope") is None

    def test_list_plans(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        p1 = planner.generate_plan(goal={"x": True}, state=WorldState(), mission_id="m1")
        p2 = planner.generate_plan(goal={"y": True}, state=WorldState(), mission_id="m2")
        planner.save_plan(p1)
        planner.save_plan(p2)
        plans = planner.list_plans()
        assert len(plans) == 2

    def test_list_plans_filter(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        p1 = planner.generate_plan(goal={"x": True}, state=WorldState(), mission_id="m1")
        p2 = planner.generate_plan(goal={"y": True}, state=WorldState(), mission_id="m2")
        planner.save_plan(p1)
        planner.save_plan(p2)
        plans = planner.list_plans(mission_id="m1")
        assert len(plans) == 1


# ===========================================================================
# Full lifecycle
# ===========================================================================


class TestGOAPLifecycle:
    def test_full_plan_execute_replan(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        state = WorldState(properties={})
        goal = {"tests_pass_after": True}

        # Generate plan
        plan = planner.generate_plan(goal=goal, state=state)
        assert len(plan.actions) > 0

        # Simulate executing first action
        first = plan.actions[0]
        state = state.apply_effects(first.effects)

        # Simulate failure on second action
        if len(plan.actions) > 1:
            failed = plan.actions[1]
            new_plan = planner.replan_after_failure(plan, failed.id, "test error")
            assert new_plan.replan_count == 1
            assert failed.id in planner._blocked_actions

    def test_deterministic_fallback_works(self, tmp_path):
        planner = GOAPPlanner(project_root=str(tmp_path))
        # Use a goal that requires many steps
        goal = {"report_generated": True}
        state = WorldState(properties={})
        plan = planner.generate_plan(goal=goal, state=state)
        assert len(plan.actions) > 0
        # All actions should have success_criteria
        for action in plan.actions:
            assert len(action.success_criteria) > 0


# ===========================================================================
# API integration
# ===========================================================================


class TestGOAPAPI:
    @pytest.fixture
    def client(self):
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_get_state(self, client):
        resp = client.get("/api/goap/state")
        assert resp.status_code == 200
        assert "properties" in resp.json()

    def test_generate_plan(self, client):
        resp = client.post("/api/goap/plan", json={
            "goal": {"repo_analyzed": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert data["status"] == "created"

    def test_list_plans(self, client):
        resp = client.get("/api/goap/plans")
        assert resp.status_code == 200

    def test_eligible_actions(self, client):
        resp = client.post("/api/goap/eligible-actions", json={
            "properties": {},
        })
        assert resp.status_code == 200
        assert "actions" in resp.json()

    def test_validate_valid_plan(self, client):
        resp = client.post("/api/goap/validate-llm-plan", json={
            "actions": [{"title": "A", "risk": "low", "success_criteria": ["ok"]}],
        })
        assert resp.json()["valid"] is True

    def test_validate_invalid_plan(self, client):
        resp = client.post("/api/goap/validate-llm-plan", json={
            "actions": [{"family": "other"}],
        })
        assert resp.json()["valid"] is False

    def test_plan_not_found(self, client):
        resp = client.get("/api/goap/plans/plan-nope")
        assert resp.status_code == 404
