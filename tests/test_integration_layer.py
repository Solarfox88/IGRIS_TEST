"""Tests for Integration Layer — Epic #62.

Validates the unified pipeline connecting Mission Controller,
GOAP Planner, Context Manager, Agent Reasoning Loop, Tool Runtime,
Safety/Rollback, Decision Memory, and Teacher/Governor.
"""

import pytest
import os
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from igris.core.integration_layer import (
    IntegrationLayer,
    DecisionReport,
    MissionReport,
)
from igris.core.agent_action_schema import AgentAction


# ---------------------------------------------------------------------------
# DecisionReport
# ---------------------------------------------------------------------------

class TestDecisionReport:
    """Test DecisionReport dataclass."""

    def test_default(self):
        dr = DecisionReport()
        assert dr.step_index == 0
        assert dr.governor_decision == "approve"

    def test_to_dict(self):
        dr = DecisionReport(
            step_index=1,
            tool_used="code_navigation",
            risk_level="low",
        )
        d = dr.to_dict()
        assert d["step_index"] == 1
        assert d["tool_used"] == "code_navigation"

    def test_to_dict_redacts_secrets(self):
        fake = "sk-" + "a" * 30
        dr = DecisionReport(tool_result=f"key={fake}")
        d = dr.to_dict()
        assert fake not in d["tool_result"]
        assert "REDACTED" in d["tool_result"]


# ---------------------------------------------------------------------------
# MissionReport
# ---------------------------------------------------------------------------

class TestMissionReport:
    """Test MissionReport dataclass."""

    def test_default(self):
        mr = MissionReport()
        assert mr.status == "pending"
        assert mr.total_steps == 0

    def test_to_dict(self):
        mr = MissionReport(
            mission_id="m-test",
            goal="Test goal",
            status="completed",
            total_steps=3,
            successful_steps=2,
            failed_steps=1,
        )
        d = mr.to_dict()
        assert d["mission_id"] == "m-test"
        assert d["status"] == "completed"
        assert d["total_steps"] == 3

    def test_to_dict_redacts_secrets(self):
        fake = "sk-" + "b" * 30
        mr = MissionReport(goal=f"use {fake}")
        d = mr.to_dict()
        assert fake not in d["goal"]

    def test_with_decisions(self):
        mr = MissionReport()
        mr.decisions.append(DecisionReport(step_index=0))
        mr.decisions.append(DecisionReport(step_index=1))
        d = mr.to_dict()
        assert len(d["decisions"]) == 2


# ---------------------------------------------------------------------------
# IntegrationLayer — Initialization
# ---------------------------------------------------------------------------

class TestIntegrationLayerInit:
    """Test layer initialization."""

    def test_default_init(self):
        layer = IntegrationLayer(project_root="/tmp")
        assert layer.project_root == "/tmp"
        assert layer.max_steps == 50
        assert layer.role == "coder"

    def test_custom_init(self):
        layer = IntegrationLayer(
            project_root="/opt",
            max_steps=10,
            role="tester",
        )
        assert layer.max_steps == 10
        assert layer.role == "tester"


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """Test that all component accessors work."""

    def test_mission_controller(self):
        layer = IntegrationLayer(project_root="/tmp")
        mc = layer._get_mission_controller()
        assert mc is not None

    def test_governor(self):
        layer = IntegrationLayer(project_root="/tmp")
        gov = layer._get_governor()
        assert gov is not None

    def test_rollback_manager(self, tmp_path):
        layer = IntegrationLayer(project_root=str(tmp_path))
        rm = layer._get_rollback_manager()
        assert rm is not None

    def test_goap_planner(self):
        layer = IntegrationLayer(project_root="/tmp")
        planner = layer._get_goap_planner()
        assert planner is not None


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

class TestPipelineStatus:
    """Test pipeline status reporting."""

    def test_all_components_available(self, tmp_path):
        layer = IntegrationLayer(project_root=str(tmp_path))
        status = layer.get_pipeline_status()
        assert "all_components_available" in status
        assert "components" in status
        assert "mission_controller" in status["components"]
        assert "context_manager" in status["components"]
        assert "model_orchestrator" in status["components"]
        assert "agent_reasoning_loop" in status["components"]
        assert "tool_runtime" in status["components"]
        assert "code_navigation" in status["components"]
        assert "decision_memory" in status["components"]
        assert "teacher_governor" in status["components"]
        assert "rollback_manager" in status["components"]
        assert "goap_planner" in status["components"]

    def test_components_report_availability(self, tmp_path):
        layer = IntegrationLayer(project_root=str(tmp_path))
        status = layer.get_pipeline_status()
        for name, info in status["components"].items():
            assert "available" in info


# ---------------------------------------------------------------------------
# Action-to-family mapping
# ---------------------------------------------------------------------------

class TestActionToFamily:
    """Test action type to family mapping."""

    def test_known_actions(self):
        assert IntegrationLayer._action_to_family("search_code") == "code_nav"
        assert IntegrationLayer._action_to_family("write_file") == "code_edit"
        assert IntegrationLayer._action_to_family("run_tests") == "test"
        assert IntegrationLayer._action_to_family("git_status") == "git"
        assert IntegrationLayer._action_to_family("raw_shell_proposal") == "shell"
        assert IntegrationLayer._action_to_family("http_check") == "http"
        assert IntegrationLayer._action_to_family("update_plan") == "planning"
        assert IntegrationLayer._action_to_family("record_memory") == "memory"
        assert IntegrationLayer._action_to_family("ask_user") == "human"
        assert IntegrationLayer._action_to_family("finish") == "terminal"
        assert IntegrationLayer._action_to_family("blocked") == "terminal"

    def test_unknown_action(self):
        assert IntegrationLayer._action_to_family("unknown_action") == "unknown"


# ---------------------------------------------------------------------------
# run_mission — with mocked reasoning loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunMission:
    """Test full mission pipeline with mocked components."""

    def test_mission_creates_and_finishes(self, tmp_path):
        """Mission that immediately finishes."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=5)

        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="Task complete",
            parameters={"summary": "Done"},
            risk_hint="low",
            confidence=0.95,
        )

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_mission(goal="Test mission")

        assert report.mission_id != ""
        assert report.status in ("completed", "blocked", "stopped", "finished")
        assert report.total_steps >= 1
        assert isinstance(report.to_dict(), dict)
        assert report.to_dict()["mission_id"] == report.mission_id

    def test_mission_tracks_decisions(self, tmp_path):
        """Verify decisions are recorded."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=5)

        call_count = [0]
        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="researcher",
                    action_type="search_code",
                    reason="Find routes",
                    parameters={"pattern": "def create_app"},
                    risk_hint="low",
                    confidence=0.9,
                ), []
            return AgentAction(
                mode="coder",
                action_type="finish",
                reason="Done",
                parameters={"summary": "Found routes"},
                risk_hint="low",
                confidence=0.95,
            ), []

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            side_effect=mock_decide,
        ):
            report = layer.run_mission(goal="Find routes")

        assert len(report.decisions) >= 2
        assert report.decisions[0].action_schema["action_type"] == "search_code"
        assert report.decisions[0].memory_recorded is True

    def test_mission_records_governor(self, tmp_path):
        """Verify governor checks are recorded per step."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=5)

        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "done"},
            risk_hint="low",
            confidence=0.95,
        )

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_mission(goal="Governor test")

        for d in report.decisions:
            assert d.governor_decision in ("approve", "reject", "shift", "escalate")

    def test_mission_blocked_path(self, tmp_path):
        """Mission that gets blocked."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=5)

        blocked_action = AgentAction(
            mode="coder",
            action_type="blocked",
            reason="Cannot proceed",
            parameters={"reason": "Missing dependency"},
            risk_hint="low",
            confidence=0.9,
        )

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(blocked_action, []),
        ):
            report = layer.run_mission(goal="Blocked mission")

        assert report.status == "blocked"

    def test_mission_duration_tracked(self, tmp_path):
        """Verify duration is tracked."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=1)

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "done"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_mission(goal="Duration test")

        assert report.total_duration_ms >= 0

    def test_mission_summary_generated(self, tmp_path):
        """Verify final summary is generated."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=1)

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "done"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_mission(goal="Summary test")

        assert report.final_summary != ""
        assert "Mission" in report.final_summary

    def test_mission_with_constraints(self, tmp_path):
        """Verify constraints are passed to mission."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=1)

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "done"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_mission(
                goal="Constrained mission",
                constraints=["no_shell", "no_secrets"],
                success_criteria=["tests pass"],
            )

        assert report.mission_id != ""

    def test_llm_unavailable_path(self, tmp_path):
        """Test when no LLM is available (common in test env)."""
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=3)
        report = layer.run_mission(goal="No LLM test")
        # Without LLM, loop blocks on first step
        assert report.status in ("blocked", "stopped", "completed")
        assert isinstance(report.to_dict(), dict)


# ---------------------------------------------------------------------------
# run_single_step
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunSingleStep:
    """Test single step execution."""

    def test_single_step(self, tmp_path):
        layer = IntegrationLayer(project_root=str(tmp_path), max_steps=1)

        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "done"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(
            AgentReasoningLoop, "_decide_action",
            return_value=(finish_action, []),
        ):
            report = layer.run_single_step(goal="Quick test")

        assert report.total_steps >= 1
