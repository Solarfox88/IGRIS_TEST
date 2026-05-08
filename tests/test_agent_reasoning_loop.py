"""Tests for Agent Reasoning Loop — Epic #61.

Validates the cognitive core: step execution, action routing,
stop conditions, governor checks, and degraded mode.
"""

import pytest
from unittest.mock import patch, MagicMock

from igris.core.agent_reasoning_loop import (
    AgentReasoningLoop,
    LoopStep,
    LoopResult,
    STOP_REASONS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_CONSECUTIVE_ERRORS,
)
from igris.core.agent_action_schema import AgentAction, ValidationResult


# ---------------------------------------------------------------------------
# LoopStep
# ---------------------------------------------------------------------------

class TestLoopStep:
    """Test LoopStep dataclass."""

    def test_default_step(self):
        s = LoopStep()
        assert s.step_number == 0
        assert s.outcome == ""

    def test_to_dict(self):
        s = LoopStep(step_number=1, action_type="search_code", outcome="success")
        d = s.to_dict()
        assert d["step_number"] == 1
        assert d["action_type"] == "search_code"
        assert d["outcome"] == "success"

    def test_to_dict_redacts_secrets(self):
        fake = "sk-" + "a" * 30
        s = LoopStep(reason=f"key is {fake}")
        d = s.to_dict()
        assert fake not in d["reason"]
        assert "REDACTED" in d["reason"]


# ---------------------------------------------------------------------------
# LoopResult
# ---------------------------------------------------------------------------

class TestLoopResult:
    """Test LoopResult dataclass."""

    def test_default_result(self):
        r = LoopResult()
        assert r.status == "pending"
        assert r.total_steps == 0

    def test_to_dict(self):
        r = LoopResult(goal="test", status="finished", total_steps=3)
        d = r.to_dict()
        assert d["goal"] == "test"
        assert d["status"] == "finished"
        assert d["total_steps"] == 3

    def test_to_dict_redacts_secrets(self):
        fake = "sk-" + "b" * 30
        r = LoopResult(goal=f"use {fake}")
        d = r.to_dict()
        assert fake not in d["goal"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module constants."""

    def test_stop_reasons(self):
        assert "finish" in STOP_REASONS
        assert "blocked" in STOP_REASONS
        assert "max_steps" in STOP_REASONS
        assert "llm_unavailable" in STOP_REASONS

    def test_default_max_steps(self):
        assert DEFAULT_MAX_STEPS == 50

    def test_default_max_errors(self):
        assert DEFAULT_MAX_CONSECUTIVE_ERRORS == 5


# ---------------------------------------------------------------------------
# AgentReasoningLoop — Initialization
# ---------------------------------------------------------------------------

class TestLoopInit:
    """Test loop initialization."""

    def test_default_init(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        assert loop.project_root == "/tmp"
        assert loop.max_steps == 50
        assert loop.role == "coder"

    def test_custom_init(self):
        loop = AgentReasoningLoop(
            project_root="/opt",
            max_steps=10,
            max_consecutive_errors=3,
            role="tester",
        )
        assert loop.max_steps == 10
        assert loop.max_consecutive_errors == 3
        assert loop.role == "tester"


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

class TestStopConditions:
    """Test loop stop condition checks."""

    def test_max_steps(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=5)
        assert loop._check_stop_conditions(6) == "max_steps"
        assert loop._check_stop_conditions(5) is None

    def test_consecutive_errors(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_consecutive_errors=3)
        loop._consecutive_errors = 3
        assert loop._check_stop_conditions(1) == "budget_exceeded"

    def test_explicit_stop_reason(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        loop._stop_reason = "governor_stop"
        assert loop._check_stop_conditions(1) == "governor_stop"

    def test_no_stop(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        assert loop._check_stop_conditions(1) is None


# ---------------------------------------------------------------------------
# Action routing
# ---------------------------------------------------------------------------

class TestActionRouting:
    """Test action routing to correct handlers."""

    def test_navigation_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="researcher",
            action_type="search_code",
            reason="test",
            parameters={"pattern": "def hello"},
        )
        result = loop._execute_action(action, "code_navigation")
        assert isinstance(result, dict)
        assert "success" in result

    def test_command_risk_engine_blocked(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="coder",
            action_type="raw_shell_proposal",
            reason="test",
            parameters={"command": "ls"},
        )
        result = loop._execute_action(action, "command_risk_engine")
        assert result["success"] is False
        assert "Command Risk Engine not yet available" in result["error"]

    def test_plan_update_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="planner",
            action_type="update_plan",
            reason="test",
            parameters={"updates": "step 1: read code"},
        )
        result = loop._execute_action(action, "mission_controller")
        assert result["success"] is True
        assert "Plan updated" in result["summary"]
        assert loop._world_state.get("last_plan_update") == "step 1: read code"

    def test_memory_record_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="memory_manager",
            action_type="record_memory",
            reason="learned something",
            parameters={"event_type": "lesson", "content": "always run tests first"},
        )
        result = loop._execute_action(action, "memory")
        assert result["success"] is True
        assert len(loop._memory_items) == 1
        assert loop._memory_items[0]["content"] == "always run tests first"

    def test_human_gate_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="coordinator",
            action_type="ask_user",
            reason="need clarification",
            parameters={"question": "Which endpoint format?"},
        )
        result = loop._execute_action(action, "human_gate")
        assert result["success"] is True
        assert "Which endpoint format?" in result["summary"]

    def test_run_tests_target_becomes_pytest_arg(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        rt = MagicMock()
        rt.run_tests.return_value = MagicMock(
            success=True,
            output="1 passed",
            error="",
        )
        action = AgentAction(
            mode="coder",
            action_type="run_tests",
            reason="run targeted test",
            parameters={"target": "tests/test_version_info.py"},
        )

        with patch.object(loop, "_get_tool_runtime", return_value=rt):
            result = loop._execute_action(action, "tool_runtime")

        assert result["success"] is True
        rt.run_tests.assert_called_once_with(args=["tests/test_version_info.py"])

    def test_run_tests_args_take_precedence_over_target(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        rt = MagicMock()
        rt.run_tests.return_value = MagicMock(
            success=True,
            output="1 passed",
            error="",
        )
        action = AgentAction(
            mode="coder",
            action_type="run_tests",
            reason="run explicit args",
            parameters={"target": "tests/test_version_info.py", "args": "tests/test_other.py -q"},
        )

        with patch.object(loop, "_get_tool_runtime", return_value=rt):
            result = loop._execute_action(action, "tool_runtime")

        assert result["success"] is True
        rt.run_tests.assert_called_once_with(args=["tests/test_other.py", "-q"])

    def test_terminal_finish_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="reporter",
            action_type="finish",
            reason="done",
            parameters={"summary": "All tasks complete"},
        )
        result = loop._execute_action(action, "terminal")
        assert result["success"] is True
        assert "All tasks complete" in result["summary"]

    def test_terminal_blocked_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="coder",
            action_type="blocked",
            reason="stuck",
            parameters={"reason": "Cannot find file"},
        )
        result = loop._execute_action(action, "terminal")
        assert result["success"] is True

    def test_unknown_route(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        action = AgentAction(
            mode="coder",
            action_type="search_code",
            reason="test",
            parameters={"pattern": "x"},
        )
        result = loop._execute_action(action, "nonexistent_route")
        assert result["success"] is False
        assert "Unknown route" in result["error"]


# ---------------------------------------------------------------------------
# Run — LLM unavailable (deterministic fallback)
# ---------------------------------------------------------------------------

class TestRunDeterministic:
    """Test loop run when no LLM is available (most common in test env)."""

    def test_run_blocks_without_llm(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=3)
        result = loop.run(goal="Test goal", mission_id="m-test")
        # Without LLM, every step returns "blocked" on first attempt
        assert result.status in ("blocked", "stopped")
        assert result.total_steps >= 1
        assert isinstance(result.to_dict(), dict)

    def test_run_tracks_goal(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        result = loop.run(goal="Add /api/ping")
        assert result.goal == "Add /api/ping"

    def test_run_with_initial_context(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        result = loop.run(
            goal="check status",
            initial_context={"repo_clean": True},
        )
        assert loop._world_state.get("repo_clean") is True

    def test_run_produces_summary(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        result = loop.run(goal="test")
        assert result.final_summary != ""
        assert "Loop" in result.final_summary


# ---------------------------------------------------------------------------
# Run — Simulated LLM responses
# ---------------------------------------------------------------------------

class TestRunWithMockedLLM:
    """Test loop with mocked LLM producing specific actions."""

    def _make_loop(self, max_steps=5):
        return AgentReasoningLoop(project_root="/tmp", max_steps=max_steps)

    def test_finish_action_stops_loop(self):
        loop = self._make_loop()
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="Task complete",
            parameters={"summary": "Added endpoint and tests"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(loop, "_decide_action", return_value=(finish_action, [])):
            result = loop.run(goal="Add endpoint")
        assert result.status == "finished"
        assert result.stop_reason == "finish"
        assert result.total_steps == 1

    def test_blocked_action_stops_loop(self):
        loop = self._make_loop()
        blocked_action = AgentAction(
            mode="coder",
            action_type="blocked",
            reason="Cannot find server.py",
            parameters={"reason": "File not found"},
            risk_hint="low",
            confidence=0.9,
        )
        with patch.object(loop, "_decide_action", return_value=(blocked_action, [])):
            result = loop.run(goal="Fix bug")
        assert result.status == "blocked"
        assert result.stop_reason == "blocked"

    def test_ask_user_stops_loop(self):
        loop = self._make_loop()
        ask_action = AgentAction(
            mode="coordinator",
            action_type="ask_user",
            reason="Need clarification",
            parameters={"question": "Which file should I modify?"},
            risk_hint="low",
            confidence=0.8,
        )
        with patch.object(loop, "_decide_action", return_value=(ask_action, [])):
            result = loop.run(goal="Unclear task")
        assert result.status == "blocked"
        assert result.stop_reason == "ask_user"

    def test_ask_user_can_be_suppressed_for_controlled_runs(self):
        loop = self._make_loop()
        ask_action = AgentAction(
            mode="coordinator",
            action_type="ask_user",
            reason="Need clarification",
            parameters={"question": "Which file should I modify?"},
            risk_hint="low",
            confidence=0.8,
        )
        with patch.object(loop, "_decide_action", return_value=(ask_action, [])):
            result = loop.run(
                goal="Unclear task",
                initial_context={"must_not_ask_user": True},
            )
        assert result.status == "stopped"
        assert result.stop_reason == "max_steps"
        assert result.steps[0].outcome == "skipped"

    def test_navigation_then_finish(self):
        loop = self._make_loop()
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="researcher",
                    action_type="search_code",
                    reason="Find server routes",
                    parameters={"pattern": "def create_app"},
                    risk_hint="low",
                    confidence=0.9,
                ), []
            else:
                return AgentAction(
                    mode="coder",
                    action_type="finish",
                    reason="Found what I needed",
                    parameters={"summary": "Found routes in server.py"},
                    risk_hint="low",
                    confidence=0.95,
                ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Understand routes")
        assert result.status == "finished"
        assert result.total_steps == 2
        assert result.successful_steps >= 1

    def test_max_steps_enforced(self):
        loop = self._make_loop(max_steps=3)
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            # Use different parameters each step to avoid anti-repeat guard
            return AgentAction(
                mode="researcher",
                action_type="search_code",
                reason="keep searching",
                parameters={"pattern": f"test_{call_count[0]}"},
                risk_hint="low",
                confidence=0.5,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Infinite search")
        assert result.status == "stopped"
        assert result.stop_reason == "max_steps"
        assert result.total_steps == 3

    def test_consecutive_errors_stop(self):
        loop = AgentReasoningLoop(
            project_root="/tmp",
            max_steps=20,
            max_consecutive_errors=3,
        )

        def mock_decide(ctx):
            return None, ["Parse error: invalid JSON"]

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Broken LLM")
        assert result.stop_reason == "budget_exceeded"
        assert result.failed_steps >= 3

    def test_memory_record_tracked(self):
        loop = self._make_loop()
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="memory_manager",
                    action_type="record_memory",
                    reason="Record finding",
                    parameters={"event_type": "lesson", "content": "routes in server.py"},
                    risk_hint="low",
                    confidence=0.9,
                ), []
            return AgentAction(
                mode="coder",
                action_type="finish",
                reason="done",
                parameters={"summary": "Recorded and done"},
                risk_hint="low",
                confidence=0.95,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Record and finish")
        assert result.status == "finished"
        assert len(loop._memory_items) == 1
        assert loop._memory_items[0]["content"] == "routes in server.py"

    def test_plan_update_tracked(self):
        loop = self._make_loop()
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="planner",
                    action_type="update_plan",
                    reason="Update plan",
                    parameters={"updates": "Step 1: find server, Step 2: add route"},
                    risk_hint="low",
                    confidence=0.85,
                ), []
            return AgentAction(
                mode="coder",
                action_type="finish",
                reason="done",
                parameters={"summary": "Plan updated"},
                risk_hint="low",
                confidence=0.95,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Plan and finish")
        assert "last_plan_update" in loop._world_state

    def test_raw_shell_blocked(self):
        loop = self._make_loop()
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="devops",
                    action_type="raw_shell_proposal",
                    reason="Install dependency",
                    parameters={"command": "pip install requests"},
                    risk_hint="medium",
                    confidence=0.7,
                ), []
            return AgentAction(
                mode="coder",
                action_type="blocked",
                reason="Shell not available",
                parameters={"reason": "Cannot install without risk engine"},
                risk_hint="low",
                confidence=0.9,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Install deps")
        assert result.total_steps == 2
        # First step should fail (command risk engine not available)
        assert result.steps[0].outcome == "failure"
        assert "Command Risk Engine" in result.steps[0].error

    def test_file_modification_tracked(self):
        loop = self._make_loop()
        call_count = [0]

        def mock_decide(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(
                    mode="coder",
                    action_type="write_file",
                    reason="Create endpoint",
                    parameters={"path": "server.py", "content": "def ping(): pass"},
                    risk_hint="low",
                    confidence=0.9,
                ), []
            return AgentAction(
                mode="coder",
                action_type="finish",
                reason="done",
                parameters={"summary": "Created endpoint"},
                risk_hint="low",
                confidence=0.95,
            ), []

        with patch.object(loop, "_decide_action", side_effect=mock_decide):
            result = loop.run(goal="Add endpoint")
        assert "server.py" in result.files_modified


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Test public API methods."""

    def test_get_steps_empty(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        assert loop.get_steps() == []

    def test_get_state(self):
        loop = AgentReasoningLoop(project_root="/tmp")
        state = loop.get_state()
        assert state["step_count"] == 0
        assert state["consecutive_errors"] == 0

    def test_get_state_after_run(self):
        loop = AgentReasoningLoop(project_root="/tmp", max_steps=1)
        finish_action = AgentAction(
            mode="coder",
            action_type="finish",
            reason="done",
            parameters={"summary": "Done"},
            risk_hint="low",
            confidence=0.95,
        )
        with patch.object(loop, "_decide_action", return_value=(finish_action, [])):
            loop.run(goal="test")
        state = loop.get_state()
        assert state["step_count"] == 1
