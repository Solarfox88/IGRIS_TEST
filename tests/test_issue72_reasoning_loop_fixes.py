"""Tests for Issue #72 — Reasoning loop repeat guard, result consumption,
initial_context validation, and FastAPI route discovery.

Covers:
  1. initial_context string handling (loop + API)
  2. Anti-repeat guard: repeated find_files detection
  3. Consuming find_files results into downstream read_file_range
  4. FastAPI route discovery via CodeNavigator
"""

import os
import json
import tempfile
import textwrap

import pytest
from unittest.mock import patch, MagicMock

from igris.core.agent_reasoning_loop import (
    AgentReasoningLoop,
    LoopStep,
    LoopResult,
)
from igris.core.agent_action_schema import AgentAction
from igris.core.code_navigation import CodeNavigator, NavResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_project(files: dict) -> str:
    """Create a temporary project directory with given files.

    Args:
        files: mapping of relative path -> content
    Returns:
        absolute path to project root
    """
    root = tempfile.mkdtemp(prefix="igris_test_")
    for rel_path, content in files.items():
        full = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    return root


# ---------------------------------------------------------------------------
# 1. initial_context string handling
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestInitialContextHandling:
    """Marked slow: calls AgentReasoningLoop.run() which makes real LLM calls."""

    def test_string_initial_context_normalised(self):
        loop = AgentReasoningLoop(max_steps=1, role="coder")
        loop.run(goal="test", initial_context="just a note")
        assert loop._world_state.get("note") == "just a note"

    def test_dict_initial_context_still_works(self):
        loop = AgentReasoningLoop(max_steps=1, role="coder")
        loop.run(goal="test", initial_context={"key": "val"})
        assert loop._world_state.get("key") == "val"

    def test_none_initial_context(self):
        loop = AgentReasoningLoop(max_steps=1, role="coder")
        loop.run(goal="test", initial_context=None)
        assert "note" not in loop._world_state

    def test_int_initial_context_normalised(self):
        loop = AgentReasoningLoop(max_steps=1, role="coder")
        loop.run(goal="test", initial_context=42)
        assert loop._world_state.get("note") == "42"


@pytest.mark.slow
class TestInitialContextAPI:
    """Marked slow: POSTs to /api/reasoning/run which makes real LLM calls."""

    @pytest.fixture
    def client(self):
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_string_initial_context_api(self, client):
        resp = client.post("/api/reasoning/run", json={
            "goal": "test goal",
            "max_steps": 1,
            "initial_context": "some string context",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "loop_id" in data

    def test_dict_initial_context_api(self, client):
        resp = client.post("/api/reasoning/run", json={
            "goal": "test goal",
            "max_steps": 1,
            "initial_context": {"project": "igris"},
        })
        assert resp.status_code == 200

    def test_list_initial_context_api_returns_400(self, client):
        resp = client.post("/api/reasoning/run", json={
            "goal": "test goal",
            "max_steps": 1,
            "initial_context": [1, 2, 3],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_null_initial_context_api(self, client):
        resp = client.post("/api/reasoning/run", json={
            "goal": "test goal",
            "max_steps": 1,
            "initial_context": None,
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Anti-repeat guard: repeated find_files detection
# ---------------------------------------------------------------------------

class TestAntiRepeatGuard:
    """The loop must block on repeated identical actions without consumption."""

    def test_action_signature_deterministic(self):
        sig1 = AgentReasoningLoop._action_signature("find_files", {"pattern": "*.py"})
        sig2 = AgentReasoningLoop._action_signature("find_files", {"pattern": "*.py"})
        assert sig1 == sig2

    def test_action_signature_different_params(self):
        sig1 = AgentReasoningLoop._action_signature("find_files", {"pattern": "*.py"})
        sig2 = AgentReasoningLoop._action_signature("find_files", {"pattern": "*.js"})
        assert sig1 != sig2

    def test_action_signature_different_types(self):
        sig1 = AgentReasoningLoop._action_signature("find_files", {"pattern": "*.py"})
        sig2 = AgentReasoningLoop._action_signature("search_code", {"pattern": "*.py"})
        assert sig1 != sig2

    def test_no_repeat_on_first_call(self):
        loop = AgentReasoningLoop(max_steps=5)
        diag = loop._check_anti_repeat("find_files", {"pattern": "*.py"})
        assert diag is None

    def test_no_repeat_on_second_call(self):
        """Second identical call is allowed (threshold is 2)."""
        loop = AgentReasoningLoop(max_steps=5)
        loop._record_action_history("find_files", {"pattern": "*.py"}, "success",
                                     result_data=["a.py", "b.py"])
        diag = loop._check_anti_repeat("find_files", {"pattern": "*.py"})
        assert diag is None

    def test_repeat_blocked_on_third_call(self):
        """Third identical call without consumption triggers guard."""
        loop = AgentReasoningLoop(max_steps=5)
        loop._record_action_history("find_files", {"pattern": "*.py"}, "success",
                                     result_data=["a.py", "b.py"])
        loop._record_action_history("find_files", {"pattern": "*.py"}, "success",
                                     result_data=["a.py", "b.py"])
        diag = loop._check_anti_repeat("find_files", {"pattern": "*.py"})
        assert diag is not None
        assert "Anti-repeat guard" in diag

    def test_repeat_allowed_if_result_consumed(self):
        """If results were consumed by a downstream action, repeat is ok."""
        loop = AgentReasoningLoop(max_steps=5)
        loop._record_action_history("find_files", {"pattern": "*.py"}, "success",
                                     result_data=["igris/web/server.py"])
        loop._record_action_history("find_files", {"pattern": "*.py"}, "success",
                                     result_data=["igris/web/server.py"])
        # Simulate downstream consumption: read_file_range used one of the paths
        loop._record_action_history("read_file_range",
                                     {"path": "igris/web/server.py", "start": 1},
                                     "success")
        diag = loop._check_anti_repeat("find_files", {"pattern": "*.py"})
        assert diag is None

    def test_failed_actions_not_counted_as_repeats(self):
        loop = AgentReasoningLoop(max_steps=5)
        loop._record_action_history("find_files", {"pattern": "*.py"}, "failure")
        loop._record_action_history("find_files", {"pattern": "*.py"}, "failure")
        loop._record_action_history("find_files", {"pattern": "*.py"}, "failure")
        diag = loop._check_anti_repeat("find_files", {"pattern": "*.py"})
        assert diag is None

    def test_repeated_read_file_range_does_not_self_consume(self):
        """Identical reads should not consume their own result path."""
        loop = AgentReasoningLoop(max_steps=5)
        params = {"path": "igris/web/server.py", "start": 50, "end": 100}
        result_data = {"path": "igris/web/server.py", "content": "def create_app():\n"}
        loop._record_action_history("read_file_range", params, "success", result_data=result_data)
        loop._record_action_history("read_file_range", params, "success", result_data=result_data)

        diag = loop._check_anti_repeat("read_file_range", params)

        assert diag is not None
        assert "Anti-repeat guard" in diag


# ---------------------------------------------------------------------------
# 3. Consuming find_files results into read_file_range
# ---------------------------------------------------------------------------

class TestResultConsumption:
    """The loop must store tool results and make them available for
    downstream consumption via world_state."""

    def test_store_tool_result_find_files(self):
        loop = AgentReasoningLoop(max_steps=5)
        loop._store_tool_result("find_files", ["a.py", "b.py", "c.py"])
        assert loop._world_state["last_tool_result"]["action_type"] == "find_files"
        assert loop._world_state["discovered_files"] == ["a.py", "b.py", "c.py"]

    def test_store_tool_result_search_code(self):
        loop = AgentReasoningLoop(max_steps=5)
        loop._store_tool_result("search_code", [
            {"file": "x.py", "line_number": 10, "line_content": "def foo():"},
        ])
        assert loop._world_state["search_matched_files"] == ["x.py"]

    def test_tool_result_history_rolling(self):
        loop = AgentReasoningLoop(max_steps=5)
        for i in range(7):
            loop._store_tool_result("find_files", [f"file_{i}.py"])
        history = loop._world_state["tool_result_history"]
        assert len(history) == 5  # only last 5 kept

    def test_result_data_in_loop_step(self):
        """LoopStep.result_data should carry structured data when available."""
        s = LoopStep(step_number=1, action_type="find_files", outcome="success")
        s.result_data = ["a.py", "b.py"]
        d = s.to_dict()
        assert d["result_data"] == ["a.py", "b.py"]

    def test_result_data_absent_when_none(self):
        s = LoopStep(step_number=1, action_type="finish", outcome="finish")
        d = s.to_dict()
        assert "result_data" not in d

    def test_execute_navigation_returns_result_data(self):
        """_execute_navigation should return result_data for find_files."""
        root = _make_temp_project({
            "hello.py": "print('hello')",
            "world.py": "print('world')",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="find_files",
            parameters={"pattern": "*.py"},
        )
        result = loop._execute_navigation(action)
        assert result["success"]
        assert result.get("result_data") is not None
        assert isinstance(result["result_data"], list)
        assert len(result["result_data"]) == 2

    def test_execute_navigation_search_code_returns_data(self):
        root = _make_temp_project({
            "app.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        })
        loop = AgentReasoningLoop(project_root=root, max_steps=1)
        action = AgentAction(
            action_type="search_code",
            parameters={"pattern": "FastAPI"},
        )
        result = loop._execute_navigation(action)
        assert result["success"]
        assert result.get("result_data") is not None
        assert len(result["result_data"]) >= 1


# ---------------------------------------------------------------------------
# 4. FastAPI route discovery
# ---------------------------------------------------------------------------

class TestFastAPIRouteDiscovery:
    """CodeNavigator must discover FastAPI routes and prefer server.py."""

    def test_discover_fastapi_routes_basic(self):
        root = _make_temp_project({
            "app/main.py": textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/api/ping")
                async def ping():
                    return {"pong": True}

                @app.post("/api/data")
                async def data():
                    return {}
            """),
            "utils.py": "# no routes here\ndef helper():\n    pass\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_fastapi_routes()
        assert result.success
        assert result.total_count >= 1
        # The app/main.py should be found with routes
        files_found = [d["file"] for d in result.data]
        assert any("main.py" in f for f in files_found)
        # Check route details
        main_entry = [d for d in result.data if "main.py" in d["file"]][0]
        assert len(main_entry["routes"]) >= 2
        assert main_entry["score"] > 0

    def test_preferred_paths_scored_higher(self):
        root = _make_temp_project({
            "igris/web/server.py": textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()
                @app.get("/api/status")
                async def status(): return {}
            """),
            "other/routes.py": textwrap.dedent("""\
                from fastapi import APIRouter
                router = APIRouter()
                @router.get("/other")
                async def other(): return {}
            """),
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_fastapi_routes()
        assert result.success
        assert len(result.data) >= 2
        # server.py should be ranked first (higher score)
        assert "server.py" in result.data[0]["file"]

    def test_no_fastapi_files(self):
        root = _make_temp_project({
            "plain.py": "print('no routes')\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_fastapi_routes()
        assert result.success
        assert result.total_count == 0
        assert result.data == []

    def test_utils_not_detected(self):
        root = _make_temp_project({
            "utils.py": "def helper():\n    return 42\n",
        })
        nav = CodeNavigator(project_root=root)
        result = nav.discover_fastapi_routes()
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# 5. Integration: anti-repeat in full loop execution
# ---------------------------------------------------------------------------

class TestAntiRepeatInLoop:
    """The anti-repeat guard exposes diagnosis when actions repeat."""

    def test_loop_step_skipped_on_repeated_read_only_action(self):
        """Repeated read-only navigation should not terminally block the loop."""
        loop = AgentReasoningLoop(max_steps=5, project_root="/tmp")

        # Simulate two successful find_files in history
        loop._record_action_history(
            "find_files", {"pattern": "*.py"}, "success",
            result_data=["a.py", "b.py"],
        )
        loop._record_action_history(
            "find_files", {"pattern": "*.py"}, "success",
            result_data=["a.py", "b.py"],
        )

        # Build a mock action that returns find_files with same params
        mock_action = AgentAction(
            action_type="find_files",
            parameters={"pattern": "*.py"},
            reason="Finding Python files again",
        )

        # Patch _decide_action to return this action
        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                step = loop._execute_step(3, "test goal", "")

        assert step.outcome == "skipped"
        assert "Anti-repeat guard" in step.error
        assert loop._world_state.get("anti_repeat_triggered") is True
        assert loop._world_state.get("anti_repeat_retryable") is True

    def test_loop_step_blocked_on_repeated_write_action(self):
        """Repeated successful writes still block to avoid unsafe edit loops."""
        loop = AgentReasoningLoop(max_steps=5, project_root="/tmp")
        params = {"path": "server.py", "anchor": "x", "content": "y"}
        loop._record_action_history("insert_after", params, "success")
        loop._record_action_history("insert_after", params, "success")

        mock_action = AgentAction(
            action_type="insert_after",
            parameters=params,
            reason="Repeating edit",
        )

        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                step = loop._execute_step(3, "test goal", "")

        assert step.outcome == "blocked"
        assert "Anti-repeat guard" in step.error
        assert loop._world_state.get("anti_repeat_triggered") is True
        assert loop._world_state.get("anti_repeat_retryable") is not True

    def test_anti_repeat_world_state_diagnosis(self):
        """World state should contain anti_repeat_diagnosis."""
        loop = AgentReasoningLoop(max_steps=5)
        loop._record_action_history(
            "find_files", {"pattern": "*.py"}, "success",
            result_data=["a.py"],
        )
        loop._record_action_history(
            "find_files", {"pattern": "*.py"}, "success",
            result_data=["a.py"],
        )

        mock_action = AgentAction(
            action_type="find_files",
            parameters={"pattern": "*.py"},
        )
        with patch.object(loop, "_decide_action", return_value=(mock_action, [])):
            with patch.object(loop, "_build_context", return_value=MagicMock()):
                step = loop._execute_step(3, "test", "")

        assert "anti_repeat_diagnosis" in loop._world_state
        assert "find_files" in loop._world_state["anti_repeat_diagnosis"]


# ---------------------------------------------------------------------------
# 6. Was-result-consumed check
# ---------------------------------------------------------------------------

class TestWasResultConsumed:
    """_was_result_consumed should detect downstream usage of results."""

    def test_consumed_when_path_used(self):
        loop = AgentReasoningLoop(max_steps=5)
        loop._action_history = [
            {
                "signature": "find_files::{...}",
                "action_type": "find_files",
                "parameters": {"pattern": "*.py"},
                "outcome": "success",
                "result_data": ["igris/web/server.py"],
            },
            {
                "signature": "read_file_range::{...}",
                "action_type": "read_file_range",
                "parameters": {"path": "igris/web/server.py", "start": 1},
                "outcome": "success",
                "result_data": None,
            },
        ]
        assert loop._was_result_consumed(["igris/web/server.py"]) is True

    def test_not_consumed_when_no_downstream(self):
        loop = AgentReasoningLoop(max_steps=5)
        loop._action_history = [
            {
                "signature": "find_files::{...}",
                "action_type": "find_files",
                "parameters": {"pattern": "*.py"},
                "outcome": "success",
                "result_data": ["igris/web/server.py"],
            },
        ]
        assert loop._was_result_consumed(["igris/web/server.py"]) is False

    def test_not_consumed_with_empty_data(self):
        loop = AgentReasoningLoop(max_steps=5)
        assert loop._was_result_consumed(None) is False
        assert loop._was_result_consumed([]) is False
