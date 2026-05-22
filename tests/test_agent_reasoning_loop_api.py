"""API tests for Agent Reasoning Loop endpoints — Epic #61."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from igris.core.agent_reasoning_loop import LoopResult
from igris.web.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _fake_result(goal: str = "Test goal") -> LoopResult:
    """Return a minimal LoopResult that satisfies to_dict()."""
    return LoopResult(goal=goal, status="finished")


@pytest.mark.slow
class TestReasoningRunAPI:
    """Test POST /api/reasoning/run — LLM call mocked; tests API contract only."""

    def test_run_basic(self, client):
        with patch(
            "igris.core.agent_reasoning_loop.AgentReasoningLoop.run",
            return_value=_fake_result("Test goal"),
        ):
            resp = client.post("/api/reasoning/run", json={
                "goal": "Test goal",
                "max_steps": 2,
                "role": "coder",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "loop_id" in data
        assert "status" in data
        assert "steps" in data
        assert data["goal"] == "Test goal"

    def test_run_empty(self, client):
        with patch(
            "igris.core.agent_reasoning_loop.AgentReasoningLoop.run",
            return_value=_fake_result(""),
        ):
            resp = client.post("/api/reasoning/run", json={"max_steps": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["total_steps"], int)

    def test_run_invalid_initial_context(self, client):
        """initial_context must be dict or str — list should return 400."""
        resp = client.post("/api/reasoning/run", json={
            "goal": "x",
            "initial_context": [1, 2, 3],
        })
        assert resp.status_code == 400


@pytest.mark.slow
class TestReasoningStepAPI:
    """Test POST /api/reasoning/step — LLM call mocked; tests API contract only."""

    def test_step_basic(self, client):
        with patch(
            "igris.core.agent_reasoning_loop.AgentReasoningLoop.run",
            return_value=_fake_result("Single step test"),
        ):
            resp = client.post("/api/reasoning/step", json={
                "goal": "Single step test",
                "role": "researcher",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] <= 1
        assert isinstance(data["steps"], list)


@pytest.mark.slow
class TestReasoningStopReasonsAPI:
    """Test GET /api/reasoning/stop-reasons — pure data, no LLM call."""

    def test_stop_reasons(self, client):
        resp = client.get("/api/reasoning/stop-reasons")
        assert resp.status_code == 200
        data = resp.json()
        assert "stop_reasons" in data
        assert "finish" in data["stop_reasons"]
        assert "blocked" in data["stop_reasons"]
        assert "max_steps" in data["stop_reasons"]
