"""API tests for Agent Reasoning Loop endpoints — Epic #61."""

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestReasoningRunAPI:
    """Test POST /api/reasoning/run."""

    def test_run_basic(self, client):
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
        resp = client.post("/api/reasoning/run", json={"max_steps": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["total_steps"], int)


class TestReasoningStepAPI:
    """Test POST /api/reasoning/step."""

    def test_step_basic(self, client):
        resp = client.post("/api/reasoning/step", json={
            "goal": "Single step test",
            "role": "researcher",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] <= 1
        assert isinstance(data["steps"], list)


class TestReasoningStopReasonsAPI:
    """Test GET /api/reasoning/stop-reasons."""

    def test_stop_reasons(self, client):
        resp = client.get("/api/reasoning/stop-reasons")
        assert resp.status_code == 200
        data = resp.json()
        assert "stop_reasons" in data
        assert "finish" in data["stop_reasons"]
        assert "blocked" in data["stop_reasons"]
        assert "max_steps" in data["stop_reasons"]
