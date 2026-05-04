"""API tests for safety policy and explainable selection (Sprint 13)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


@pytest.fixture
def client(tmp_path):
    from igris.models.config import CONFIG
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    for d in [".igris/tasks", ".igris/timeline", ".igris/memory"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    from igris.core.safe_policy import reset_rate_limits
    reset_rate_limits()
    return TestClient(create_app())


class TestSafetyPolicyAPI:
    def test_policy_status(self, client):
        r = client.get("/api/safety/policy")
        assert r.status_code == 200
        d = r.json()
        assert "allowed_commands" in d
        assert "blocked_commands" in d
        assert "remaining_capacity" in d

    def test_policy_check_allowed(self, client):
        r = client.post("/api/safety/policy/check", json={"command_id": "git_status"})
        assert r.status_code == 200
        assert r.json()["allowed"] is True

    def test_policy_check_blocked(self, client):
        r = client.post("/api/safety/policy/check", json={"command_id": "rm_rf"})
        assert r.status_code == 200
        assert r.json()["allowed"] is False

    def test_policy_check_missing_id(self, client):
        r = client.post("/api/safety/policy/check", json={})
        assert r.status_code == 400

    def test_no_secrets_in_policy(self, client):
        r = client.get("/api/safety/policy")
        text = json.dumps(r.json())
        assert "sk-" not in text


class TestExplainableSelectionAPI:
    def test_explain_empty(self, client):
        r = client.get("/api/tasks/selection/explain")
        assert r.status_code == 200
        d = r.json()
        assert "candidates" in d
        assert "summary" in d

    def test_explain_with_tasks(self, client):
        client.post("/api/tasks", json={
            "title": "test task",
            "description": "run tests"
        })
        client.post("/api/tasks", json={
            "title": "deploy task",
            "description": "deploy app"
        })
        r = client.get("/api/tasks/selection/explain")
        assert r.status_code == 200
        d = r.json()
        assert len(d["candidates"]) >= 2
        assert d["selected"] is not None

    def test_explain_scores_present(self, client):
        client.post("/api/tasks", json={
            "title": "task",
            "description": "do something"
        })
        r = client.get("/api/tasks/selection/explain")
        d = r.json()
        for c in d["candidates"]:
            assert "score" in c
            assert "why" in c

    def test_no_secrets_in_explain(self, client):
        r = client.get("/api/tasks/selection/explain")
        text = json.dumps(r.json())
        assert "sk-" not in text
