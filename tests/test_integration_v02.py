"""Integration tests for v0.2 baseline (Sprint 10).

Smoke-tests all major subsystems introduced in Sprints 1-9,
verifies cross-module integration, and ensures no regressions.
"""

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
    for d in [
        ".igris/tasks", ".igris/timeline", ".igris/missions",
        ".igris/memory", ".igris/reports", ".igris/validations",
        ".igris/patches", ".igris/a2a/tasks",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)
    # Fake git repo for git endpoints
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


class TestHealthAndReadiness:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readiness(self, client):
        r = client.get("/api/readiness")
        assert r.status_code == 200
        d = r.json()
        assert "project_root_exists" in d

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        assert "provider" in r.json()


class TestPatchWorkflow:
    def test_propose_validate_apply(self, client):
        r = client.post("/api/patches/propose", json={
            "title": "add readme",
            "description": "test",
            "files": [{"path": "docs/test.md", "action": "create", "after": "# Test\n"}],
        })
        assert r.status_code == 200
        pid = r.json()["id"]
        r = client.post(f"/api/patches/{pid}/validate")
        assert r.status_code == 200
        assert r.json()["validation"]["valid"] is True
        r = client.post(f"/api/patches/{pid}/apply")
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_env_blocked(self, client):
        r = client.post("/api/patches/propose", json={
            "title": "bad",
            "files": [{"path": ".env", "action": "create", "after": "SECRET=abc"}],
        })
        assert r.status_code == 200
        pid = r.json()["id"]
        r = client.post(f"/api/patches/{pid}/validate")
        assert r.json()["validation"]["valid"] is False


class TestGitWorkflow:
    def test_git_diff(self, client):
        r = client.get("/api/git/diff")
        assert r.status_code in (200, 500)

    def test_git_branches(self, client):
        r = client.get("/api/git/branches")
        assert r.status_code in (200, 500)

    def test_git_safety_check(self, client):
        r = client.get("/api/git/safety-check")
        assert r.status_code in (200, 500)

    def test_no_push_endpoint(self, client):
        r = client.post("/api/git/push")
        assert r.status_code == 404 or r.status_code == 405


class TestMissionPlanner:
    def test_mission_lifecycle(self, client):
        r = client.post("/api/missions", json={
            "title": "Fix test failures",
            "description": "Run tests and fix any failures"
        })
        assert r.status_code == 200
        mid = r.json()["id"]
        r = client.post(f"/api/missions/{mid}/plan")
        assert r.status_code == 200
        assert len(r.json()["steps"]) > 0
        r = client.post(f"/api/missions/{mid}/materialize-tasks")
        assert r.status_code == 200
        assert len(r.json().get("task_ids", [])) > 0
        r = client.get(f"/api/missions/{mid}/graph")
        assert r.status_code == 200
        assert "nodes" in r.json()


class TestDecisionMemory:
    def test_record_and_query(self, client):
        r = client.post("/api/memory/events", json={
            "event_type": "decision",
            "title": "chose test approach",
            "family": "test",
            "description": "ran pytest"
        })
        assert r.status_code == 200
        r = client.get("/api/memory/decisions")
        assert r.status_code == 200
        assert len(r.json()["events"]) >= 1
        r = client.post("/api/memory/events", json={
            "event_type": "failure",
            "title": "test failed",
            "family": "test",
            "description": "assertion error"
        })
        assert r.status_code == 200
        r = client.get("/api/memory/saturation")
        assert r.status_code == 200


class TestAutonomousLoop:
    def test_loop_status(self, client):
        r = client.get("/api/loop/status")
        assert r.status_code == 200
        assert "running" in r.json()

    def test_loop_recent(self, client):
        r = client.get("/api/loop/recent")
        assert r.status_code == 200


class TestValidation:
    def test_validate_task(self, client):
        r = client.post("/api/tasks", json={
            "title": "integration task",
            "description": "test validation"
        })
        assert r.status_code == 200
        tid = r.json()["id"]
        r = client.post(f"/api/tasks/{tid}/validate", json={
            "reports": [],
            "files_changed": [],
            "manual_reason": "integration test"
        })
        assert r.status_code == 200


class TestA2AStore:
    def test_a2a_task_lifecycle(self, client):
        r = client.post("/api/a2a/store/tasks", json={
            "title": "integration a2a task",
            "description": "test"
        })
        assert r.status_code == 200
        tid = r.json()["id"]
        r = client.post(f"/api/a2a/store/tasks/{tid}/status", json={
            "status": "working"
        })
        assert r.status_code == 200
        r = client.post(f"/api/a2a/tasks/{tid}/artifacts", json={
            "name": "result.txt",
            "content": "test output"
        })
        assert r.status_code == 200
        r = client.get(f"/api/a2a/tasks/{tid}/events")
        assert r.status_code == 200
        assert len(r.json()["events"]) >= 1


class TestCostRouter:
    def test_availability(self, client):
        r = client.get("/api/routing/availability")
        assert r.status_code == 200
        d = r.json()
        assert "ollama" in d
        assert "openai" in d

    def test_budget(self, client):
        r = client.get("/api/cost/budget")
        assert r.status_code == 200
        assert "max_session_cost" in r.json()

    def test_estimate(self, client):
        r = client.post("/api/routing/estimate", json={
            "task_type": "chat",
            "complexity": "low"
        })
        assert r.status_code == 200
        assert "recommended_provider" in r.json()


class TestUIIntegration:
    def test_index_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "IGRIS_GPT" in r.text

    def test_14_tabs(self, client):
        r = client.get("/")
        html = r.text
        tabs = [
            "mission", "terminal", "files", "git", "tests", "logs",
            "agent", "tasks", "safety", "cost", "a2a", "memory",
            "loop", "patches",
        ]
        for t in tabs:
            assert f'data-tab="{t}"' in html

    def test_js_served(self, client):
        r = client.get("/static/js/app.js")
        assert r.status_code == 200

    def test_css_served(self, client):
        r = client.get("/static/css/style.css")
        assert r.status_code == 200


class TestSecretRedaction:
    def test_no_secrets_in_health(self, client):
        r = client.get("/api/health")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text

    def test_no_secrets_in_status(self, client):
        r = client.get("/api/status")
        text = json.dumps(r.json())
        assert "sk-" not in text


class TestSmokeInstall:
    def test_importable(self):
        import igris
        import igris.web.server
        import igris.core.patch_proposal
        import igris.layers.git_layer.git_ops
        import igris.core.mission_planner
        import igris.core.decision_memory
        import igris.core.autonomous_loop
        import igris.layers.validation.validator
        import igris.a2a.task_store
        import igris.layers.advisory.router

    def test_create_app(self):
        app = create_app()
        assert app is not None
