"""API tests for diagnostics endpoints (Sprint 12)."""

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
    return TestClient(create_app())


class TestDiagnosticsAPI:
    def test_diagnostics_endpoint(self, client):
        r = client.get("/api/diagnostics")
        assert r.status_code == 200
        d = r.json()
        assert "findings" in d
        assert "summary" in d
        assert "finding_count" in d

    def test_diagnostics_summary_endpoint(self, client):
        r = client.get("/api/diagnostics/summary")
        assert r.status_code == 200
        d = r.json()
        assert "healthy" in d
        assert "task_stats" in d
        assert "finding_count" in d

    def test_diagnostics_with_tasks(self, client):
        # Create a task first
        client.post("/api/tasks", json={
            "title": "test task",
            "description": "diagnostic test"
        })
        r = client.get("/api/diagnostics")
        assert r.status_code == 200
        d = r.json()
        assert d["summary"]["total_tasks"] >= 1

    def test_diagnostics_with_blocked(self, client):
        r = client.post("/api/tasks", json={
            "title": "block me",
            "description": "will block"
        })
        tid = r.json()["id"]
        client.post(f"/api/tasks/{tid}/block", json={"reason": "test block"})
        r = client.get("/api/diagnostics/summary")
        assert r.status_code == 200
        assert r.json()["task_stats"]["blocked"] >= 1

    def test_no_secrets_in_diagnostics(self, client):
        r = client.get("/api/diagnostics")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text

    def test_summary_healthy_empty(self, client):
        r = client.get("/api/diagnostics/summary")
        assert r.json()["healthy"] is True
