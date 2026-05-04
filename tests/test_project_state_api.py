"""API tests for project state endpoints (Sprint 14)."""

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


class TestProjectStateAPI:
    def test_get_state(self, client):
        r = client.get("/api/project-state")
        assert r.status_code == 200
        d = r.json()
        assert "families" in d
        assert "cooling_down" in d

    def test_recovery_summary(self, client):
        r = client.get("/api/project-state/recovery")
        assert r.status_code == 200
        d = r.json()
        assert "families" in d
        assert "memory_constraints" in d

    def test_family_availability(self, client):
        r = client.get("/api/project-state/family/test")
        assert r.status_code == 200
        assert r.json()["available"] is True

    def test_reset_cooldown_404(self, client):
        r = client.post("/api/project-state/family/nonexistent/reset-cooldown")
        assert r.status_code == 404

    def test_fingerprints(self, client):
        r = client.get("/api/project-state/fingerprints")
        assert r.status_code == 200
        assert "fingerprints" in r.json()

    def test_no_secrets(self, client):
        r = client.get("/api/project-state")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text
