"""API tests for context-enriched chat endpoints (Sprint 17)."""

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
    for d in [".igris/tasks", ".igris/timeline", ".igris/memory", ".igris/reports/decisions"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


class TestChatContextAPI:
    def test_get_context(self, client):
        r = client.get("/api/chat/context")
        assert r.status_code == 200
        d = r.json()
        assert "sections" in d
        assert "missions" in d["sections"]
        assert "tasks" in d["sections"]

    def test_get_context_summary(self, client):
        r = client.get("/api/chat/context/summary")
        assert r.status_code == 200
        d = r.json()
        assert "tasks_pending" in d
        assert "git_branch" in d
        assert "provider" in d

    def test_no_secrets_in_context(self, client):
        r = client.get("/api/chat/context")
        text = json.dumps(r.json())
        assert "sk-" not in text
        assert "ghp_" not in text

    def test_enriched_stream(self, client):
        r = client.post("/api/chat/stream", json={
            "message": "what tasks are pending?",
            "enrich": True,
        })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        lines = [l for l in r.text.split("\n") if l.startswith("data: ")]
        assert len(lines) >= 1

    def test_enriched_stream_no_secrets(self, client):
        r = client.post("/api/chat/stream", json={
            "message": "show me sk-abcdefghijklmnopqrstuvwxyz",
            "enrich": True,
        })
        text = r.text
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in text

    def test_non_enriched_stream_still_works(self, client):
        r = client.post("/api/chat/stream", json={
            "message": "hello",
            "enrich": False,
        })
        assert r.status_code == 200
        lines = [l for l in r.text.split("\n") if l.startswith("data: ")]
        done = json.loads(lines[-1][6:])
        assert done["type"] == "done"
