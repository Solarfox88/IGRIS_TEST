"""Tests for Autonomous Loop API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from igris.web.server import create_app


def _client(tmp_path):
    from igris.models.config import CONFIG
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / ".igris" / "tasks").mkdir(parents=True)
    (root / ".igris" / "timeline").mkdir(parents=True)
    (root / ".igris" / "missions").mkdir(parents=True)
    (root / ".igris" / "memory").mkdir(parents=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


def test_loop_status(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/loop/status")
    assert r.status_code == 200
    data = r.json()
    assert "running" in data
    assert "max_steps" in data


def test_loop_recent_empty(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/loop/recent")
    assert r.status_code == 200
    assert "steps" in r.json()


def test_loop_step_no_tasks(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/loop/step")
    assert r.status_code == 200
    data = r.json()
    assert data["action_type"] == "stop"
    assert data["outcome"] == "stopped"


def test_loop_step_with_task(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "run unit tests", "family": "test"})
    r = c.post("/api/loop/step")
    assert r.status_code == 200
    data = r.json()
    assert data["action_type"] in ("execute_command", "skip", "stop")


def test_loop_run(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/loop/run", json={"max_steps": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["max_steps"] == 1
    assert data["running"] is False


def test_loop_run_invalid_steps(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/loop/run", json={"max_steps": -1})
    assert r.status_code == 400


def test_loop_run_with_tasks(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "check git status", "family": "git"})
    c.post("/api/tasks", json={"description": "list project files", "family": "analyze"})
    r = c.post("/api/loop/run", json={"max_steps": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["steps_completed"] >= 1


def test_no_secrets_in_loop_response(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "API_KEY=sk-secret1234567890123456", "family": "test"})
    r = c.post("/api/loop/step")
    assert r.status_code == 200
    assert "sk-secret1234567890123456" not in r.text


def test_loop_creates_timeline(tmp_path):
    c = _client(tmp_path)
    c.post("/api/loop/run", json={"max_steps": 1})
    r = c.get("/api/agent/timeline")
    assert r.status_code == 200
    events = r.json().get("timeline", [])
    loop_events = [e for e in events if e.get("type") == "loop"]
    assert len(loop_events) >= 1


def test_no_auto_push_endpoint(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/loop/push")
    assert r.status_code in (404, 405)
