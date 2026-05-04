"""Tests for Validation API endpoints."""

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
    (root / ".igris" / "validations").mkdir(parents=True)
    (root / ".igris" / "reports").mkdir(parents=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


def test_validate_task_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/tasks/999/validate")
    assert r.status_code == 404


def test_validate_task_no_criteria(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task with no criteria"})
    r = c.post("/api/tasks/1/validate")
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert data["overall_status"] == "needs_review"


def test_validate_task_with_manual_reason(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "manual task"})
    r = c.post("/api/tasks/1/validate", json={"manual_completion_reason": "Checked OK"})
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True
    assert data["overall_status"] == "completed"


def test_get_task_validations(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "some task"})
    c.post("/api/tasks/1/validate")
    r = c.get("/api/tasks/1/validations")
    assert r.status_code == 200
    assert len(r.json()["validations"]) >= 1


def test_get_task_validations_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/tasks/999/validations")
    assert r.status_code == 404


def test_get_validation_by_id(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task"})
    vr = c.post("/api/tasks/1/validate")
    vid = vr.json()["validation_id"]
    r = c.get(f"/api/validations/{vid}")
    assert r.status_code == 200
    assert r.json()["validation_id"] == vid


def test_get_validation_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/validations/nonexistent")
    assert r.status_code == 404


def test_complete_task_without_validation(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "not validated"})
    r = c.post("/api/tasks/1/complete")
    assert r.status_code == 400


def test_complete_task_with_validation(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task"})
    c.post("/api/tasks/1/validate", json={"manual_completion_reason": "OK"})
    r = c.post("/api/tasks/1/complete")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"


def test_complete_task_with_manual_reason_no_validation(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task"})
    r = c.post("/api/tasks/1/complete", json={"manual_completion_reason": "Reviewed manually"})
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


def test_no_secret_in_validation_response(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task"})
    r = c.post("/api/tasks/1/validate", json={
        "manual_completion_reason": "API_KEY=sk-secrettest1234567890123456"
    })
    assert "sk-secrettest1234567890123456" not in r.text


def test_timeline_event_created(tmp_path):
    c = _client(tmp_path)
    c.post("/api/tasks", json={"description": "task"})
    c.post("/api/tasks/1/validate")
    r = c.get("/api/agent/timeline")
    events = r.json().get("timeline", [])
    val_events = [e for e in events if e.get("type") == "validation"]
    assert len(val_events) >= 1
