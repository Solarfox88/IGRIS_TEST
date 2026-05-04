"""Tests for Mission API endpoints."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from igris.web.server import create_app


def _client(tmp_path):
    from pathlib import Path
    from igris.models.config import CONFIG
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / ".igris" / "tasks").mkdir(parents=True)
    (root / ".igris" / "timeline").mkdir(parents=True)
    (root / ".igris" / "missions").mkdir(parents=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


def test_list_missions_empty(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/missions")
    assert r.status_code == 200
    assert r.json()["missions"] == []


def test_create_mission(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/missions", json={
        "title": "Test Mission",
        "description": "1. Analyze\n2. Implement\n3. Test",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Test Mission"
    assert data["status"] == "created"
    assert "id" in data


def test_create_mission_no_title(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/missions", json={"description": "whatever"})
    assert r.status_code == 400


def test_get_mission(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={"title": "M1", "description": "D1"})
    mid = r1.json()["id"]
    r2 = c.get(f"/api/missions/{mid}")
    assert r2.status_code == 200
    assert r2.json()["title"] == "M1"


def test_get_mission_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/missions/nonexistent")
    assert r.status_code == 404


def test_plan_mission(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={
        "title": "Plan me",
        "description": "1. Read code\n2. Fix bug\n3. Test fix",
    })
    mid = r1.json()["id"]
    r2 = c.post(f"/api/missions/{mid}/plan")
    assert r2.status_code == 200
    data = r2.json()
    mission = data.get("mission", data)
    assert mission["status"] == "planned"
    assert len(mission["steps"]) == 3


def test_plan_mission_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/missions/nonexistent/plan")
    assert r.status_code == 404


def test_materialize_tasks(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={
        "title": "Materialize me",
        "description": "1. Analyze repo\n2. Implement feature",
    })
    mid = r1.json()["id"]
    c.post(f"/api/missions/{mid}/plan")
    r3 = c.post(f"/api/missions/{mid}/materialize-tasks")
    assert r3.status_code == 200
    data = r3.json()
    assert data["status"] == "active"
    assert len(data.get("task_ids", [])) == 2


def test_materialize_without_plan(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={"title": "NoPlan"})
    mid = r1.json()["id"]
    r2 = c.post(f"/api/missions/{mid}/materialize-tasks")
    assert r2.status_code == 404


def test_mission_graph(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={
        "title": "Graph me",
        "description": "1. Step A\n2. Step B\n3. Step C",
    })
    mid = r1.json()["id"]
    c.post(f"/api/missions/{mid}/plan")
    r3 = c.get(f"/api/missions/{mid}/graph")
    assert r3.status_code == 200
    data = r3.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2


def test_graph_not_found(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/missions/nonexistent/graph")
    assert r.status_code == 404


def test_materialized_tasks_appear_in_task_list(tmp_path):
    c = _client(tmp_path)
    r1 = c.post("/api/missions", json={
        "title": "Tasks visible",
        "description": "1. Do thing",
    })
    mid = r1.json()["id"]
    c.post(f"/api/missions/{mid}/plan")
    c.post(f"/api/missions/{mid}/materialize-tasks")
    r4 = c.get("/api/tasks")
    assert r4.status_code == 200
    tasks = r4.json()["tasks"]
    assert len(tasks) >= 1
    assert any(t["source"] == "mission" for t in tasks)


def test_no_secrets_in_mission_response(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/missions", json={
        "title": "Secret mission",
        "description": "API_KEY=sk-abc123 do things",
    })
    assert r.status_code == 200
    text = r.text
    assert "sk-abc123" not in text


def test_list_after_create(tmp_path):
    c = _client(tmp_path)
    c.post("/api/missions", json={"title": "Listed"})
    r = c.get("/api/missions")
    assert r.status_code == 200
    assert len(r.json()["missions"]) >= 1
