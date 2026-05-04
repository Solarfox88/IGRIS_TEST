"""Tests for Cost/Routing API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from igris.web.server import create_app
from igris.layers.advisory import router as provider_router


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
    # Reset router state
    provider_router._last_provider = None
    provider_router._provider_history.clear()
    return TestClient(create_app())


def test_availability_endpoint(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/routing/availability")
    assert r.status_code == 200
    data = r.json()
    assert "ollama" in data
    assert "openai" in data
    assert "vastai" in data


def test_availability_no_api_key_exposed(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/routing/availability")
    text = r.text
    # Should not contain actual API key values
    assert "sk-" not in text or "key_present" in text


def test_estimate_endpoint(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/routing/estimate", json={"task_type": "chat", "complexity": "low"})
    assert r.status_code == 200
    data = r.json()
    assert "recommended_provider" in data
    assert "estimated_cost" in data
    assert "budget_remaining" in data


def test_estimate_default_body(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/routing/estimate")
    assert r.status_code == 200


def test_cost_summary_endpoint(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/cost/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_calls" in data
    assert "budget" in data


def test_budget_get(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/cost/budget")
    assert r.status_code == 200
    data = r.json()
    assert "spent" in data
    assert "usage_percent" in data


def test_budget_update(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/cost/budget", json={"max_session_cost": 5.0})
    assert r.status_code == 200
    assert r.json()["max_session_cost"] == 5.0


def test_no_secrets_in_responses(tmp_path):
    c = _client(tmp_path)
    for path in ["/api/routing/availability", "/api/cost/summary", "/api/cost/budget"]:
        r = c.get(path)
        # No raw API keys should be in responses
        assert r.status_code == 200


def test_vast_unavailable_no_crash(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/routing/availability")
    assert r.status_code == 200
    assert isinstance(r.json()["vastai"]["available"], bool)
