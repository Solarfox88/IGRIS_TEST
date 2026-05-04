"""Tests for Sprint 26 — Human Acceptance Verification.

Verifies:
- Acceptance test document exists and covers all required checks
- Acceptance script exists and is executable
- All endpoints referenced in acceptance test are reachable
- No secrets in any acceptance test endpoint response
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Documentation checks
# ---------------------------------------------------------------------------


class TestAcceptanceDocumentation:
    """Verify acceptance test documentation."""

    def test_acceptance_doc_exists(self):
        doc = Path("docs/HUMAN_ACCEPTANCE_TEST.md")
        assert doc.exists(), "HUMAN_ACCEPTANCE_TEST.md must exist"

    def test_acceptance_doc_covers_all_checks(self):
        doc = Path("docs/HUMAN_ACCEPTANCE_TEST.md")
        content = doc.read_text()
        required_checks = [
            "Clone",
            "Install",
            "Start",
            "Health",
            "UI",
            "Chat",
            "mission",
            "Plan",
            "deterministic",
            "LLM",
            "Materialize",
            "Loop",
            "Patch",
            "Validate",
            "Decision",
            "Diagnostics",
            "Memory",
            "GitHub",
            "Vast.ai",
            "git status",
        ]
        for check in required_checks:
            assert check.lower() in content.lower(), f"Missing check: {check}"

    def test_acceptance_script_exists(self):
        script = Path("scripts/acceptance_check.sh")
        assert script.exists(), "acceptance_check.sh must exist"

    def test_acceptance_script_executable(self):
        script = Path("scripts/acceptance_check.sh")
        assert os.access(str(script), os.X_OK), "acceptance_check.sh must be executable"


# ---------------------------------------------------------------------------
# Endpoint reachability
# ---------------------------------------------------------------------------


class TestEndpointReachability:
    """All endpoints in acceptance test must be reachable."""

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_readiness(self, client):
        r = client.get("/api/readiness")
        assert r.status_code == 200

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_chat_stream(self, client):
        r = client.post("/api/chat/stream", json={"message": "test"})
        assert r.status_code == 200

    def test_create_mission(self, client):
        r = client.post("/api/missions", json={
            "title": "Acceptance test",
            "description": "1. Check\n2. Verify",
        })
        assert r.status_code == 200
        assert "id" in r.json()

    def test_plan_deterministic(self, client):
        r = client.post("/api/missions", json={
            "title": "Plan test",
            "description": "1. Analyze\n2. Fix",
        })
        mid = r.json()["id"]
        r = client.post(f"/api/missions/{mid}/plan?mode=deterministic")
        assert r.status_code == 200
        data = r.json()
        assert data["planning"]["mode"] == "deterministic"

    def test_plan_explain(self, client):
        r = client.post("/api/missions", json={
            "title": "Explain test",
            "description": "1. Read\n2. Write",
        })
        mid = r.json()["id"]
        client.post(f"/api/missions/{mid}/plan")
        r = client.get(f"/api/missions/{mid}/plan/explain")
        assert r.status_code == 200
        assert "explanation" in r.json()

    def test_materialize(self, client):
        r = client.post("/api/missions", json={
            "title": "Materialize test",
            "description": "1. Step one\n2. Step two",
        })
        mid = r.json()["id"]
        client.post(f"/api/missions/{mid}/plan")
        r = client.post(f"/api/missions/{mid}/materialize-tasks")
        assert r.status_code == 200

    def test_loop_step(self, client):
        r = client.post("/api/loop/step")
        assert r.status_code == 200

    def test_diagnostics(self, client):
        r = client.get("/api/diagnostics")
        assert r.status_code == 200

    def test_diagnostics_summary(self, client):
        r = client.get("/api/diagnostics/summary")
        assert r.status_code == 200

    def test_memory_analyze(self, client):
        r = client.post("/api/memory/analyze")
        assert r.status_code == 200
        assert r.json()["advisory_only"] is True

    def test_memory_analysis(self, client):
        r = client.get("/api/memory/analysis")
        assert r.status_code == 200

    def test_memory_lessons(self, client):
        r = client.get("/api/memory/lessons")
        assert r.status_code == 200

    def test_vastai_estimate(self, client):
        r = client.post("/api/vastai/estimate", json={
            "model": "deepseek-r1:32b", "hours": 1,
        })
        assert r.status_code == 200

    def test_vastai_provision_rejected(self, client):
        r = client.post("/api/vastai/provision", json={
            "model": "deepseek-r1:32b",
        })
        assert r.status_code == 200
        data = r.json()
        assert "error" in data or "approval" in json.dumps(data).lower()

    def test_vastai_config(self, client):
        r = client.get("/api/vastai/config")
        assert r.status_code == 200

    def test_decision_reports(self, client):
        r = client.get("/api/decision-reports")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Safety cross-checks
# ---------------------------------------------------------------------------


class TestAcceptanceSafety:
    """Cross-cutting safety for acceptance test."""

    def test_no_secrets_in_health(self, client):
        r = client.get("/api/health")
        text = json.dumps(r.json())
        assert "sk-" not in text or len(text.split("sk-")[1].split('"')[0]) < 5
        assert "ghp_" not in text

    def test_no_secrets_in_vastai_config(self, client):
        r = client.get("/api/vastai/config")
        text = json.dumps(r.json())
        assert "api_key" not in text or r.json().get("api_key_present") is not None

    def test_no_secrets_in_memory(self, client):
        r = client.post("/api/memory/analyze")
        text = json.dumps(r.json())
        assert "ghp_" not in text

    def test_chat_no_crash_without_ollama(self, client):
        """Chat must not crash even without Ollama."""
        r = client.post("/api/chat/stream", json={"message": "test"})
        assert r.status_code == 200

    def test_all_acceptance_endpoints_no_500(self, client):
        """No endpoint in acceptance test should return 500."""
        endpoints = [
            ("GET", "/api/health"),
            ("GET", "/api/readiness"),
            ("GET", "/api/status"),
            ("GET", "/api/diagnostics"),
            ("GET", "/api/diagnostics/summary"),
            ("GET", "/api/memory/analysis"),
            ("GET", "/api/memory/lessons"),
            ("GET", "/api/vastai/config"),
            ("GET", "/api/vastai/status"),
            ("GET", "/api/decision-reports"),
        ]
        for method, url in endpoints:
            r = client.get(url) if method == "GET" else client.post(url)
            assert r.status_code != 500, f"{method} {url} returned 500"
