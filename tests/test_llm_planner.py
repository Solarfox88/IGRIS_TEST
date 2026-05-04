"""Tests for Sprint 23 — LLM-Based Planning, Safe Schema Mode.

Verifies:
- Schema validation (valid/invalid JSON, missing fields, unsafe capabilities)
- Deterministic fallback on invalid LLM output
- Success criteria enforced
- Risk validation
- Secret redaction
- No auto-execution
- Endpoint mode parameter
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from igris.core.llm_planner import (
    PlanValidationResult,
    SAFE_CAPABILITIES,
    UNSAFE_CAPABILITIES,
    VALID_FAMILIES,
    VALID_RISKS,
    _extract_json,
    explain_plan,
    plan_mission_with_mode,
    validate_plan_schema,
)
from igris.core.mission_planner import Mission, PlanStep, save_mission
from igris.models.config import CONFIG
from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_env(tmp_path):
    """Set up a project with .igris directories."""
    root = tmp_path / "test_project"
    root.mkdir()
    for d in [".igris/missions", ".igris/tasks", ".igris/timeline",
              ".igris/memory", ".igris/reports/decisions"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def sample_mission(project_env):
    """Create and save a sample mission."""
    m = Mission(
        title="Fix login bug",
        description="1. Analyze the login flow\n2. Fix the authentication check\n3. Add tests for login",
    )
    save_mission(m, project_root=str(project_env))
    return m, project_env


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Schema validation — valid plans
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Schema validation for LLM plan output."""

    def test_valid_plan(self):
        plan = json.dumps({
            "steps": [
                {
                    "title": "Analyze requirements",
                    "description": "Understand the login flow",
                    "family": "analyze",
                    "success_criteria": ["Requirements documented"],
                    "risk": "low",
                },
                {
                    "title": "Fix auth check",
                    "description": "Update the authentication logic",
                    "family": "code",
                    "success_criteria": ["Auth check works", "Tests pass"],
                    "safe_capabilities": ["read", "write"],
                    "risk": "medium",
                },
            ],
        })
        result = validate_plan_schema(plan)
        assert result.valid is True
        assert len(result.steps) == 2
        assert len(result.errors) == 0

    def test_invalid_json(self):
        result = validate_plan_schema("not valid json {{{")
        assert result.valid is False
        assert any("invalid json" in e.lower() for e in result.errors)

    def test_missing_steps_key(self):
        result = validate_plan_schema('{"plan": []}')
        assert result.valid is False
        assert any("missing" in e.lower() for e in result.errors)

    def test_empty_steps_array(self):
        result = validate_plan_schema('{"steps": []}')
        assert result.valid is False

    def test_step_missing_required_fields(self):
        plan = json.dumps({
            "steps": [{"title": "incomplete step"}],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False
        assert any("missing fields" in e.lower() for e in result.errors)

    def test_empty_success_criteria_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Step",
                "description": "desc",
                "family": "code",
                "success_criteria": [],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False
        assert any("success_criteria" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Schema validation — unsafe capabilities rejected
# ---------------------------------------------------------------------------


class TestUnsafeCapabilities:
    """Unsafe capabilities must be rejected."""

    def test_shell_exec_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Run command",
                "description": "Execute shell",
                "family": "code",
                "success_criteria": ["Command runs"],
                "safe_capabilities": ["shell_exec"],
                "risk": "high",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False
        assert any("unsafe" in e.lower() for e in result.errors)

    def test_auto_push_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Push",
                "description": "Auto push",
                "family": "deploy",
                "success_criteria": ["Pushed"],
                "safe_capabilities": ["auto_push"],
                "risk": "high",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False

    def test_force_push_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Force push",
                "description": "Force",
                "family": "deploy",
                "success_criteria": ["Pushed"],
                "safe_capabilities": ["force_push"],
                "risk": "high",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False

    def test_safe_capabilities_accepted(self):
        plan = json.dumps({
            "steps": [{
                "title": "Read and write",
                "description": "Safe ops",
                "family": "code",
                "success_criteria": ["Done"],
                "safe_capabilities": ["read", "write", "patch_propose"],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is True

    def test_all_unsafe_capabilities_defined(self):
        for cap in UNSAFE_CAPABILITIES:
            assert cap not in SAFE_CAPABILITIES


# ---------------------------------------------------------------------------
# Schema validation — secret content rejected
# ---------------------------------------------------------------------------


class TestSecretRejection:
    """Secret-like content in plan fields rejected."""

    def test_secret_in_title_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Use key sk-abcdefghijklmnopqrstuvwxyz1234",
                "description": "desc",
                "family": "code",
                "success_criteria": ["done"],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False
        assert any("secret" in e.lower() for e in result.errors)

    def test_secret_in_criteria_rejected(self):
        plan = json.dumps({
            "steps": [{
                "title": "Safe title",
                "description": "desc",
                "family": "code",
                "success_criteria": ["verify ghp_abcdefghijklmnopqrstuvwxyz1234567890 works"],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


class TestJSONExtraction:
    """Extract JSON from LLM responses."""

    def test_plain_json(self):
        result = _extract_json('{"steps": []}')
        assert result == '{"steps": []}'

    def test_markdown_fenced(self):
        text = '```json\n{"steps": []}\n```'
        result = _extract_json(text)
        assert '"steps"' in result

    def test_text_with_json(self):
        text = 'Here is the plan:\n{"steps": [{"title": "a"}]}'
        result = _extract_json(text)
        assert '"steps"' in result

    def test_no_json(self):
        result = _extract_json("no json here")
        assert result == "no json here"


# ---------------------------------------------------------------------------
# Risk and family validation
# ---------------------------------------------------------------------------


class TestRiskAndFamily:
    """Risk and family field validation."""

    def test_unknown_family_defaults_to_other(self):
        plan = json.dumps({
            "steps": [{
                "title": "Step",
                "description": "desc",
                "family": "unknown_family",
                "success_criteria": ["done"],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is True
        assert result.steps[0]["family"] == "other"
        assert len(result.warnings) > 0

    def test_unknown_risk_defaults_to_low(self):
        plan = json.dumps({
            "steps": [{
                "title": "Step",
                "description": "desc",
                "family": "code",
                "success_criteria": ["done"],
                "risk": "extreme",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is True
        assert result.steps[0]["risk"] == "low"

    def test_valid_families(self):
        assert "analyze" in VALID_FAMILIES
        assert "code" in VALID_FAMILIES
        assert "test" in VALID_FAMILIES

    def test_valid_risks(self):
        assert "low" in VALID_RISKS
        assert "medium" in VALID_RISKS
        assert "high" in VALID_RISKS


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


class TestDeterministicFallback:
    """Deterministic planner as fallback."""

    def test_deterministic_mode(self, sample_mission):
        mission, root = sample_mission
        result = plan_mission_with_mode(
            mission.id, mode="deterministic", project_root=str(root),
        )
        assert result is not None
        assert result["planning"]["mode"] == "deterministic"
        assert result["planning"]["fallback_used"] is False
        assert len(result["mission"]["steps"]) >= 1

    def test_llm_mode_falls_back(self, sample_mission):
        """LLM mode falls back to deterministic when LLM unavailable."""
        mission, root = sample_mission
        result = plan_mission_with_mode(
            mission.id, mode="llm", project_root=str(root),
        )
        assert result is not None
        # Should fallback since no LLM is available
        assert result["planning"]["mode"] == "deterministic"
        assert result["planning"]["fallback_used"] is True

    def test_auto_mode_falls_back(self, sample_mission):
        mission, root = sample_mission
        result = plan_mission_with_mode(
            mission.id, mode="auto", project_root=str(root),
        )
        assert result is not None
        assert result["planning"]["fallback_used"] is True

    def test_invalid_mode_returns_none(self, sample_mission):
        mission, root = sample_mission
        result = plan_mission_with_mode(
            mission.id, mode="invalid", project_root=str(root),
        )
        assert result is None

    def test_nonexistent_mission(self, project_env):
        result = plan_mission_with_mode(
            "nonexistent", mode="deterministic", project_root=str(project_env),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Plan explanation
# ---------------------------------------------------------------------------


class TestPlanExplanation:
    """Plan explanation endpoint."""

    def test_explain_planned_mission(self, sample_mission):
        mission, root = sample_mission
        plan_mission_with_mode(
            mission.id, mode="deterministic", project_root=str(root),
        )
        explanation = explain_plan(mission.id, project_root=str(root))
        assert explanation is not None
        assert explanation["step_count"] >= 1
        assert "explanation" in explanation
        assert explanation["max_risk"] in VALID_RISKS

    def test_explain_unplanned_mission(self, project_env):
        m = Mission(title="Unplanned", description="Nothing yet")
        save_mission(m, project_root=str(project_env))
        explanation = explain_plan(m.id, project_root=str(project_env))
        assert explanation is not None
        assert explanation["status"] == "no plan"

    def test_explain_nonexistent(self, project_env):
        result = explain_plan("nonexistent", project_root=str(project_env))
        assert result is None

    def test_explain_no_secrets(self, sample_mission):
        mission, root = sample_mission
        plan_mission_with_mode(
            mission.id, mode="deterministic", project_root=str(root),
        )
        explanation = explain_plan(mission.id, project_root=str(root))
        text = json.dumps(explanation)
        assert "sk-" not in text
        assert "ghp_" not in text


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """HTTP endpoint tests."""

    def test_plan_invalid_mode(self, client):
        r = client.post("/api/missions/fake/plan?mode=invalid")
        assert r.status_code == 400

    def test_plan_nonexistent_mission(self, client):
        r = client.post("/api/missions/nonexistent-id/plan?mode=deterministic")
        assert r.status_code == 404

    def test_explain_nonexistent(self, client):
        r = client.get("/api/missions/nonexistent-id/plan/explain")
        assert r.status_code == 404

    def test_plan_default_mode_is_deterministic(self, client):
        """Default mode should be deterministic."""
        # Create a mission first
        r = client.post("/api/missions", json={
            "title": "Test mission",
            "description": "1. Analyze code\n2. Fix bug",
        })
        assert r.status_code == 200
        mission_id = r.json()["id"]

        # Plan with default mode
        r = client.post(f"/api/missions/{mission_id}/plan")
        assert r.status_code == 200
        data = r.json()
        assert data["planning"]["mode"] == "deterministic"

    def test_explain_endpoint_after_plan(self, client):
        r = client.post("/api/missions", json={
            "title": "Explain test",
            "description": "Fix the auth module",
        })
        mission_id = r.json()["id"]
        client.post(f"/api/missions/{mission_id}/plan")
        r = client.get(f"/api/missions/{mission_id}/plan/explain")
        assert r.status_code == 200
        data = r.json()
        assert "step_count" in data
        assert "explanation" in data


# ---------------------------------------------------------------------------
# Safety cross-checks
# ---------------------------------------------------------------------------


class TestSafetyCrossChecks:
    """Cross-cutting safety verifications."""

    def test_no_auto_execution_in_plan(self):
        """Plans should never contain auto-execution capabilities."""
        plan = json.dumps({
            "steps": [{
                "title": "Auto execute",
                "description": "Execute automatically",
                "family": "code",
                "success_criteria": ["Done"],
                "safe_capabilities": ["auto_merge", "auto_push", "shell_exec"],
                "risk": "high",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False

    def test_success_criteria_enforced(self):
        """Every step must have success_criteria."""
        plan = json.dumps({
            "steps": [{
                "title": "No criteria",
                "description": "Missing criteria",
                "family": "code",
                "success_criteria": [],
                "risk": "low",
            }],
        })
        result = validate_plan_schema(plan)
        assert result.valid is False
        assert any("success_criteria" in e.lower() for e in result.errors)

    def test_risk_field_required(self):
        """Every step must have risk."""
        plan = json.dumps({
            "steps": [{
                "title": "No risk",
                "description": "Missing risk",
                "family": "code",
                "success_criteria": ["done"],
            }],
        })
        result = validate_plan_schema(plan)
        # risk is in REQUIRED_STEP_FIELDS
        assert result.valid is False

    def test_validation_result_to_dict(self):
        r = PlanValidationResult(valid=True, steps=[{"a": 1}])
        d = r.to_dict()
        assert d["valid"] is True
        assert d["step_count"] == 1
