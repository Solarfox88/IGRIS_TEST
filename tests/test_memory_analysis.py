"""Tests for Sprint 24 — LLM Memory Analysis.

Verifies:
- Failure pattern detection
- Root cause identification
- Remediation suggestions
- Lessons learned extraction
- Deterministic fallback
- Secret redaction
- Advisory-only (no execution)
- Endpoints
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from igris.core.decision_memory import (
    record_decision,
    record_failure,
    record_saturation,
    record_remediation_attempt,
)
from igris.core.memory_analysis import (
    _analyze_failure_patterns,
    _identify_root_causes,
    _infer_cause,
    _suggest_remediations,
    _extract_lessons,
    analyze_memory,
    get_analysis_summary,
    get_lessons_learned,
)
from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_env(tmp_path):
    root = tmp_path / "test_project"
    root.mkdir()
    for d in [".igris/memory", ".igris/tasks", ".igris/timeline",
              ".igris/missions", ".igris/reports/decisions"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def populated_memory(project_env):
    """Create a project with diverse memory events."""
    pr = str(project_env)
    # Multiple failures in 'code' family
    for i in range(4):
        record_failure(
            title=f"Code task {i} failed",
            family="code",
            reason="syntax error in generated code",
            project_root=pr,
        )
    # Some failures in 'test' family
    for i in range(2):
        record_failure(
            title=f"Test {i} failed",
            family="test",
            reason="assertion failed",
            project_root=pr,
        )
    # Single failure in 'deploy'
    record_failure(
        title="Deploy timeout",
        family="deploy",
        reason="timed out waiting for response",
        project_root=pr,
    )
    # Decisions
    for i in range(3):
        record_decision(
            title=f"Selected code task {i}",
            family="code",
            outcome="success" if i < 2 else "failure",
            project_root=pr,
        )
    # Saturation
    record_saturation(
        family="config",
        reason="Too many config changes",
        project_root=pr,
    )
    # Remediation
    record_remediation_attempt(
        title="Fix code generation",
        family="code",
        outcome="success",
        project_root=pr,
    )
    return project_env


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Failure pattern detection
# ---------------------------------------------------------------------------


class TestFailurePatterns:
    """Detect repeated failure patterns."""

    def test_no_failures_empty(self, project_env):
        patterns = _analyze_failure_patterns(str(project_env))
        assert patterns == []

    def test_detects_repeated_family(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        assert len(patterns) > 0
        code_pattern = next((p for p in patterns if p["family"] == "code"), None)
        assert code_pattern is not None
        assert code_pattern["failure_count"] >= 4
        assert code_pattern["is_repeated"] is True

    def test_single_failure_not_pattern(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        deploy_pattern = next((p for p in patterns if p["family"] == "deploy"), None)
        assert deploy_pattern is None  # Only 1 failure, threshold is 2

    def test_includes_sample_reasons(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        code_pattern = next((p for p in patterns if p["family"] == "code"), None)
        assert len(code_pattern["sample_reasons"]) > 0


# ---------------------------------------------------------------------------
# Root cause identification
# ---------------------------------------------------------------------------


class TestRootCauses:
    """Identify likely root causes."""

    def test_critical_severity(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        causes = _identify_root_causes(patterns)
        # code has 4+ failures -> should be high or critical
        code_cause = next((c for c in causes if c["family"] == "code"), None)
        assert code_cause is not None
        assert code_cause["severity"] in ("high", "critical")

    def test_moderate_severity(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        causes = _identify_root_causes(patterns)
        test_cause = next((c for c in causes if c["family"] == "test"), None)
        assert test_cause is not None
        assert test_cause["severity"] == "moderate"

    def test_infer_cause_timeout(self):
        cause = _infer_cause("deploy", ["operation timed out"])
        assert "timeout" in cause.lower()

    def test_infer_cause_permission(self):
        cause = _infer_cause("deploy", ["permission denied"])
        assert "permission" in cause.lower()

    def test_infer_cause_missing(self):
        cause = _infer_cause("config", ["file not found"])
        assert "missing" in cause.lower()

    def test_infer_cause_unknown(self):
        cause = _infer_cause("other", ["something weird"])
        assert "unknown" in cause.lower() or "investigation" in cause.lower()


# ---------------------------------------------------------------------------
# Remediation suggestions
# ---------------------------------------------------------------------------


class TestRemediations:
    """Suggest remediation strategies."""

    def test_has_strategies(self, populated_memory):
        patterns = _analyze_failure_patterns(str(populated_memory))
        causes = _identify_root_causes(patterns)
        remediations = _suggest_remediations(patterns, causes)
        assert len(remediations) > 0
        for r in remediations:
            assert len(r["strategies"]) > 0

    def test_critical_has_block_suggestion(self):
        causes = [{
            "family": "code",
            "severity": "critical",
            "likely_cause": "test",
            "suggestion": "test",
            "evidence_count": 5,
        }]
        remediations = _suggest_remediations([], causes)
        assert any("block" in s.lower() for r in remediations for s in r["strategies"])


# ---------------------------------------------------------------------------
# Lessons learned
# ---------------------------------------------------------------------------


class TestLessonsLearned:
    """Extract lessons from memory."""

    def test_no_events_no_lessons(self, project_env):
        lessons = _extract_lessons(str(project_env))
        assert lessons == []

    def test_detects_recovery(self, populated_memory):
        lessons = _extract_lessons(str(populated_memory))
        recovery = [l for l in lessons if l["type"] == "recovery"]
        assert len(recovery) > 0
        assert any("code" in l["family"] for l in recovery)

    def test_detects_persistent_saturation(self, populated_memory):
        lessons = _extract_lessons(str(populated_memory))
        persistent = [l for l in lessons if l["type"] == "persistent_block"]
        assert len(persistent) > 0
        assert any("config" in l["family"] for l in persistent)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


class TestFullAnalysis:
    """Full memory analysis pipeline."""

    def test_empty_memory(self, project_env):
        result = analyze_memory(str(project_env))
        assert result["advisory_only"] is True
        assert "deterministic" in result
        assert "latency_ms" in result

    def test_populated_memory(self, populated_memory):
        result = analyze_memory(str(populated_memory))
        assert result["advisory_only"] is True
        det = result["deterministic"]
        assert len(det["failure_patterns"]) > 0
        assert len(det["root_causes"]) > 0

    def test_analysis_summary(self, populated_memory):
        summary = get_analysis_summary(str(populated_memory))
        assert summary["advisory_only"] is True
        assert summary["pattern_count"] > 0
        assert "saturated_families" in summary
        assert "avoid_families" in summary

    def test_lessons_endpoint(self, populated_memory):
        result = get_lessons_learned(str(populated_memory))
        assert result["advisory_only"] is True
        assert result["count"] > 0
        assert len(result["lessons"]) > 0


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    """Secrets must not appear in analysis output."""

    def test_secrets_redacted_in_analysis(self, project_env):
        pr = str(project_env)
        record_failure(
            title="Failed with sk-abcdefghijklmnopqrstuvwxyz1234",
            family="code",
            reason="API key sk-abcdefghijklmnopqrstuvwxyz1234 leaked",
            project_root=pr,
        )
        record_failure(
            title="Another code failure",
            family="code",
            reason="second failure",
            project_root=pr,
        )
        result = analyze_memory(pr)
        text = json.dumps(result)
        assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in text


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """HTTP endpoint tests."""

    def test_analyze_endpoint(self, client):
        r = client.post("/api/memory/analyze")
        assert r.status_code == 200
        data = r.json()
        assert data["advisory_only"] is True

    def test_analysis_summary_endpoint(self, client):
        r = client.get("/api/memory/analysis")
        assert r.status_code == 200
        data = r.json()
        assert data["advisory_only"] is True

    def test_lessons_endpoint(self, client):
        r = client.get("/api/memory/lessons")
        assert r.status_code == 200
        data = r.json()
        assert data["advisory_only"] is True
        assert "count" in data

    def test_no_secrets_in_endpoints(self, client):
        r1 = client.post("/api/memory/analyze")
        r2 = client.get("/api/memory/analysis")
        r3 = client.get("/api/memory/lessons")
        for r in [r1, r2, r3]:
            text = json.dumps(r.json())
            assert "sk-" not in text or "sk-" in text and len(text.split("sk-")[1].split('"')[0]) < 5

    def test_advisory_only_flag(self, client):
        """All analysis endpoints must declare advisory_only."""
        for url in ["/api/memory/analyze", "/api/memory/analysis", "/api/memory/lessons"]:
            if url.startswith("/api/memory/analyze") and not url.endswith("e"):
                continue
            method = "post" if url == "/api/memory/analyze" else "get"
            r = getattr(client, method)(url)
            assert r.status_code == 200
            assert r.json().get("advisory_only") is True


# ---------------------------------------------------------------------------
# Safety cross-checks
# ---------------------------------------------------------------------------


class TestSafetyCrossChecks:
    """Cross-cutting safety verifications."""

    def test_analysis_never_executes(self, populated_memory):
        """Analysis is advisory — must never trigger actions."""
        result = analyze_memory(str(populated_memory))
        assert result["advisory_only"] is True
        # Check no execution keys
        text = json.dumps(result)
        assert "execute" not in text.lower() or "auto_execute" not in text.lower()

    def test_bounded_output(self, project_env):
        """Analysis output should be bounded even with many events."""
        pr = str(project_env)
        for i in range(50):
            record_failure(
                title=f"Failure {i}",
                family="code",
                reason=f"Error {i}",
                project_root=pr,
            )
        result = analyze_memory(pr)
        text = json.dumps(result)
        # Should not be enormous
        assert len(text) < 50000

    def test_deterministic_fallback_always_works(self, populated_memory):
        """Deterministic analysis always produces results."""
        result = analyze_memory(str(populated_memory))
        det = result["deterministic"]
        assert "failure_patterns" in det
        assert "root_causes" in det
        assert "remediations" in det
        assert "lessons" in det
        assert "constraints" in det
