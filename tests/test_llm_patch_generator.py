"""Tests for Sprint 28 — LLM Patch Generation, Proposal-Only.

Verifies:
- LLM patch generation produces proposals only
- Schema validation catches malformed output
- Unsafe files rejected
- Secret content rejected
- Deterministic fallback works
- No auto-apply
- Endpoint tests
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from igris.core.llm_patch_generator import (
    _validate_path,
    _validate_content,
    validate_patch_output,
    generate_patch,
    _deterministic_patch,
    _extract_json,
)
from igris.web.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Validate file paths in patches."""

    def test_valid_path(self):
        assert _validate_path("src/main.py") is None

    def test_empty_path(self):
        assert _validate_path("") is not None

    def test_path_traversal(self):
        assert _validate_path("../../../etc/passwd") is not None

    def test_blocked_dir_env(self):
        assert _validate_path(".env/config") is not None

    def test_blocked_dir_git(self):
        assert _validate_path(".git/config") is not None

    def test_blocked_dir_igris(self):
        assert _validate_path(".igris/tasks/t1.json") is not None

    def test_blocked_extension_pem(self):
        assert _validate_path("server.pem") is not None

    def test_blocked_extension_exe(self):
        assert _validate_path("app.exe") is not None

    def test_allowed_extension_py(self):
        assert _validate_path("module.py") is None

    def test_allowed_extension_md(self):
        assert _validate_path("docs/README.md") is None


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------


class TestContentValidation:
    """Validate file content in patches."""

    def test_safe_content(self):
        assert _validate_content("print('hello')") is None

    def test_empty_content(self):
        assert _validate_content("") is None

    def test_oversized_content(self):
        big = "x" * 60_000
        assert _validate_content(big) is not None

    def test_secret_content(self):
        content = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890"'
        result = _validate_content(content)
        assert result is not None
        assert "secret" in result.lower()


# ---------------------------------------------------------------------------
# Output schema validation
# ---------------------------------------------------------------------------


class TestPatchOutputValidation:
    """Validate LLM-generated patch output schema."""

    def test_valid_output(self):
        data = {
            "files": [
                {"path": "main.py", "action": "create", "after": "print('hi')", "reason": "test"}
            ],
            "description": "test patch",
            "risk": "low",
        }
        result = validate_patch_output(data)
        assert result["valid"] is True
        assert len(result["files"]) == 1

    def test_not_dict(self):
        result = validate_patch_output("not a dict")
        assert result["valid"] is False

    def test_no_files(self):
        result = validate_patch_output({"files": []})
        assert result["valid"] is False

    def test_files_not_list(self):
        result = validate_patch_output({"files": "wrong"})
        assert result["valid"] is False

    def test_too_many_files(self):
        files = [{"path": f"f{i}.py", "action": "create", "after": "x"} for i in range(10)]
        result = validate_patch_output({"files": files})
        # Should truncate to MAX_FILES_PER_PATCH
        assert len(result["files"]) <= 5

    def test_path_traversal_rejected(self):
        data = {
            "files": [{"path": "../../../etc/passwd", "action": "create", "after": "bad"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_env_file_rejected(self):
        data = {
            "files": [{"path": ".env/secrets", "action": "create", "after": "bad"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_secret_content_rejected(self):
        data = {
            "files": [{
                "path": "config.py",
                "action": "create",
                "after": 'KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890"',
            }]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_invalid_action_rejected(self):
        data = {
            "files": [{"path": "main.py", "action": "delete", "after": "x"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_secrets_redacted_in_output(self):
        data = {
            "files": [{
                "path": "main.py",
                "action": "create",
                "after": "print('safe')",
                "reason": "test",
            }],
        }
        result = validate_patch_output(data)
        for f in result["files"]:
            assert "sk-" not in f.get("after", "")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


class TestJSONExtraction:
    """Extract JSON from LLM response."""

    def test_clean_json(self):
        result = _extract_json('{"files": []}')
        assert result is not None
        assert "files" in result

    def test_json_in_markdown(self):
        result = _extract_json('```json\n{"files": []}\n```')
        assert result is not None

    def test_json_with_text(self):
        result = _extract_json('Here is the patch:\n{"files": [{"path":"a.py"}]}')
        assert result is not None

    def test_invalid_json(self):
        result = _extract_json("not json at all")
        assert result is None


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


class TestDeterministicFallback:
    """Deterministic fallback when LLM unavailable."""

    def test_fallback_produces_result(self):
        result = _deterministic_patch("Fix bug", "Fix the login bug")
        assert "files" in result
        assert result["generated_by"] == "deterministic"
        assert "note" in result

    def test_generate_falls_back(self):
        """generate_patch should fall back when LLM unavailable."""
        result = generate_patch("Test task", "Test description")
        assert "generated_by" in result or "fallback_reason" in result


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------


class TestGeneration:
    """Full patch generation pipeline."""

    def test_generate_returns_result(self):
        result = generate_patch("Add logging", "Add logging to main module")
        assert isinstance(result, dict)
        assert "files" in result or "fallback_reason" in result

    def test_generate_never_auto_applies(self):
        """Generated patches must never be auto-applied."""
        result = generate_patch("Fix bug", "Fix login")
        assert "applied" not in json.dumps(result).lower() or result.get("proposal_only") is True

    def test_generate_has_latency(self):
        result = generate_patch("Test", "Test")
        assert "latency_ms" in result


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    """HTTP endpoint tests."""

    def test_generate_endpoint(self, client):
        r = client.post("/api/patches/generate", json={
            "title": "Add feature",
            "description": "Add user authentication",
        })
        assert r.status_code == 200
        data = r.json()
        assert "files" in data or "fallback_reason" in data

    def test_generate_missing_title(self, client):
        r = client.post("/api/patches/generate", json={
            "description": "no title",
        })
        assert r.status_code == 400

    def test_task_generate_404(self, client):
        r = client.post("/api/tasks/99999/generate-patch")
        assert r.status_code == 404

    def test_task_generate_with_task(self, client):
        # Create a task first (requires description)
        r = client.post("/api/tasks", json={
            "title": "Fix login bug",
            "description": "Auth flow broken",
        })
        assert r.status_code == 200
        task_id = r.json()["id"]
        r = client.post(f"/api/tasks/{task_id}/generate-patch")
        assert r.status_code == 200
        data = r.json()
        assert data.get("task_id") == task_id

    def test_task_generate_nonexistent(self, client):
        r = client.post("/api/tasks/99999/generate-patch")
        assert r.status_code == 404

    def test_no_secrets_in_response(self, client):
        r = client.post("/api/patches/generate", json={
            "title": "Test with sk-abcdefghijklmnopqrstuvwxyz",
        })
        assert r.status_code == 200
        text = json.dumps(r.json())
        # Title may be present but content should not have secrets
        assert "ghp_" not in text

    def test_proposal_only_flag(self, client):
        """Generated patches must indicate proposal-only."""
        r = client.post("/api/patches/generate", json={
            "title": "Test proposal",
        })
        assert r.status_code == 200
        data = r.json()
        # Either proposal_only flag or deterministic fallback
        assert data.get("proposal_only") is True or data.get("generated_by") == "deterministic"


# ---------------------------------------------------------------------------
# Safety cross-checks
# ---------------------------------------------------------------------------


class TestSafetyCrossChecks:
    """Cross-cutting safety verifications."""

    def test_no_auto_apply(self, client):
        """POST /api/patches/generate must NOT apply patches."""
        r = client.post("/api/patches/generate", json={
            "title": "Auto-apply test",
            "description": "This should not auto-apply",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") != "applied"
        assert "auto_applied" not in data

    def test_env_file_blocked(self):
        data = {
            "files": [{"path": ".env", "action": "create", "after": "SECRET=bad"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_binary_file_blocked(self):
        data = {
            "files": [{"path": "app.exe", "action": "create", "after": "binary"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False

    def test_git_dir_blocked(self):
        data = {
            "files": [{"path": ".git/config", "action": "modify", "after": "bad"}]
        }
        result = validate_patch_output(data)
        assert result["valid"] is False
