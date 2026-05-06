"""Tests for igris.core.config_validator — config validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from igris.core.config_validator import (
    ConfigIssue,
    ConfigValidationResult,
    validate_all,
    validate_budget,
    validate_config_json,
    validate_env,
    validate_provider,
    validate_safety_policy,
)


# ---------------------------------------------------------------------------
# ConfigIssue model
# ---------------------------------------------------------------------------


class TestConfigIssue:
    def test_to_dict_basic(self):
        i = ConfigIssue(field="FOO", severity="warning", message="not set")
        d = i.to_dict()
        assert d["field"] == "FOO"
        assert d["severity"] == "warning"
        assert "fix_suggestion" not in d

    def test_to_dict_with_fix(self):
        i = ConfigIssue(field="X", severity="error", message="bad", fix_suggestion="fix it")
        d = i.to_dict()
        assert d["fix_suggestion"] == "fix it"

    def test_secret_redacted(self):
        i = ConfigIssue(field="key", severity="info", message="key=sk-1234567890abcdef1234567890abcdef")
        d = i.to_dict()
        assert "sk-" not in d["message"]


# ---------------------------------------------------------------------------
# ConfigValidationResult
# ---------------------------------------------------------------------------


class TestConfigValidationResult:
    def test_empty_valid(self):
        r = ConfigValidationResult()
        d = r.to_dict()
        assert d["valid"] is True
        assert d["issue_count"] == 0

    def test_error_makes_invalid(self):
        r = ConfigValidationResult()
        r.add_issue(ConfigIssue(field="x", severity="error", message="bad"))
        d = r.to_dict()
        assert d["valid"] is False
        assert d["has_errors"] is True

    def test_warning_stays_valid(self):
        r = ConfigValidationResult()
        r.add_issue(ConfigIssue(field="x", severity="warning", message="hmm"))
        d = r.to_dict()
        assert d["valid"] is True
        assert d["has_warnings"] is True


# ---------------------------------------------------------------------------
# .env validation
# ---------------------------------------------------------------------------


class TestValidateEnv:
    def test_no_env_no_vars(self, tmp_path):
        with mock.patch.dict(os.environ, {}, clear=True):
            issues = validate_env(str(tmp_path))
            # Should warn about missing .env
            severity_set = {i.severity for i in issues}
            field_set = {i.field for i in issues}
            assert ".env" in field_set or any("env" in i.message.lower() for i in issues)

    def test_env_exists_no_issues(self, tmp_path):
        (tmp_path / ".env").write_text("# config", encoding="utf-8")
        with mock.patch.dict(os.environ, {
            "LOCAL_LLM_PROVIDER": "ollama",
            "LOCAL_LLM_MODEL": "phi4-mini",
            "LOCAL_LLM_BASE_URL": "http://127.0.0.1:11434",
        }):
            issues = validate_env(str(tmp_path))
            errors = [i for i in issues if i.severity == "error"]
            assert len(errors) == 0

    def test_missing_expected_keys(self, tmp_path):
        (tmp_path / ".env").write_text("# empty", encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            issues = validate_env(str(tmp_path))
            fields = {i.field for i in issues}
            assert "LOCAL_LLM_PROVIDER" in fields or any("not set" in i.message for i in issues)


# ---------------------------------------------------------------------------
# config.json validation
# ---------------------------------------------------------------------------


class TestValidateConfigJson:
    def test_valid_config(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.sample.json").write_text(
            json.dumps({
                "local_llm_provider": "ollama",
                "local_llm_model": "phi4-mini",
                "local_llm_base_url": "http://127.0.0.1:11434",
            }),
            encoding="utf-8",
        )
        issues = validate_config_json(str(tmp_path))
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_invalid_json(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.sample.json").write_text("{bad", encoding="utf-8")
        issues = validate_config_json(str(tmp_path))
        assert any(i.severity == "error" for i in issues)

    def test_missing_file(self, tmp_path):
        issues = validate_config_json(str(tmp_path))
        assert any(i.severity == "warning" for i in issues)


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------


class TestValidateProvider:
    def test_valid_providers(self):
        with mock.patch.dict(os.environ, {
            "LOCAL_LLM_PROVIDER": "ollama",
            "FALLBACK_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
        }):
            issues = validate_provider()
            warnings = [i for i in issues if i.severity == "warning"]
            assert len(warnings) == 0

    def test_unknown_provider(self):
        with mock.patch.dict(os.environ, {"LOCAL_LLM_PROVIDER": "foo_provider"}):
            issues = validate_provider()
            assert any("unknown" in i.message.lower() for i in issues)

    def test_openai_fallback_no_key(self):
        with mock.patch.dict(os.environ, {
            "FALLBACK_LLM_PROVIDER": "openai",
        }, clear=True):
            issues = validate_provider()
            assert any("api key" in i.message.lower() for i in issues)


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


class TestValidateBudget:
    def test_valid_budget(self):
        with mock.patch.dict(os.environ, {"VASTAI_MAX_HOURLY_COST": "0.50"}):
            issues = validate_budget()
            errors = [i for i in issues if i.severity == "error"]
            assert len(errors) == 0

    def test_zero_budget(self):
        with mock.patch.dict(os.environ, {"VASTAI_MAX_HOURLY_COST": "0"}):
            issues = validate_budget()
            assert any(i.severity == "warning" for i in issues)

    def test_high_budget(self):
        with mock.patch.dict(os.environ, {"VASTAI_MAX_HOURLY_COST": "10.00"}):
            issues = validate_budget()
            assert any(i.severity == "warning" for i in issues)

    def test_invalid_budget(self):
        with mock.patch.dict(os.environ, {"VASTAI_MAX_HOURLY_COST": "notanumber"}):
            issues = validate_budget()
            assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Safety policy validation
# ---------------------------------------------------------------------------


class TestValidateSafetyPolicy:
    def test_safe_defaults(self):
        with mock.patch.dict(os.environ, {
            "AUTO_COMMIT": "false",
            "AUTO_PUSH": "false",
            "VASTAI_AUTO_PROVISION": "false",
            "VASTAI_REQUIRE_APPROVAL": "true",
        }):
            issues = validate_safety_policy()
            assert len(issues) == 0

    def test_auto_push_flagged(self):
        with mock.patch.dict(os.environ, {"AUTO_PUSH": "true"}):
            issues = validate_safety_policy()
            assert any(i.severity == "error" and "AUTO_PUSH" in i.field for i in issues)

    def test_auto_commit_flagged(self):
        with mock.patch.dict(os.environ, {"AUTO_COMMIT": "true"}):
            issues = validate_safety_policy()
            assert any(i.severity == "warning" and "AUTO_COMMIT" in i.field for i in issues)

    def test_vastai_auto_provision_flagged(self):
        with mock.patch.dict(os.environ, {"VASTAI_AUTO_PROVISION": "true"}):
            issues = validate_safety_policy()
            assert any("VASTAI_AUTO_PROVISION" in i.field for i in issues)

    def test_vastai_no_approval_flagged(self):
        with mock.patch.dict(os.environ, {"VASTAI_REQUIRE_APPROVAL": "false"}):
            issues = validate_safety_policy()
            assert any("VASTAI_REQUIRE_APPROVAL" in i.field for i in issues)


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------


class TestValidateAll:
    def test_runs_all_sections(self, tmp_path):
        result = validate_all(str(tmp_path))
        d = result.to_dict()
        assert "timestamp" in d
        assert "valid" in d
        assert "issues" in d
        assert len(result.validated_sections) >= 5

    def test_includes_all_section_names(self, tmp_path):
        result = validate_all(str(tmp_path))
        assert "env" in result.validated_sections
        assert "config_json" in result.validated_sections
        assert "provider" in result.validated_sections
        assert "budget" in result.validated_sections
        assert "safety_policy" in result.validated_sections

    def test_no_secret_leak(self, tmp_path):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-1234567890abcdef1234567890abcdef"}):
            result = validate_all(str(tmp_path))
            text = json.dumps(result.to_dict())
            assert "sk-" not in text
