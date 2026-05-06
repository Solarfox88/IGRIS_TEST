"""Tests for igris.core.crash_recovery — crash handling, reports, recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from igris.core.crash_recovery import (
    FAILURE_CATEGORIES,
    REMEDIATION_SUGGESTIONS,
    CrashReport,
    classify_exception,
    get_crash_report,
    handle_crash,
    list_crash_reports,
    load_good_state,
    redact_stacktrace,
    save_crash_report,
    save_good_state,
)


# ---------------------------------------------------------------------------
# CrashReport model
# ---------------------------------------------------------------------------


class TestCrashReport:
    def test_to_dict_structure(self):
        r = CrashReport(
            failure_category="import_error",
            failure_description="No module named foo",
            exception_type="ImportError",
            redacted_stacktrace="Traceback...",
        )
        d = r.to_dict()
        assert d["failure_category"] == "import_error"
        assert d["exception_type"] == "ImportError"
        assert d["id"].startswith("crash-")
        assert d["timestamp"]

    def test_secret_redaction_in_report(self):
        r = CrashReport(
            failure_description="Error with key sk-1234567890abcdef1234567890abcdef",
        )
        d = r.to_dict()
        assert "sk-" not in d["failure_description"]

    def test_to_markdown(self):
        r = CrashReport(
            failure_category="connection_error",
            failure_description="Cannot connect",
            exception_type="ConnectionError",
            redacted_stacktrace="Traceback ...",
            suggested_remediation="Check network",
            mission_id="m1",
            task_id="t1",
        )
        md = r.to_markdown()
        assert "# IGRIS Crash Report" in md
        assert "connection_error" in md
        assert "Check network" in md
        assert "Mission: m1" in md
        assert "Task: t1" in md

    def test_to_timeline_event(self):
        r = CrashReport(
            failure_category="timeout_error",
            failure_description="Timed out",
            mission_id="m1",
        )
        ev = r.to_timeline_event()
        assert ev["type"] == "crash"
        assert ev["mission_id"] == "m1"
        assert "timeout" in ev["title"].lower()

    def test_context_redacted(self):
        r = CrashReport(
            context={"key": "sk-1234567890abcdef1234567890abcdef"},
        )
        d = r.to_dict()
        assert "sk-" not in d["context"]["key"]


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


class TestClassifyException:
    def test_import_error(self):
        assert classify_exception(ImportError("no foo")) == "import_error"

    def test_module_not_found(self):
        assert classify_exception(ModuleNotFoundError("no bar")) == "import_error"

    def test_connection_error(self):
        assert classify_exception(ConnectionError("refused")) == "connection_error"

    def test_timeout_error(self):
        assert classify_exception(TimeoutError("timed out")) == "timeout_error"

    def test_permission_error(self):
        assert classify_exception(PermissionError("denied")) == "permission_error"

    def test_file_not_found(self):
        assert classify_exception(FileNotFoundError("gone")) == "file_not_found"

    def test_json_error(self):
        assert classify_exception(json.JSONDecodeError("bad", "doc", 0)) == "json_error"

    def test_validation_error(self):
        assert classify_exception(ValueError("validation failed")) == "validation_error"

    def test_unknown(self):
        assert classify_exception(RuntimeError("something random")) == "unknown"

    def test_config_error(self):
        assert classify_exception(RuntimeError("missing configuration key")) == "config_error"

    def test_llm_error(self):
        assert classify_exception(RuntimeError("Ollama unreachable")) == "llm_error"

    def test_git_error(self):
        assert classify_exception(RuntimeError("git fatal: not a repo")) == "git_error"

    def test_timeout_in_message(self):
        assert classify_exception(RuntimeError("operation timeout exceeded")) == "timeout_error"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedactStacktrace:
    def test_redacts_secrets(self):
        tb = "File foo.py: api_key=sk-1234567890abcdef1234567890abcdef"
        result = redact_stacktrace(tb)
        assert "sk-" not in result

    def test_preserves_normal_text(self):
        tb = "File foo.py, line 42, in bar\n    return x + 1"
        result = redact_stacktrace(tb)
        assert "foo.py" in result


# ---------------------------------------------------------------------------
# Good state persistence
# ---------------------------------------------------------------------------


class TestGoodState:
    def test_save_and_load(self, tmp_path):
        state = {"tests_passing": True, "step": 3, "loop_count": 5}
        save_good_state(state, str(tmp_path))
        loaded = load_good_state(str(tmp_path))
        assert loaded is not None
        assert loaded["tests_passing"] is True
        assert loaded["step"] == 3
        assert "saved_at" in loaded

    def test_load_nonexistent(self, tmp_path):
        result = load_good_state(str(tmp_path))
        assert result is None

    def test_overwrite_state(self, tmp_path):
        save_good_state({"v": 1}, str(tmp_path))
        save_good_state({"v": 2}, str(tmp_path))
        loaded = load_good_state(str(tmp_path))
        assert loaded["v"] == 2


# ---------------------------------------------------------------------------
# Crash report persistence
# ---------------------------------------------------------------------------


class TestCrashPersistence:
    def test_save_and_list(self, tmp_path):
        r = CrashReport(
            failure_category="test_failure",
            failure_description="assertion failed",
        )
        path = save_crash_report(r, str(tmp_path))
        assert path.exists()

        # Also creates markdown
        md_path = path.with_suffix(".md")
        assert md_path.exists()

        reports = list_crash_reports(str(tmp_path))
        assert len(reports) == 1
        assert reports[0]["failure_category"] == "test_failure"

    def test_get_specific_report(self, tmp_path):
        r = CrashReport(failure_category="config_error")
        save_crash_report(r, str(tmp_path))
        loaded = get_crash_report(r.id, str(tmp_path))
        assert loaded is not None
        assert loaded["id"] == r.id

    def test_get_nonexistent(self, tmp_path):
        result = get_crash_report("crash-nonexistent", str(tmp_path))
        assert result is None

    def test_list_respects_limit(self, tmp_path):
        for i in range(5):
            r = CrashReport(failure_description=f"crash {i}")
            save_crash_report(r, str(tmp_path))
        reports = list_crash_reports(str(tmp_path), limit=3)
        assert len(reports) == 3

    def test_list_newest_first(self, tmp_path):
        import time
        r1 = CrashReport(failure_description="first")
        save_crash_report(r1, str(tmp_path))
        time.sleep(0.05)
        r2 = CrashReport(failure_description="second")
        save_crash_report(r2, str(tmp_path))
        reports = list_crash_reports(str(tmp_path))
        assert reports[0]["id"] == r2.id


# ---------------------------------------------------------------------------
# handle_crash (main entry)
# ---------------------------------------------------------------------------


class TestHandleCrash:
    def test_handles_import_error(self, tmp_path):
        try:
            raise ImportError("No module named 'nonexistent'")
        except ImportError as exc:
            report = handle_crash(
                exc,
                mission_id="m1",
                task_id="t1",
                project_root=str(tmp_path),
            )
        assert report.failure_category == "import_error"
        assert report.mission_id == "m1"
        assert report.redacted_stacktrace
        assert report.suggested_remediation

    def test_handles_connection_error(self, tmp_path):
        try:
            raise ConnectionError("Connection refused")
        except ConnectionError as exc:
            report = handle_crash(exc, project_root=str(tmp_path))
        assert report.failure_category == "connection_error"

    def test_persists_report(self, tmp_path):
        try:
            raise RuntimeError("test crash")
        except RuntimeError as exc:
            report = handle_crash(exc, project_root=str(tmp_path))
        loaded = get_crash_report(report.id, str(tmp_path))
        assert loaded is not None

    def test_includes_trace_id(self, tmp_path):
        try:
            raise ValueError("bad value")
        except ValueError as exc:
            report = handle_crash(exc, trace_id="trace-abc", project_root=str(tmp_path))
        assert report.trace_id == "trace-abc"

    def test_auto_trace_id(self, tmp_path):
        try:
            raise ValueError("oops")
        except ValueError as exc:
            report = handle_crash(exc, project_root=str(tmp_path))
        assert report.trace_id.startswith("trace-")

    def test_redacts_secrets_in_stacktrace(self, tmp_path):
        try:
            api_key = "sk-1234567890abcdef1234567890abcdef"
            raise RuntimeError(f"Failed with key {api_key}")
        except RuntimeError as exc:
            report = handle_crash(exc, project_root=str(tmp_path))
        assert "sk-" not in report.redacted_stacktrace

    def test_includes_context(self, tmp_path):
        try:
            raise RuntimeError("oops")
        except RuntimeError as exc:
            report = handle_crash(
                exc,
                context={"step": 5, "family": "test_repair"},
                project_root=str(tmp_path),
            )
        d = report.to_dict()
        assert "step" in d["context"]

    def test_includes_good_state(self, tmp_path):
        save_good_state({"tests": True}, str(tmp_path))
        try:
            raise RuntimeError("crash")
        except RuntimeError as exc:
            report = handle_crash(exc, project_root=str(tmp_path))
        assert report.last_known_good_state is not None
        assert report.last_known_good_state["tests"] is True


# ---------------------------------------------------------------------------
# Constants coverage
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_categories_have_descriptions(self):
        for cat in FAILURE_CATEGORIES:
            assert FAILURE_CATEGORIES[cat]

    def test_all_categories_have_remediation(self):
        for cat in FAILURE_CATEGORIES:
            assert cat in REMEDIATION_SUGGESTIONS
