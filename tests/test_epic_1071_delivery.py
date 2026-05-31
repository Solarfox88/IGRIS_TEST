"""Tests for Epic #1071 — DeliveryWorkflow improvements.

Covers: CI failure diagnosis with test parsing, branch hygiene check,
and PR review gate (wait for CI before merge).
"""

import time
import pytest
from unittest.mock import patch, MagicMock, call

from igris.core.delivery_workflow import DeliveryWorkflow, CIStatus, BranchHygieneReport, STALE_BRANCH_AGE_SECONDS


def _make_wf():
    return DeliveryWorkflow(project_root="/tmp")


# ---------------------------------------------------------------------------
# CI failure diagnosis — parse_failing_tests
# ---------------------------------------------------------------------------

class TestParseFailing:
    """parse_failing_tests extracts test node IDs from pytest output."""

    def test_simple_failing_test(self):
        log = "FAILED tests/test_foo.py::TestFoo::test_bar - AssertionError"
        result = _make_wf().parse_failing_tests(log)
        assert "tests/test_foo.py::TestFoo::test_bar" in result

    def test_multiple_failing_tests(self):
        log = (
            "FAILED tests/test_a.py::test_one\n"
            "FAILED tests/test_b.py::TestB::test_two\n"
        )
        result = _make_wf().parse_failing_tests(log)
        assert len(result) == 2
        assert "tests/test_a.py::test_one" in result
        assert "tests/test_b.py::TestB::test_two" in result

    def test_no_duplicates(self):
        log = "FAILED tests/test_a.py::test_one\nFAILED tests/test_a.py::test_one\n"
        result = _make_wf().parse_failing_tests(log)
        assert result.count("tests/test_a.py::test_one") == 1

    def test_empty_log(self):
        assert _make_wf().parse_failing_tests("") == []

    def test_log_without_failures(self):
        log = "1 passed in 0.3s"
        assert _make_wf().parse_failing_tests(log) == []


# ---------------------------------------------------------------------------
# CI failure diagnosis — diagnose_ci_failure_structured
# ---------------------------------------------------------------------------

class TestDiagnoseCiFailure:
    """diagnose_ci_failure_structured returns structured diagnosis."""

    def test_import_error_detected(self):
        log = "ImportError: cannot import name 'foo' from 'bar'"
        result = _make_wf().diagnose_ci_failure_structured(log, ["pytest"])
        assert result["failure_type"] == "import_error"

    def test_syntax_error_detected(self):
        log = "SyntaxError: invalid syntax (foo.py, line 42)"
        result = _make_wf().diagnose_ci_failure_structured(log, ["pytest"])
        assert result["failure_type"] == "syntax_error"

    def test_lint_error_detected(self):
        log = "ruff check: F401 'os' imported but unused"
        result = _make_wf().diagnose_ci_failure_structured(log, ["ruff"])
        assert result["failure_type"] == "lint_error"

    def test_test_failure_detected_with_tests(self):
        log = "FAILED tests/test_foo.py::test_bar - AssertionError\nAssertionError"
        result = _make_wf().diagnose_ci_failure_structured(log, ["pytest"])
        assert result["failure_type"] == "test_failure"
        assert "tests/test_foo.py::test_bar" in result["failing_tests"]

    def test_unknown_failure_type(self):
        result = _make_wf().diagnose_ci_failure_structured("some random output", ["build"])
        assert result["failure_type"] == "unknown"

    def test_result_has_summary(self):
        log = "FAILED tests/test_a.py::test_x\nAssertionError"
        result = _make_wf().diagnose_ci_failure_structured(log, ["pytest"])
        assert "summary" in result
        assert len(result["summary"]) > 0

    def test_log_excerpt_truncated(self):
        log = "A" * 5000
        result = _make_wf().diagnose_ci_failure_structured(log, [])
        assert len(result["log_excerpt"]) <= 2000

    def test_required_fields_present(self):
        result = _make_wf().diagnose_ci_failure_structured("", [])
        for key in ("failure_type", "failing_tests", "failed_jobs", "log_excerpt", "summary"):
            assert key in result


# ---------------------------------------------------------------------------
# Branch hygiene check
# ---------------------------------------------------------------------------

class TestBranchHygiene:
    """check_branch_hygiene detects stale branches."""

    @patch("subprocess.run")
    def test_fresh_branch_ok(self, mock_run):
        """Branch with recent commit → recommendation=ok."""
        mock_run.return_value = MagicMock(returncode=0, stdout=str(int(time.time())))
        report = _make_wf().check_branch_hygiene("rank-test-branch")
        assert report.is_stale is False
        assert report.recommendation == "ok"

    @patch("subprocess.run")
    def test_stale_branch_warn(self, mock_run):
        """Branch 20 days old → is_stale=True, recommendation=warn."""
        stale_ts = int(time.time()) - (20 * 86400)
        mock_run.return_value = MagicMock(returncode=0, stdout=str(stale_ts))
        report = _make_wf().check_branch_hygiene("rank-old-branch")
        assert report.is_stale is True
        assert report.recommendation == "warn"

    @patch("subprocess.run")
    def test_very_old_branch_delete(self, mock_run):
        """Branch 35 days old → recommendation=delete."""
        stale_ts = int(time.time()) - (35 * 86400)
        mock_run.return_value = MagicMock(returncode=0, stdout=str(stale_ts))
        report = _make_wf().check_branch_hygiene("rank-ancient-branch")
        assert report.recommendation == "delete"

    @patch("subprocess.run")
    def test_git_failure_returns_ok(self, mock_run):
        """If git log fails, returns safe default (not stale)."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        report = _make_wf().check_branch_hygiene("bad-branch")
        assert report.is_stale is False
        assert report.recommendation == "ok"

    @patch("subprocess.run")
    def test_report_fields(self, mock_run):
        """BranchHygieneReport has all expected fields."""
        mock_run.return_value = MagicMock(returncode=0, stdout=str(int(time.time())))
        report = _make_wf().check_branch_hygiene("test-branch")
        assert report.branch == "test-branch"
        assert isinstance(report.age_days, float)
        assert isinstance(report.last_commit_ts, float)
        assert report.recommendation in ("ok", "warn", "delete")


# ---------------------------------------------------------------------------
# PR review gate
# ---------------------------------------------------------------------------

class TestPrReviewGate:
    """pr_review_gate blocks merge until CI passes."""

    def test_green_ci_allows_merge(self):
        wf = _make_wf()
        with patch.object(wf, "wait_for_ci", return_value=CIStatus("green", [], "")):
            ok, reason = wf.pr_review_gate(42)
        assert ok is True
        assert reason == "green"

    def test_red_ci_blocks_merge(self):
        wf = _make_wf()
        with patch.object(wf, "wait_for_ci", return_value=CIStatus("red", ["pytest"], "")):
            ok, reason = wf.pr_review_gate(42)
        assert ok is False
        assert "ci_red" in reason

    def test_timeout_blocks_merge(self):
        wf = _make_wf()
        with patch.object(wf, "wait_for_ci", return_value=CIStatus("timeout", [], "")):
            ok, reason = wf.pr_review_gate(42)
        assert ok is False
        assert "timeout" in reason

    def test_bypassed_when_not_required(self):
        wf = _make_wf()
        ok, reason = wf.pr_review_gate(42, require_green_ci=False)
        assert ok is True
        assert reason == "bypassed"

    def test_merge_pr_after_ci_calls_merge_on_green(self):
        wf = _make_wf()
        with patch.object(wf, "wait_for_ci", return_value=CIStatus("green", [], "")), \
             patch.object(wf, "merge_pr", return_value=True) as mock_merge:
            ok, reason = wf.merge_pr_after_ci(42)
        assert ok is True
        mock_merge.assert_called_once_with(42)

    def test_merge_pr_after_ci_skips_merge_on_red(self):
        wf = _make_wf()
        with patch.object(wf, "wait_for_ci", return_value=CIStatus("red", ["build"], "")), \
             patch.object(wf, "merge_pr") as mock_merge:
            ok, reason = wf.merge_pr_after_ci(42)
        assert ok is False
        mock_merge.assert_not_called()
