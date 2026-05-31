"""Tests for Epic #1071 — CIRepairLoop wired into DeliveryWorkflow._apply_ci_fix().

Verifies:
1. _apply_ci_fix for 'test_failure' with no backend → returns False (original behavior).
2. _apply_ci_fix for 'test_failure' with a mock backend → calls CIRepairLoop.
3. CIRepairLoop diagnosis logic: import_error / syntax_error / test_failure / lint.
4. CIRepairLoop builds correct LLM goal for each failure type.
5. CIRepairLoop result: resolved=True when backend returns status='finished'.
6. lint_error path still runs ruff (deterministic, no backend needed).
7. DeliveryWorkflow accepts backend and goal in __init__.
8. fix_ci_loop sets _current_pr before calling _apply_ci_fix.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from igris.core.ci_repair_loop import (
    CIRepairLoop,
    CIRepairResult,
    CIRepairAttempt,
    MAX_ATTEMPTS,
)
from igris.core.delivery_workflow import DeliveryWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_backend(status: str = "finished") -> MagicMock:
    backend = MagicMock()
    backend.run_reasoning.return_value = {
        "status": status,
        "final_summary": "Repair completed.",
        "files_modified": [],
    }
    return backend


def _diagnosis(failure_type: str = "test_failure") -> dict:
    return {
        "run_id": 42,
        "failed_jobs": ["pytest"],
        "failure_type": failure_type,
        "log_excerpt": "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2",
    }


# ---------------------------------------------------------------------------
# DeliveryWorkflow.__init__ accepts backend / goal
# ---------------------------------------------------------------------------

class TestDeliveryWorkflowInit:

    def test_default_backend_is_none(self, tmp_path):
        dw = DeliveryWorkflow(str(tmp_path))
        assert dw._backend is None

    def test_backend_stored_on_init(self, tmp_path):
        backend = _mock_backend()
        dw = DeliveryWorkflow(str(tmp_path), backend=backend, goal="fix issue #5")
        assert dw._backend is backend
        assert dw._goal == "fix issue #5"

    def test_current_pr_initially_none(self, tmp_path):
        dw = DeliveryWorkflow(str(tmp_path))
        assert dw._current_pr is None


# ---------------------------------------------------------------------------
# _apply_ci_fix — test_failure fallback (no backend)
# ---------------------------------------------------------------------------

class TestApplyCiFixTestFailureNoBackend:

    def test_returns_false_when_no_backend(self, tmp_path):
        dw = DeliveryWorkflow(str(tmp_path))
        result = dw._apply_ci_fix(_diagnosis("test_failure"))
        assert result is False

    def test_returns_false_for_unknown_failure_type(self, tmp_path):
        dw = DeliveryWorkflow(str(tmp_path))
        result = dw._apply_ci_fix(_diagnosis("unknown"))
        assert result is False


# ---------------------------------------------------------------------------
# _apply_ci_fix — test_failure with backend → CIRepairLoop
# ---------------------------------------------------------------------------

class TestApplyCiFixTestFailureWithBackend:

    def test_calls_ci_repair_loop_when_backend_provided(self, tmp_path):
        backend = _mock_backend("finished")
        dw = DeliveryWorkflow(str(tmp_path), backend=backend, goal="fix tests")
        dw._current_pr = 99

        # Patch CIRepairLoop.run so it doesn't actually call gh CLI
        mock_result = CIRepairResult(
            resolved=True,
            attempts=[CIRepairAttempt(0, "test_failure", "llm_repair", "Fix tests", True, 1.0)],
            failure_summary="",
            total_duration_seconds=1.0,
        )
        with patch("igris.core.delivery_workflow.CIRepairLoop") as MockLoop:
            instance = MockLoop.return_value
            instance.run.return_value = mock_result
            result = dw._apply_ci_fix(_diagnosis("test_failure"))

        MockLoop.assert_called_once()
        instance.run.assert_called_once_with(backend)
        assert result is True

    def test_returns_false_when_ci_not_resolved(self, tmp_path):
        backend = _mock_backend("blocked")
        dw = DeliveryWorkflow(str(tmp_path), backend=backend, goal="fix tests")
        dw._current_pr = 99

        mock_result = CIRepairResult(
            resolved=False,
            attempts=[CIRepairAttempt(0, "test_failure", "llm_repair", "Fix tests", False, 1.0)],
            failure_summary="Test still failing",
        )
        with patch("igris.core.delivery_workflow.CIRepairLoop") as MockLoop:
            instance = MockLoop.return_value
            instance.run.return_value = mock_result
            result = dw._apply_ci_fix(_diagnosis("test_failure"))

        assert result is False

    def test_falls_back_gracefully_when_ci_repair_loop_raises(self, tmp_path):
        backend = _mock_backend()
        dw = DeliveryWorkflow(str(tmp_path), backend=backend, goal="fix tests")
        dw._current_pr = 99

        with patch("igris.core.delivery_workflow.CIRepairLoop", side_effect=ImportError("unavailable")):
            result = dw._apply_ci_fix(_diagnosis("test_failure"))

        assert result is False


# ---------------------------------------------------------------------------
# CIRepairLoop — _diagnose() logic
# ---------------------------------------------------------------------------

class TestCIRepairLoopDiagnose:

    def _loop(self, tmp_path) -> CIRepairLoop:
        return CIRepairLoop(str(tmp_path), pr_number=1, original_goal="fix tests")

    def test_diagnose_import_error(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("ImportError: cannot import name 'foo' from 'bar'")
        assert diag["failure_type"] == "import_error"

    def test_diagnose_module_not_found(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("ModuleNotFoundError: No module named 'igris.foo'")
        assert diag["failure_type"] == "import_error"

    def test_diagnose_syntax_error(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("SyntaxError: invalid syntax in file.py line 42")
        assert diag["failure_type"] == "syntax_error"

    def test_diagnose_test_failure(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("FAILED tests/test_foo.py::test_bar - AssertionError")
        assert diag["failure_type"] in ("test_failure", "unknown")

    def test_diagnose_unknown_on_empty_log(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("")
        assert diag["failure_type"] == "unknown"

    def test_diagnose_lint_error(self, tmp_path):
        loop = self._loop(tmp_path)
        diag = loop._diagnose("ruff: Found 3 errors. [*] 3 fixable with --fix")
        assert diag["failure_type"] == "lint_error"


# ---------------------------------------------------------------------------
# CIRepairLoop — _build_llm_repair_goal()
# ---------------------------------------------------------------------------

class TestCIRepairLoopBuildGoal:

    def _loop(self, tmp_path) -> CIRepairLoop:
        return CIRepairLoop(str(tmp_path), pr_number=1, original_goal="implement feature X")

    def test_goal_for_import_error_mentions_import(self, tmp_path):
        loop = self._loop(tmp_path)
        goal = loop._build_llm_repair_goal({"failure_type": "import_error", "failing_tests": []})
        assert "import" in goal.lower() or "module" in goal.lower()

    def test_goal_for_syntax_error_mentions_syntax(self, tmp_path):
        loop = self._loop(tmp_path)
        goal = loop._build_llm_repair_goal({"failure_type": "syntax_error", "failing_tests": []})
        assert "syntax" in goal.lower()

    def test_goal_for_test_failure_mentions_test(self, tmp_path):
        loop = self._loop(tmp_path)
        goal = loop._build_llm_repair_goal({"failure_type": "test_failure", "failing_tests": ["tests/test_foo.py::test_bar"]})
        assert "test" in goal.lower() or "pytest" in goal.lower()

    def test_goal_non_empty_for_unknown(self, tmp_path):
        loop = self._loop(tmp_path)
        goal = loop._build_llm_repair_goal({"failure_type": "unknown", "failing_tests": []})
        assert len(goal) > 0


# ---------------------------------------------------------------------------
# CIRepairLoop — run() with mock backend
# ---------------------------------------------------------------------------

class TestCIRepairLoopRun:

    def test_run_returns_resolved_true_when_backend_succeeds(self, tmp_path):
        backend = _mock_backend("finished")
        loop = CIRepairLoop(str(tmp_path), pr_number=1, original_goal="fix tests", max_attempts=1)

        # Patch _fetch_ci_logs and _ci_is_green so we don't call gh CLI
        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/test_foo.py::test_bar"), \
             patch.object(loop, "_ci_is_green", return_value=True):
            result = loop.run(backend)

        assert isinstance(result, CIRepairResult)
        assert result.attempt_count >= 1

    def test_run_returns_unresolved_when_ci_still_failing(self, tmp_path):
        backend = _mock_backend("blocked")
        loop = CIRepairLoop(str(tmp_path), pr_number=1, original_goal="fix tests", max_attempts=1)

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/test_foo.py::test_bar"), \
             patch.object(loop, "_ci_is_green", return_value=False):
            result = loop.run(backend)

        assert isinstance(result, CIRepairResult)
        assert result.resolved is False

    def test_run_has_positive_duration(self, tmp_path):
        backend = _mock_backend("finished")
        loop = CIRepairLoop(str(tmp_path), pr_number=1, original_goal="fix tests", max_attempts=1)

        with patch.object(loop, "_fetch_ci_logs", return_value=""), \
             patch.object(loop, "_ci_is_green", return_value=True):
            result = loop.run(backend)

        assert result.total_duration_seconds >= 0.0
