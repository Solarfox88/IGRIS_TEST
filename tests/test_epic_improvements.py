"""Tests for quality improvements to all 8 perfezionamento epics.

These tests validate the deeper functionality added on top of the initial
epic implementations — the parts that were shallow or missing.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Epic #1074 — classify_failure_from_output (standalone, testable)
# ---------------------------------------------------------------------------

from igris.core.self_repair_supervisor import (
    classify_failure_from_output,
    classify_failure_severity,
)


class TestClassifyFailureFromOutput:
    def test_timeout_wins(self):
        assert classify_failure_from_output("", "", 1, timed_out=True) == "test_runner_timeout"

    def test_syntax_error(self):
        out = "SyntaxError: invalid syntax at line 5"
        assert classify_failure_from_output(out, "", 1) == "syntax_error"

    def test_indentation_error(self):
        out = "IndentationError: unexpected indent"
        assert classify_failure_from_output(out, "", 1) == "syntax_error"

    def test_import_error(self):
        out = "ModuleNotFoundError: No module named 'foo'"
        assert classify_failure_from_output(out, "", 1) == "infrastructure_bug"

    def test_pytest_failure(self):
        out = "FAILED tests/test_foo.py::TestBar::test_baz"
        assert classify_failure_from_output(out, "", 1) == "pytest_failure"

    def test_assertion_error(self):
        out = "AssertionError: expected True"
        assert classify_failure_from_output(out, "", 1) == "pytest_failure"

    def test_nonzero_generic(self):
        assert classify_failure_from_output("some other error", "", 1) == "reasoning_loop_blocked"

    def test_success_returns_empty(self):
        assert classify_failure_from_output("all good", "", 0) == ""

    def test_timeout_overrides_syntax(self):
        # timed_out wins even if output contains SyntaxError
        out = "SyntaxError: ..."
        assert classify_failure_from_output(out, "", 1, timed_out=True) == "test_runner_timeout"


class TestClassifyFailureSeverity:
    def test_critical_classes(self):
        for fc in ("syntax_error", "infrastructure_bug", "invalid_bootstrap"):
            assert classify_failure_severity(fc) == "critical"

    def test_high_classes(self):
        for fc in ("pytest_failure", "wrong_file_edit", "semantic_incomplete"):
            assert classify_failure_severity(fc) == "high"

    def test_medium_classes(self):
        for fc in ("missing_tests", "missing_ui_visibility", "reasoning_loop_blocked"):
            assert classify_failure_severity(fc) == "medium"

    def test_unknown_fallback(self):
        assert classify_failure_severity("something_random") == "unknown"

    def test_empty_string(self):
        assert classify_failure_severity("") == "unknown"


# ---------------------------------------------------------------------------
# Epic #1072 — Precheck/postcheck hooks + rollback suggestions
# ---------------------------------------------------------------------------

from igris.core.command_risk_engine import CommandRiskEngine


class TestCommandRiskEngineHooks:
    def test_precheck_blocks_command(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        engine.register_precheck(lambda cmd: "no sudo allowed" if "sudo" in cmd else None)
        event, _ = engine.evaluate_command("sudo rm -rf /")
        assert event.decision == "blocked"
        assert "precheck" in event.reason
        assert "no sudo allowed" in event.reason

    def test_precheck_allows_safe_command(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        engine.register_precheck(lambda cmd: "no sudo" if "sudo" in cmd else None)
        event, _ = engine.evaluate_command("ls -la")
        # ls should not be blocked by this precheck
        assert event.decision != "blocked" or "precheck" not in event.reason

    def test_postcheck_vetoes_allowed_or_approval_command(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        engine.register_postcheck(lambda cmd, evt: "no writes in tests" if "write" in cmd else None)
        # write_file gets 'needs_approval' — postcheck should still veto it
        event, _ = engine.evaluate_command("write_file output.txt")
        assert event.decision == "blocked"
        assert "postcheck" in event.reason

    def test_postcheck_not_called_on_blocked(self):
        """Postcheck should not override a precheck block."""
        engine = CommandRiskEngine(use_llm_reviewer=False)
        engine.register_precheck(lambda cmd: "precheck block")
        called = []
        engine.register_postcheck(lambda cmd, evt: called.append(cmd) or None)
        engine.evaluate_command("ls")
        assert len(called) == 0  # postcheck not called when precheck blocked

    def test_multiple_prechecks_first_wins(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        engine.register_precheck(lambda cmd: "first block" if True else None)
        engine.register_precheck(lambda cmd: "second block")
        event, _ = engine.evaluate_command("ls")
        assert event.reason == "precheck: first block"

    def test_rollback_suggestion_rm(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        event, _ = engine.evaluate_command("rm -rf /tmp/test")
        event.final_risk = "critical"
        suggestion = engine.get_rollback_suggestion("rm -rf /tmp/test", event)
        assert "git checkout" in suggestion or "backup" in suggestion.lower()

    def test_rollback_suggestion_drop_table(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        event = MagicMock()
        event.final_risk = "critical"
        suggestion = engine.get_rollback_suggestion("DROP TABLE users", event)
        assert "snapshot" in suggestion.lower() or "backup" in suggestion.lower()

    def test_rollback_suggestion_low_risk(self):
        engine = CommandRiskEngine(use_llm_reviewer=False)
        event = MagicMock()
        event.final_risk = "low"
        suggestion = engine.get_rollback_suggestion("ls -la", event)
        assert suggestion == ""  # no rollback needed for low risk


# ---------------------------------------------------------------------------
# Epic #1075 — detect_file_conflicts
# ---------------------------------------------------------------------------

from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts


class TestDetectFileConflicts:
    def test_no_conflict_disjoint_files(self):
        t1 = ParallelTask("t1", "goal1", initial_context={"file_scopes": ["src/a.py"]})
        t2 = ParallelTask("t2", "goal2", initial_context={"file_scopes": ["src/b.py"]})
        assert detect_file_conflicts([t1, t2]) == {}

    def test_conflict_same_file(self):
        t1 = ParallelTask("t1", "goal1", initial_context={"file_scopes": ["src/shared.py"]})
        t2 = ParallelTask("t2", "goal2", initial_context={"file_scopes": ["src/shared.py"]})
        conflicts = detect_file_conflicts([t1, t2])
        assert "src/shared.py" in conflicts
        assert set(conflicts["src/shared.py"]) == {"t1", "t2"}

    def test_no_conflict_serialised_via_depends_on(self):
        # t2 depends on t1 → they run sequentially, no real conflict
        t1 = ParallelTask("t1", "goal1", initial_context={"file_scopes": ["src/shared.py"]})
        t2 = ParallelTask("t2", "goal2", depends_on=["t1"],
                          initial_context={"file_scopes": ["src/shared.py"]})
        conflicts = detect_file_conflicts([t1, t2])
        assert "src/shared.py" not in conflicts

    def test_no_file_scopes_no_conflict(self):
        t1 = ParallelTask("t1", "goal1")
        t2 = ParallelTask("t2", "goal2")
        assert detect_file_conflicts([t1, t2]) == {}

    def test_three_tasks_two_conflict(self):
        t1 = ParallelTask("t1", "g1", initial_context={"file_scopes": ["src/shared.py"]})
        t2 = ParallelTask("t2", "g2", initial_context={"file_scopes": ["src/shared.py"]})
        t3 = ParallelTask("t3", "g3", initial_context={"file_scopes": ["src/other.py"]})
        conflicts = detect_file_conflicts([t1, t2, t3])
        assert "src/shared.py" in conflicts
        assert "src/other.py" not in conflicts


# ---------------------------------------------------------------------------
# Epic #1071 — suggest_ci_repair_goal + delete_merged_branch
# ---------------------------------------------------------------------------

from igris.core.delivery_workflow import DeliveryWorkflow


class TestDeliveryWorkflowImprovements:
    def _make_wf(self, tmp_path):
        return DeliveryWorkflow(str(tmp_path))

    def test_suggest_ci_repair_goal_test_failure(self, tmp_path):
        wf = self._make_wf(tmp_path)
        diagnosis = {
            "failure_type": "test_failure",
            "failing_tests": ["tests/test_foo.py::test_bar", "tests/test_baz.py::test_x"],
            "failed_jobs": ["test"],
        }
        goal = wf.suggest_ci_repair_goal(diagnosis, "original goal here")
        assert "tests/test_foo.py::test_bar" in goal
        assert "CI repair" in goal
        assert "Do NOT change any test file" in goal

    def test_suggest_ci_repair_goal_lint(self, tmp_path):
        wf = self._make_wf(tmp_path)
        goal = wf.suggest_ci_repair_goal(
            {"failure_type": "lint_error", "failing_tests": [], "failed_jobs": ["lint"]},
            "fix the code"
        )
        assert "lint" in goal.lower() or "ruff" in goal.lower()

    def test_suggest_ci_repair_goal_import_error(self, tmp_path):
        wf = self._make_wf(tmp_path)
        goal = wf.suggest_ci_repair_goal(
            {"failure_type": "import_error", "failing_tests": [], "failed_jobs": ["ci"],
             "summary": "ImportError: no module"},
            "original"
        )
        assert "import" in goal.lower() or "ModuleNotFoundError" in goal

    def test_suggest_ci_repair_goal_includes_original_context(self, tmp_path):
        wf = self._make_wf(tmp_path)
        goal = wf.suggest_ci_repair_goal(
            {"failure_type": "test_failure", "failing_tests": ["tests/t.py::f"],
             "failed_jobs": []},
            "implement the feature X for user Y"
        )
        assert "implement the feature X" in goal

    @patch("subprocess.run")
    def test_delete_merged_branch_remote(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        wf = self._make_wf(tmp_path)
        result = wf.delete_merged_branch("feature/old-branch", remote=True)
        assert result is True
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("push" in c and "delete" in c for c in calls)

    @patch("subprocess.run")
    def test_delete_merged_branch_handles_not_found(self, mock_run, tmp_path):
        # Remote ref doesn't exist — should not count as failure
        mock_run.return_value = MagicMock(
            returncode=1, stderr="remote ref does not exist", stdout=""
        )
        wf = self._make_wf(tmp_path)
        result = wf.delete_merged_branch("feature/gone", remote=True)
        assert result is True


# ---------------------------------------------------------------------------
# Epic #1073 — _tree_write now logs on failure (not silent)
# ---------------------------------------------------------------------------

import logging


class TestMemoryGraphTreeWriteLogging:
    def test_tree_write_logs_warning_on_failure(self, caplog, tmp_path):
        """_tree_write must log a warning when ContentStore raises."""
        from igris.core.memory_graph import MemoryGraph

        mg = MemoryGraph(project_root=str(tmp_path))

        # Force _tree_write to fail by patching ContentStore import
        with patch("igris.core.memory_content_store.ContentStore") as mock_cs:
            mock_cs.side_effect = RuntimeError("disk full")
            with caplog.at_level(logging.WARNING, logger="igris.memory.graph"):
                # add_node triggers _tree_write
                mg.add_node("lesson", {"msg": "hello"}, confidence=0.5)

        # Should have logged the warning
        assert any(
            "tree_write" in r.message.lower() or "ContentStore" in r.message
            for r in caplog.records
        ), f"Expected tree_write warning, got: {[r.message for r in caplog.records]}"
