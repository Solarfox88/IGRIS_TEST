"""
test_mbop_bugfixes.py

Regression tests for the 3 MBOP bugs fixed in watchdog session:
  1. Quality Gate vacuous PASS → now "warning" when pytest skipped
  2. Satisfaction Gate vacuous PASS → now "advisory" when no ACs extracted
  3. quality_scores.json gets written after phase 11

Plus a test verifying _parse_issue_number extracts from goal text (Bug 1 in supervisor).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from igris.core.mbop_runner import (
    MBOPIntakeResult,
    MBOPQualityGateResult,
    MBOPSatisfactionGateResult,
    mbop_phase9_quality_gate,
    mbop_phase10_satisfaction_gate,
    mbop_post_run,
)
from igris.core.self_repair_supervisor import _parse_issue_number


# ---------------------------------------------------------------------------
# Bug 1 — supervisor: _parse_issue_number used for MBOP
# ---------------------------------------------------------------------------

class TestParseIssueNumber:
    def test_explicit_number_used(self):
        assert _parse_issue_number(42, "some goal") == 42

    def test_zero_falls_back_to_goal(self):
        assert _parse_issue_number(0, "Implement GitHub issue #5: feat stuff") == 5

    def test_none_falls_back_to_goal(self):
        assert _parse_issue_number(None, "fix #99 bug in core") == 99

    def test_explicit_beats_goal(self):
        assert _parse_issue_number(7, "fix #99 bug") == 7

    def test_no_issue_returns_zero(self):
        assert _parse_issue_number(0, "do something without issue number") == 0

    def test_large_issue_number(self):
        assert _parse_issue_number(0, "Implements Epic #1078: decomposition quality") == 1078


# ---------------------------------------------------------------------------
# Bug 2 — mbop_runner: Quality Gate "warning" when pytest skipped
# ---------------------------------------------------------------------------

class TestQualityGateStatus:
    """Test the qg_status logic in mbop_post_run (phases 9)."""

    def _make_run(self, status="completed", repair_cycles=0):
        run = MagicMock()
        run.status = status
        run.failure_class = ""
        run.repair_cycles_used = repair_cycles
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []
        return run

    def _make_intake(self, issue=5, acs=None):
        intake = MBOPIntakeResult(issue_number=issue)
        intake.acceptance_criteria = acs or []
        intake.extraction_ok = bool(acs)
        return intake

    def test_quality_gate_warning_when_pytest_skipped(self, tmp_path):
        """When no test files in diff → pytest skipped → gate should be 'warning', not 'pass'."""
        run = self._make_run()
        intake = self._make_intake()

        # MBOP phase9: no modified files, no stubs → passed=True, pytest_ran=False
        events_written = []
        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=""), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="fix: something"), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 10,
                enforce_quality_gate=False,
                run_id="test-qg-warning",
            )

        qg_events = [e for e in events_written if "quality_gate" in str(e)]
        assert qg_events, "No quality gate event written"
        # Find the phase9 status in the event args
        phase9 = next((e for e in events_written if "mbop_phase9_quality_gate" in str(e)), None)
        assert phase9 is not None
        # status is the 4th positional arg: (project_root, run_id, issue, phase, STATUS, detail, extra)
        status_arg = phase9[4]
        assert status_arg == "warning", (
            f"Expected 'warning' when pytest skipped, got '{status_arg}'. "
            "Vacuous PASS must not be treated as real PASS."
        )

    def test_quality_gate_pass_when_pytest_ran_ok(self, tmp_path):
        """When pytest ran and passed → gate should be real 'pass'."""
        run = self._make_run()
        intake = self._make_intake()

        events_written = []

        # Fake quality gate result: pytest ran + passed
        fake_qg = MBOPQualityGateResult()
        fake_qg.passed = True
        fake_qg.pytest_ran = True
        fake_qg.pytest_passed = True
        fake_qg.evidence = "pytest passed"

        with patch("igris.core.mbop_runner._get_modified_files", return_value=["tests/test_x.py"]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=""), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="fix: ok"), \
             patch("igris.core.mbop_runner.mbop_phase9_quality_gate", return_value=fake_qg), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 5,
                run_id="test-qg-real-pass",
            )

        phase9 = next((e for e in events_written if "mbop_phase9_quality_gate" in str(e)), None)
        assert phase9 is not None
        assert phase9[4] == "pass", f"Expected 'pass' when pytest ran OK, got '{phase9[4]}'"

    def test_quality_gate_fail_when_stubs_found(self, tmp_path):
        """When stubs detected → gate should be 'fail'."""
        run = self._make_run()
        intake = self._make_intake()

        fake_qg = MBOPQualityGateResult()
        fake_qg.passed = False
        fake_qg.pytest_ran = False
        fake_qg.stub_patterns_found = ["# placeholder"]
        fake_qg.evidence = "stub found"

        events_written = []
        with patch("igris.core.mbop_runner._get_modified_files", return_value=["igris/core/x.py"]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=""), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value=""), \
             patch("igris.core.mbop_runner.mbop_phase9_quality_gate", return_value=fake_qg), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 5,
                run_id="test-qg-fail",
            )

        phase9 = next((e for e in events_written if "mbop_phase9_quality_gate" in str(e)), None)
        assert phase9 is not None
        assert phase9[4] == "fail", f"Expected 'fail' when stubs found, got '{phase9[4]}'"


# ---------------------------------------------------------------------------
# Bug 2b — Satisfaction Gate "advisory" when no ACs
# ---------------------------------------------------------------------------

class TestSatisfactionGateStatus:
    def test_advisory_when_no_acs(self, tmp_path):
        """When intake found no ACs → satisfaction gate must stay 'advisory'."""
        run = MagicMock()
        run.status = "completed"
        run.failure_class = ""
        run.repair_cycles_used = 0
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []

        # No ACs in intake
        intake = MBOPIntakeResult(issue_number=0)
        intake.acceptance_criteria = []
        intake.extraction_ok = False

        events_written = []
        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value="diff content"), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="feat: add stuff"), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 5,
                run_id="test-sg-no-acs",
            )

        phase10 = next((e for e in events_written if "mbop_phase10_satisfaction_gate" in str(e)), None)
        assert phase10 is not None
        assert phase10[4] == "advisory", (
            f"Expected 'advisory' when no ACs, got '{phase10[4]}'. "
            "Vacuous pass with no criteria must not be 'pass'."
        )

    def test_pass_when_acs_covered(self, tmp_path):
        """When ACs are extracted and keyword-matched in diff → 'pass'."""
        run = MagicMock()
        run.status = "completed"
        run.failure_class = ""
        run.repair_cycles_used = 0
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []

        intake = MBOPIntakeResult(issue_number=5)
        intake.acceptance_criteria = ["GitHubReadGateway class implemented with scope validation"]
        intake.extraction_ok = True

        # Diff contains relevant keywords
        diff = "GitHubReadGateway scope validation authorization check"

        events_written = []
        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=diff), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="feat: implement gateway"), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 5,
                run_id="test-sg-covered",
            )

        phase10 = next((e for e in events_written if "mbop_phase10_satisfaction_gate" in str(e)), None)
        assert phase10 is not None
        assert phase10[4] == "pass", (
            f"Expected 'pass' when ACs keyword-matched in diff, got '{phase10[4]}'"
        )

    def test_advisory_when_acs_not_covered(self, tmp_path):
        """When ACs extracted but keywords NOT in diff → 'advisory'."""
        run = MagicMock()
        run.status = "completed"
        run.failure_class = ""
        run.repair_cycles_used = 0
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []

        intake = MBOPIntakeResult(issue_number=5)
        intake.acceptance_criteria = ["endpoint returns correct JSON schema format"]
        intake.extraction_ok = True

        # Diff doesn't contain any AC keywords
        diff = "added some unrelated code here"

        events_written = []
        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=diff), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="fix: something"), \
             patch("igris.core.mbop_runner._persist_event",
                   side_effect=lambda *a, **kw: events_written.append(a)):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 5,
                run_id="test-sg-not-covered",
            )

        phase10 = next((e for e in events_written if "mbop_phase10_satisfaction_gate" in str(e)), None)
        assert phase10 is not None
        assert phase10[4] == "advisory"


# ---------------------------------------------------------------------------
# Bug 3 — quality_scores.json written after phase 11
# ---------------------------------------------------------------------------

class TestQualityScoresJson:
    def test_quality_scores_written_after_run(self, tmp_path):
        """quality_scores.json must be populated after mbop_post_run."""
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()

        run = MagicMock()
        run.status = "completed"
        run.failure_class = ""
        run.repair_cycles_used = 0
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []

        intake = MBOPIntakeResult(issue_number=5)
        intake.acceptance_criteria = ["Feature works correctly"]
        intake.extraction_ok = True

        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value="feature works correctly"), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value="feat: done"), \
             patch("igris.core.mbop_runner._persist_event"):  # suppress file writes for events
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 10,
                run_id="test-qs-written",
            )

        qs_path = igris_dir / "quality_scores.json"
        assert qs_path.exists(), "quality_scores.json was not created"
        records = json.loads(qs_path.read_text())
        assert isinstance(records, list)
        assert len(records) >= 1
        r = records[-1]
        assert r["run_id"] == "test-qs-written"
        assert r["issue_number"] == 5
        assert "quality_gate" in r
        assert "satisfaction_gate" in r
        assert "pytest_ran" in r
        assert "duration_seconds" in r

    def test_quality_scores_appends_across_runs(self, tmp_path):
        """Successive runs append to quality_scores.json."""
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()

        def run_once(run_id: str):
            run = MagicMock()
            run.status = "completed"
            run.failure_class = ""
            run.repair_cycles_used = 0
            run.completion_mode = ""
            run.degraded_reason = ""
            run.events = []
            intake = MBOPIntakeResult(issue_number=1)
            intake.acceptance_criteria = []
            intake.extraction_ok = False
            with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
                 patch("igris.core.mbop_runner._get_diff_text", return_value=""), \
                 patch("igris.core.mbop_runner._get_last_commit_message", return_value=""), \
                 patch("igris.core.mbop_runner._persist_event"):
                mbop_post_run(
                    run=run, intake=intake,
                    project_root=str(tmp_path),
                    run_start_ts=time.time() - 1,
                    run_id=run_id,
                )

        run_once("run-1")
        run_once("run-2")
        run_once("run-3")

        qs_path = igris_dir / "quality_scores.json"
        records = json.loads(qs_path.read_text())
        assert len(records) == 3
        ids = [r["run_id"] for r in records]
        assert ids == ["run-1", "run-2", "run-3"]

    def test_quality_scores_capped_at_500(self, tmp_path):
        """quality_scores.json is capped at 500 records."""
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()
        # Pre-populate with 498 records
        existing = [{"run_id": f"old-{i}", "ts_epoch": 0} for i in range(498)]
        (igris_dir / "quality_scores.json").write_text(json.dumps(existing))

        run = MagicMock()
        run.status = "completed"
        run.failure_class = ""
        run.repair_cycles_used = 0
        run.completion_mode = ""
        run.degraded_reason = ""
        run.events = []
        intake = MBOPIntakeResult(issue_number=0)
        intake.acceptance_criteria = []
        intake.extraction_ok = False

        with patch("igris.core.mbop_runner._get_modified_files", return_value=[]), \
             patch("igris.core.mbop_runner._get_diff_text", return_value=""), \
             patch("igris.core.mbop_runner._get_last_commit_message", return_value=""), \
             patch("igris.core.mbop_runner._persist_event"):
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 1,
                run_id="run-499",
            )
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 1,
                run_id="run-500",
            )
            mbop_post_run(
                run=run, intake=intake,
                project_root=str(tmp_path),
                run_start_ts=time.time() - 1,
                run_id="run-501-overflow",
            )

        records = json.loads((igris_dir / "quality_scores.json").read_text())
        assert len(records) <= 500
        # Latest should be preserved
        assert records[-1]["run_id"] == "run-501-overflow"
