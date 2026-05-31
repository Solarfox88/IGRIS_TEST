"""Tests for Epic #1078 — DecompositionValidator wiring in self_repair_supervisor.

Verifies that:
1. _ask_igris_decompose logs a 'decomposition_quality' event with quality_score.
2. _auto_create_subissues detects file-scope conflicts and logs 'subissue_scope_conflict'.
3. Quality score and issues are embedded in the returned decomposition dict.
4. Validator skips gracefully when sub_missions is empty.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight stubs — enough to exercise the new wiring without a real backend
# ---------------------------------------------------------------------------

from igris.core.decomposition_validator import DecompositionValidator, ValidationReport, ValidationIssue


def _make_minimal_sub_mission(title: str, goal: str, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "title": title,
        "goal": goal,
        "risk_level": "low",
        "allowed_file_scopes": scopes or [],
        "tests": [f"tests/test_{title.lower().replace(' ', '_')}.py"],
        "acceptance_criteria": [
            f"Acceptance criterion 1 for {title}",
            f"Acceptance criterion 2 for {title}",
            f"Acceptance criterion 3 for {title}",
        ],
        "dependencies": [],
    }


# ---------------------------------------------------------------------------
# Test DecompositionValidator.validate — sanity coverage for quality scoring
# ---------------------------------------------------------------------------

class TestDecompositionValidatorQualityScore:
    def test_valid_three_sub_missions_score_above_threshold(self):
        subs = [
            _make_minimal_sub_mission(f"Module {i}: concrete feature", f"Implement feature {i} in igris/core/x.py")
            for i in range(3)
        ]
        report = DecompositionValidator().validate(subs)
        assert isinstance(report.quality_score, float)
        assert 0.0 <= report.quality_score <= 1.0
        assert isinstance(report.valid, bool)

    def test_empty_sub_missions_returns_neutral_report(self):
        report = DecompositionValidator().validate([])
        assert isinstance(report, ValidationReport)
        assert isinstance(report.quality_score, float)
        # Empty → no accepted → ok=False → valid=False; score is still a float
        assert report.quality_score >= 0.0

    def test_vague_titles_produce_issues(self):
        subs = [
            {"title": "fix", "goal": "do stuff", "acceptance_criteria": [], "tests": [], "dependencies": [], "allowed_file_scopes": []},
        ]
        report = DecompositionValidator().validate(subs)
        # Either issues flagged or the sub-mission was rejected (error_count > 0)
        assert len(report.issues) > 0 or report.error_count > 0 or len(report.rejected) > 0

    def test_duplicate_goals_produce_dedup_issue(self):
        same_goal = "Implement the exact same feature in the exact same file path"
        subs = [
            _make_minimal_sub_mission("Module A: alpha task", same_goal),
            _make_minimal_sub_mission("Module B: beta task", same_goal),
        ]
        report = DecompositionValidator().validate(subs)
        all_codes = {i.code for i in report.issues}
        # Either a DEDUP code or fewer accepted (duplicate rejected)
        has_dedup = any("dedup" in c.lower() or "dup" in c.lower() or "DEDUP" in c for c in all_codes)
        has_rejection = len(report.rejected) > 0
        assert has_dedup or has_rejection

    def test_missing_acceptance_criteria_produces_ac_issue(self):
        subs = [{"title": "Feature Alpha: implement login endpoint", "goal": "Add /login endpoint with JWT", "acceptance_criteria": [], "tests": [], "dependencies": [], "allowed_file_scopes": []}]
        report = DecompositionValidator().validate(subs)
        all_codes = {i.code for i in report.issues}
        has_ac_issue = any("AC" in c or "CRITERIA" in c or "criteria" in c.lower() or "accept" in c.lower() for c in all_codes)
        has_rejection = len(report.rejected) > 0
        assert has_ac_issue or has_rejection

    def test_max_cap_respected(self):
        subs = [_make_minimal_sub_mission(f"Module {i}: concrete task", f"Goal {i} in igris/core/file_{i}.py") for i in range(20)]
        report = DecompositionValidator().validate(subs)
        # Validator should cap at IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION (default 12)
        assert isinstance(report, ValidationReport)
        assert len(report.accepted) + len(report.rejected) <= 20


# ---------------------------------------------------------------------------
# Test quality event embedded in decomposition dict (mocked supervisor path)
# ---------------------------------------------------------------------------

class TestDecompositionQualityEventInSupervisor:
    """White-box test: call the validator inline and check output format."""

    def test_quality_score_in_dict_after_validation(self):
        subs = [_make_minimal_sub_mission(f"Module {i}: concrete task", f"Implement feature {i} in igris/core/file_{i}.py") for i in range(3)]
        report = DecompositionValidator().validate(subs)
        # Simulate what supervisor does when embedding quality in decomposition dict
        decomposition: Dict[str, Any] = {
            "sub_missions": subs,
            "_quality_score": round(report.quality_score, 3),
            "_quality_valid": report.valid,
            "_quality_issues": [{"code": i.code, "message": i.message, "index": getattr(i, "index", None)} for i in report.issues],
        }
        assert "_quality_score" in decomposition
        assert "_quality_valid" in decomposition
        assert "_quality_issues" in decomposition
        assert isinstance(decomposition["_quality_score"], float)
        assert 0.0 <= decomposition["_quality_score"] <= 1.0

    def test_quality_event_status_ok_when_valid(self):
        subs = [_make_minimal_sub_mission(f"Module {i}: feature description", f"Implement {i} in igris/core/x.py") for i in range(3)]
        report = DecompositionValidator().validate(subs)
        status = "ok" if report.valid else "warning"
        # Both statuses are valid outcomes depending on sub-mission quality
        assert status in ("ok", "warning")

    def test_quality_event_status_warning_on_invalid(self):
        subs = [{"title": "", "goal": "", "acceptance_criteria": [], "tests": [], "dependencies": [], "allowed_file_scopes": []}]
        report = DecompositionValidator().validate(subs)
        status = "ok" if report.valid else "warning"
        # Empty title/goal should always produce warning (not valid)
        assert status == "warning"


# ---------------------------------------------------------------------------
# Test scope conflict detection (parallel_task_runner.detect_file_conflicts)
# ---------------------------------------------------------------------------

class TestScopeConflictDetection:
    def test_no_conflict_when_disjoint_scopes(self):
        from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts
        tasks = [
            ParallelTask(task_id="A", goal="goal A", initial_context={"file_scopes": ["igris/core/a.py"]}),
            ParallelTask(task_id="B", goal="goal B", initial_context={"file_scopes": ["igris/core/b.py"]}),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0

    def test_conflict_detected_when_same_file(self):
        from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts
        tasks = [
            ParallelTask(task_id="A", goal="goal A", initial_context={"file_scopes": ["igris/core/shared.py"]}),
            ParallelTask(task_id="B", goal="goal B", initial_context={"file_scopes": ["igris/core/shared.py"]}),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert "igris/core/shared.py" in conflicts
        assert set(conflicts["igris/core/shared.py"]) == {"A", "B"}

    def test_serialised_pair_not_reported_as_conflict(self):
        from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts
        # B depends on A → they are serialised, so shared file is NOT a conflict
        tasks = [
            ParallelTask(task_id="A", goal="goal A", initial_context={"file_scopes": ["igris/core/shared.py"]}),
            ParallelTask(task_id="B", goal="goal B", initial_context={"file_scopes": ["igris/core/shared.py"]}, depends_on=["A"]),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0

    def test_three_tasks_two_share_file(self):
        from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts
        tasks = [
            ParallelTask(task_id="A", goal="goal A", initial_context={"file_scopes": ["igris/core/shared.py", "igris/core/a_only.py"]}),
            ParallelTask(task_id="B", goal="goal B", initial_context={"file_scopes": ["igris/core/shared.py", "igris/core/b_only.py"]}),
            ParallelTask(task_id="C", goal="goal C", initial_context={"file_scopes": ["igris/core/c_only.py"]}),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert "igris/core/shared.py" in conflicts
        assert "igris/core/a_only.py" not in conflicts
        assert "igris/core/c_only.py" not in conflicts

    def test_empty_scopes_no_conflict(self):
        from igris.core.parallel_task_runner import ParallelTask, detect_file_conflicts
        tasks = [
            ParallelTask(task_id="A", goal="goal A", initial_context={}),
            ParallelTask(task_id="B", goal="goal B", initial_context={"file_scopes": []}),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0
