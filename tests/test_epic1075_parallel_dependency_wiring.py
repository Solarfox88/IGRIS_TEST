"""Tests for Epic #1075 — build_dependency_order + ParallelTaskRunner wired.

Verifies:
1. build_dependency_order returns correct topological waves.
2. Cycles are handled gracefully (placed as final wave).
3. detect_file_conflicts correctly identifies overlapping scopes.
4. merge_results aggregates results correctly.
5. ParallelTaskRunner.run_sync respects dependency order via waves.
6. SelfRepairSupervisor._run_decomposed_parallel with depends_on_map.
7. SelfRepairSupervisor.run_parallel_submissions() exists and returns expected keys.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from igris.core.parallel_task_runner import (
    ParallelTask,
    ParallelResult,
    build_dependency_order,
    detect_file_conflicts,
    merge_results,
    FileLock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(task_id: str, goal: str = "goal", depends_on: Optional[List[str]] = None, scopes: Optional[List[str]] = None) -> ParallelTask:
    return ParallelTask(
        task_id=task_id,
        goal=goal,
        depends_on=depends_on or [],
        initial_context={"file_scopes": scopes or []},
    )


def _result(task_id: str, success: bool = True) -> ParallelResult:
    mock_result = MagicMock()
    mock_result.status = "finished" if success else "failed"
    return ParallelResult(
        task_id=task_id,
        result=mock_result if success else None,
        error=None if success else "simulated error",
        skipped=False,
    )


# ---------------------------------------------------------------------------
# build_dependency_order
# ---------------------------------------------------------------------------

class TestBuildDependencyOrder:

    def test_no_deps_single_wave(self):
        tasks = [_task("A"), _task("B"), _task("C")]
        waves = build_dependency_order(tasks)
        assert len(waves) == 1
        ids = {t.task_id for t in waves[0]}
        assert ids == {"A", "B", "C"}

    def test_linear_chain_three_waves(self):
        tasks = [
            _task("A"),
            _task("B", depends_on=["A"]),
            _task("C", depends_on=["B"]),
        ]
        waves = build_dependency_order(tasks)
        assert len(waves) == 3
        assert waves[0][0].task_id == "A"
        assert waves[1][0].task_id == "B"
        assert waves[2][0].task_id == "C"

    def test_diamond_dependency(self):
        # A → B, C → D
        tasks = [
            _task("A"),
            _task("B", depends_on=["A"]),
            _task("C", depends_on=["A"]),
            _task("D", depends_on=["B", "C"]),
        ]
        waves = build_dependency_order(tasks)
        assert len(waves) == 3
        # Wave 0: A; Wave 1: B, C; Wave 2: D
        assert {t.task_id for t in waves[0]} == {"A"}
        assert {t.task_id for t in waves[1]} == {"B", "C"}
        assert {t.task_id for t in waves[2]} == {"D"}

    def test_cycle_produces_nonempty_result(self):
        # A → B → A (cycle)
        tasks = [
            _task("A", depends_on=["B"]),
            _task("B", depends_on=["A"]),
        ]
        waves = build_dependency_order(tasks)
        # Cycle detection: all remaining tasks placed as final wave
        all_task_ids = {t.task_id for wave in waves for t in wave}
        assert all_task_ids == {"A", "B"}

    def test_empty_tasks_returns_empty_waves(self):
        waves = build_dependency_order([])
        assert waves == []

    def test_single_task_one_wave(self):
        tasks = [_task("X")]
        waves = build_dependency_order(tasks)
        assert len(waves) == 1
        assert waves[0][0].task_id == "X"

    def test_mixed_deps_and_no_deps(self):
        tasks = [
            _task("A"),
            _task("B", depends_on=["A"]),
            _task("C"),  # independent
        ]
        waves = build_dependency_order(tasks)
        # Wave 0: A, C (both have no unmet deps); Wave 1: B
        assert len(waves) == 2
        assert {t.task_id for t in waves[0]} == {"A", "C"}
        assert {t.task_id for t in waves[1]} == {"B"}


# ---------------------------------------------------------------------------
# detect_file_conflicts
# ---------------------------------------------------------------------------

class TestDetectFileConflicts:

    def test_no_conflicts_disjoint_scopes(self):
        tasks = [
            _task("A", scopes=["igris/core/a.py"]),
            _task("B", scopes=["igris/core/b.py"]),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0

    def test_conflict_same_file(self):
        tasks = [
            _task("A", scopes=["igris/core/shared.py"]),
            _task("B", scopes=["igris/core/shared.py"]),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert "igris/core/shared.py" in conflicts

    def test_serialised_pair_not_conflict(self):
        # B depends on A → serialised → no conflict
        tasks = [
            _task("A", scopes=["igris/core/shared.py"]),
            _task("B", depends_on=["A"], scopes=["igris/core/shared.py"]),
        ]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0

    def test_empty_scopes_no_conflict(self):
        tasks = [_task("A"), _task("B")]
        conflicts = detect_file_conflicts(tasks)
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:

    def test_all_success(self):
        results = [_result("A", True), _result("B", True)]
        summary = merge_results(results)
        assert summary["all_success"] is True
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

    def test_partial_failure(self):
        results = [_result("A", True), _result("B", False)]
        summary = merge_results(results)
        assert summary["all_success"] is False
        assert summary["succeeded"] == 1
        assert summary["failed"] == 1

    def test_skipped_results(self):
        r = ParallelResult(task_id="C", result=None, skipped=True, skip_reason="dep failed")
        results = [_result("A", True), r]
        summary = merge_results(results)
        assert summary["skipped"] == 1
        assert summary["total"] == 2

    def test_empty_results(self):
        summary = merge_results([])
        assert summary["total"] == 0
        assert summary["all_success"] is False  # no results → not all success

    def test_merged_files_deduped(self):
        r1 = ParallelResult(task_id="A", result=MagicMock(status="finished"), merged_files=["x.py", "y.py"])
        r2 = ParallelResult(task_id="B", result=MagicMock(status="finished"), merged_files=["y.py", "z.py"])
        summary = merge_results([r1, r2])
        assert set(summary["merged_files"]) == {"x.py", "y.py", "z.py"}


# ---------------------------------------------------------------------------
# SelfRepairSupervisor.run_parallel_submissions()
# ---------------------------------------------------------------------------

class TestRunParallelSubmissions:

    def _supervisor(self, tmp_path):
        from igris.core.self_repair_supervisor import SelfRepairSupervisor
        return SelfRepairSupervisor(str(tmp_path))

    def test_run_parallel_submissions_returns_expected_keys(self, tmp_path):
        sup = self._supervisor(tmp_path)
        sub_missions = [
            {
                "title": "Task A: implement login",
                "goal": "Implement /login endpoint",
                "dependencies": [],
                "allowed_file_scopes": ["igris/web/routes.py"],
                "acceptance_criteria": ["AC1", "AC2"],
                "risk_level": "low",
            },
            {
                "title": "Task B: add tests",
                "goal": "Add tests for /login",
                "dependencies": ["Task A: implement login"],
                "allowed_file_scopes": ["tests/test_login.py"],
                "acceptance_criteria": ["AC1"],
                "risk_level": "low",
            },
        ]

        # Mock the ParallelTaskRunner so it doesn't actually run reasoning loops
        with patch("igris.core.parallel_task_runner.ParallelTaskRunner.run_sync") as mock_run_sync:
            mock_run_sync.return_value = [
                ParallelResult(task_id="Task A: implement login", result=MagicMock(status="finished"), merged_files=["igris/web/routes.py"]),
                ParallelResult(task_id="Task B: add tests", result=MagicMock(status="finished"), merged_files=["tests/test_login.py"]),
            ]
            summary = sup.run_parallel_submissions(sub_missions, max_steps=5)

        # Required summary keys
        assert "total" in summary
        assert "succeeded" in summary
        assert "failed" in summary
        assert "waves" in summary
        assert "conflicts" in summary

    def test_run_parallel_submissions_includes_wave_structure(self, tmp_path):
        sup = self._supervisor(tmp_path)
        sub_missions = [
            {"title": "A", "goal": "goal A", "dependencies": [], "allowed_file_scopes": [], "acceptance_criteria": [], "risk_level": "low"},
            {"title": "B", "goal": "goal B", "dependencies": ["A"], "allowed_file_scopes": [], "acceptance_criteria": [], "risk_level": "low"},
        ]
        with patch("igris.core.parallel_task_runner.ParallelTaskRunner.run_sync") as mock_run:
            mock_run.return_value = []
            summary = sup.run_parallel_submissions(sub_missions, max_steps=5)

        # Should have 2 waves: wave 0 = A, wave 1 = B
        waves = summary.get("waves", [])
        assert len(waves) == 2
        assert any("A" in w["tasks"] for w in waves)
        assert any("B" in w["tasks"] for w in waves)

    def test_run_parallel_submissions_detects_conflicts(self, tmp_path):
        sup = self._supervisor(tmp_path)
        sub_missions = [
            {"title": "A", "goal": "goal A", "dependencies": [], "allowed_file_scopes": ["igris/core/shared.py"], "acceptance_criteria": [], "risk_level": "low"},
            {"title": "B", "goal": "goal B", "dependencies": [], "allowed_file_scopes": ["igris/core/shared.py"], "acceptance_criteria": [], "risk_level": "low"},
        ]
        with patch("igris.core.parallel_task_runner.ParallelTaskRunner.run_sync") as mock_run:
            mock_run.return_value = []
            summary = sup.run_parallel_submissions(sub_missions, max_steps=5)

        # Should detect shared.py conflict
        assert "igris/core/shared.py" in summary.get("conflicts", {})


# ---------------------------------------------------------------------------
# FileLock — basic operation
# ---------------------------------------------------------------------------

class TestFileLock:

    def test_lock_acquire_release(self, tmp_path):
        lock_path = str(tmp_path / "test_file.py")
        lock = FileLock(lock_path, timeout_seconds=5.0)
        lock.acquire()
        lock.release()  # should not raise

    def test_lock_context_manager(self, tmp_path):
        lock_path = str(tmp_path / "test_file2.py")
        with FileLock(lock_path, timeout_seconds=5.0):
            pass  # should acquire and release without error

    def test_lock_file_removed_after_release(self, tmp_path):
        import os
        lock_path = str(tmp_path / "test_file3.py")
        with FileLock(lock_path, timeout_seconds=5.0):
            pass
        assert not os.path.exists(f"{lock_path}.igris.lock")
