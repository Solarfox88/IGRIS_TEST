"""Tests for Epic #1075 — Parallel decomposition improvements.

Covers: FileLock mechanism, dependency graph ordering, result merge logic.
"""

import asyncio
import os
import threading
import time
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from igris.core.parallel_task_runner import (
    ParallelTask,
    ParallelResult,
    FileLock,
    build_dependency_order,
    merge_results,
)


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

class TestFileLock:
    """FileLock provides exclusive file locking."""

    def test_acquire_and_release(self, tmp_path):
        path = str(tmp_path / "test.py")
        lock = FileLock(path)
        lock.acquire()
        lock.release()
        # Lock file should be cleaned up
        assert not os.path.exists(f"{path}.igris.lock")

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "test.py")
        with FileLock(path):
            pass  # Should not raise
        assert not os.path.exists(f"{path}.igris.lock")

    def test_exclusive_blocks_second_acquire(self, tmp_path):
        """Second acquire on same path times out."""
        path = str(tmp_path / "test.py")
        lock1 = FileLock(path, timeout_seconds=0.3)
        lock1.acquire()
        try:
            lock2 = FileLock(path, timeout_seconds=0.2)
            with pytest.raises(TimeoutError):
                lock2.acquire()
        finally:
            lock1.release()

    def test_released_lock_reacquirable(self, tmp_path):
        path = str(tmp_path / "test.py")
        with FileLock(path, timeout_seconds=1.0):
            pass
        # Should succeed immediately
        with FileLock(path, timeout_seconds=1.0):
            pass

    def test_concurrent_threads_serialized(self, tmp_path):
        """Two threads fight for the lock; only one at a time proceeds."""
        path = str(tmp_path / "shared.py")
        results = []
        errors = []

        def worker(name):
            try:
                with FileLock(path, timeout_seconds=5.0):
                    results.append(f"{name}_start")
                    time.sleep(0.05)
                    results.append(f"{name}_end")
            except Exception as exc:
                errors.append(str(exc))

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Lock errors: {errors}"
        assert len(results) == 4
        # Verify non-interleaving: each thread's start/end are adjacent
        a_start = results.index("A_start")
        a_end = results.index("A_end")
        b_start = results.index("B_start")
        b_end = results.index("B_end")
        assert a_end == a_start + 1 or b_end == b_start + 1, \
            f"Lock was not exclusive: {results}"


# ---------------------------------------------------------------------------
# Dependency graph ordering
# ---------------------------------------------------------------------------

class TestDependencyOrder:
    """build_dependency_order returns topological waves."""

    def test_no_deps_single_wave(self):
        tasks = [
            ParallelTask("a", "goal a"),
            ParallelTask("b", "goal b"),
        ]
        waves = build_dependency_order(tasks)
        assert len(waves) == 1
        assert set(t.task_id for t in waves[0]) == {"a", "b"}

    def test_linear_chain(self):
        """a → b → c produces 3 waves."""
        tasks = [
            ParallelTask("a", "goal a"),
            ParallelTask("b", "goal b", depends_on=["a"]),
            ParallelTask("c", "goal c", depends_on=["b"]),
        ]
        waves = build_dependency_order(tasks)
        assert len(waves) == 3
        assert waves[0][0].task_id == "a"
        assert waves[1][0].task_id == "b"
        assert waves[2][0].task_id == "c"

    def test_parallel_fan_out(self):
        """a → (b, c) produces wave [a], wave [b, c]."""
        tasks = [
            ParallelTask("a", "goal a"),
            ParallelTask("b", "goal b", depends_on=["a"]),
            ParallelTask("c", "goal c", depends_on=["a"]),
        ]
        waves = build_dependency_order(tasks)
        assert len(waves) == 2
        assert waves[0][0].task_id == "a"
        assert set(t.task_id for t in waves[1]) == {"b", "c"}

    def test_diamond_deps(self):
        """a → (b, c) → d produces 3 waves."""
        tasks = [
            ParallelTask("a", "goal a"),
            ParallelTask("b", "goal b", depends_on=["a"]),
            ParallelTask("c", "goal c", depends_on=["a"]),
            ParallelTask("d", "goal d", depends_on=["b", "c"]),
        ]
        waves = build_dependency_order(tasks)
        assert waves[0][0].task_id == "a"
        assert {t.task_id for t in waves[1]} == {"b", "c"}
        assert waves[-1][0].task_id == "d"

    def test_cycle_does_not_crash(self):
        """Cyclic dependencies don't cause infinite loop."""
        tasks = [
            ParallelTask("a", "goal a", depends_on=["b"]),
            ParallelTask("b", "goal b", depends_on=["a"]),
        ]
        waves = build_dependency_order(tasks)
        # At least the tasks are returned, even if not ordered
        all_ids = {t.task_id for wave in waves for t in wave}
        assert all_ids == {"a", "b"}

    def test_empty_tasks(self):
        assert build_dependency_order([]) == []


# ---------------------------------------------------------------------------
# Result merge
# ---------------------------------------------------------------------------

class TestMergeResults:
    """merge_results aggregates parallel results into a summary."""

    def _result(self, task_id, success=True, files=None, skipped=False, skip_reason=""):
        mock_loop = MagicMock()
        mock_loop.status = "finished" if success else "failed"
        pr = ParallelResult(
            task_id=task_id,
            result=mock_loop if success else None,
            merged_files=files or [],
            skipped=skipped,
            skip_reason=skip_reason,
        )
        return pr

    def test_all_success(self):
        results = [self._result("a"), self._result("b")]
        summary = merge_results(results)
        assert summary["all_success"] is True
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

    def test_one_failure(self):
        results = [self._result("a"), self._result("b", success=False)]
        summary = merge_results(results)
        assert summary["all_success"] is False
        assert summary["failed"] == 1
        assert "b" in summary["failed_task_ids"]

    def test_skipped_counted(self):
        results = [
            self._result("a"),
            self._result("b", skipped=True, success=False),
        ]
        summary = merge_results(results)
        assert summary["skipped"] == 1
        assert "b" in summary["skipped_task_ids"]

    def test_merged_files_union(self):
        results = [
            self._result("a", files=["igris/core/foo.py"]),
            self._result("b", files=["igris/core/bar.py", "igris/core/foo.py"]),
        ]
        summary = merge_results(results)
        assert "igris/core/foo.py" in summary["merged_files"]
        assert "igris/core/bar.py" in summary["merged_files"]
        assert summary["merged_files"].count("igris/core/foo.py") == 1  # no dups

    def test_total_count(self):
        results = [self._result("a"), self._result("b"), self._result("c", success=False)]
        summary = merge_results(results)
        assert summary["total"] == 3

    def test_empty_results(self):
        summary = merge_results([])
        assert summary["total"] == 0
        assert summary["all_success"] is False

    def test_required_keys(self):
        results = [self._result("a")]
        summary = merge_results(results)
        for key in ("total", "succeeded", "failed", "skipped", "succeeded_task_ids",
                    "failed_task_ids", "merged_files", "all_success"):
            assert key in summary
