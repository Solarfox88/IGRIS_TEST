from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from igris.core.agent_reasoning_loop import AgentReasoningLoop, LoopResult

logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    task_id: str
    goal: str
    max_steps: int = 20
    task_type: str = "code_reasoning"
    preferred_profile: Optional[str] = None
    initial_context: dict = field(default_factory=dict)
    # Epic #1075 — dependency graph support
    depends_on: List[str] = field(default_factory=list)  # task_ids this task waits for
    can_run_parallel: bool = True


@dataclass
class ParallelResult:
    task_id: str
    result: Optional[LoopResult]
    error: Optional[str] = None
    # Epic #1075 — result merge metadata
    merged_files: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def success(self) -> bool:
        return self.result is not None and self.result.status == "finished"


class FileLock:
    """Epic #1075 — File-based lock to prevent concurrent writes to the same path.

    Uses fcntl.flock (Linux/macOS). Acquired with a context manager:

        with FileLock("/path/to/file.py"):
            # safe to write here

    Times out after *timeout_seconds* if the lock cannot be acquired.
    """

    def __init__(self, path: str, timeout_seconds: float = 30.0) -> None:
        self._path = path
        self._lock_path = f"{path}.igris.lock"
        self._timeout = timeout_seconds
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Acquire exclusive lock, blocking until *timeout_seconds*."""
        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_WRONLY)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise TimeoutError(
                        f"Could not acquire file lock on {self._path!r} within {self._timeout}s"
                    )
                time.sleep(0.1)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception:
                pass
            finally:
                self._fd = None
                try:
                    os.unlink(self._lock_path)
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def build_dependency_order(tasks: List[ParallelTask]) -> List[List[ParallelTask]]:
    """Epic #1075 — Topological sort: group tasks into execution waves.

    Each wave contains tasks whose dependencies are all satisfied by
    previous waves. Tasks with no dependencies are in wave 0.
    Tasks that cannot be ordered (cycle) are placed last.

    Returns a list of waves (list of task groups).
    """
    task_map: Dict[str, ParallelTask] = {t.task_id: t for t in tasks}
    completed: Set[str] = set()
    remaining = list(tasks)
    waves: List[List[ParallelTask]] = []
    max_iterations = len(tasks) + 1

    for _ in range(max_iterations):
        if not remaining:
            break
        wave = [
            t for t in remaining
            if all(dep in completed for dep in t.depends_on)
        ]
        if not wave:
            # Cycle or unsatisfiable deps — add remaining as final wave
            waves.append(remaining)
            break
        waves.append(wave)
        completed.update(t.task_id for t in wave)
        remaining = [t for t in remaining if t.task_id not in completed]

    return waves


def merge_results(results: List[ParallelResult]) -> Dict[str, Any]:
    """Epic #1075 — Merge parallel results into a summary dict.

    Returns:
        total: int
        succeeded: int
        failed: int
        skipped: int
        failed_task_ids: List[str]
        succeeded_task_ids: List[str]
        merged_files: List[str]  # union of all files modified
        all_success: bool
    """
    succeeded = [r for r in results if r.success and not r.skipped]
    failed = [r for r in results if not r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]

    all_merged_files: List[str] = []
    seen_files: Set[str] = set()
    for r in succeeded:
        for f in r.merged_files:
            if f not in seen_files:
                all_merged_files.append(f)
                seen_files.add(f)

    return {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "succeeded_task_ids": [r.task_id for r in succeeded],
        "failed_task_ids": [r.task_id for r in failed],
        "skipped_task_ids": [r.task_id for r in skipped],
        "merged_files": all_merged_files,
        "all_success": len(failed) == 0 and len(results) > 0,
    }


class ParallelTaskRunner:
    """Runs multiple AgentReasoningLoop instances concurrently.

    Epic #1075 additions:
    - FileLock prevents concurrent writes to the same file
    - Dependency graph: tasks run in topological order via build_dependency_order()
    - merge_results() aggregates outputs into a summary
    """

    def __init__(self, project_root: str, max_concurrent: int = 3) -> None:
        self.project_root = project_root
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._completed_results: Dict[str, ParallelResult] = {}

    async def _run_one(self, task: ParallelTask) -> ParallelResult:
        async with self._semaphore:
            try:
                loop = AgentReasoningLoop(
                    project_root=self.project_root,
                    max_steps=task.max_steps,
                    task_type=task.task_type,
                    preferred_profile=task.preferred_profile,
                )
                result = await asyncio.to_thread(
                    loop.run,
                    goal=task.goal,
                    initial_context=task.initial_context,
                )
                pr = ParallelResult(task_id=task.task_id, result=result)
                # Collect modified files for merge
                if hasattr(result, "files_modified"):
                    pr.merged_files = list(result.files_modified or [])
                return pr
            except Exception as exc:
                logger.error("parallel task %s failed: %s", task.task_id, exc)
                return ParallelResult(task_id=task.task_id, result=None, error=str(exc))

    async def run(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """Run tasks respecting dependency order (Epic #1075)."""
        if not tasks:
            return []

        # Check if any task has dependencies — if not, run all in parallel
        has_deps = any(t.depends_on for t in tasks)
        if not has_deps:
            coros = [self._run_one(t) for t in tasks]
            return list(await asyncio.gather(*coros))

        # Run in waves (topological order)
        waves = build_dependency_order(tasks)
        all_results: List[ParallelResult] = []
        completed_ids: Set[str] = set()

        for wave in waves:
            wave_coros = [self._run_one(t) for t in wave]
            wave_results = list(await asyncio.gather(*wave_coros))
            all_results.extend(wave_results)
            for r in wave_results:
                self._completed_results[r.task_id] = r
                if r.success:
                    completed_ids.add(r.task_id)

        return all_results

    def run_sync(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        return asyncio.run(self.run(tasks))
