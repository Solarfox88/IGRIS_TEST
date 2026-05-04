"""
Simple task engine used by IGRIS_GPT.

The task engine maintains a list of tasks and selects the next one to execute
while avoiding repetitive task families using the anti‑loop heuristics.
"""

from __future__ import annotations

from typing import List, Optional

from igris.core import anti_loop
from igris.models.task import Task, TaskStatus


class TaskEngine:
    """In‑memory task engine for selecting and managing tasks."""

    def __init__(self) -> None:
        self._tasks: List[Task] = []
        self._next_id: int = 1

    @property
    def tasks(self) -> List[Task]:
        return self._tasks

    def add_task(self, description: str) -> Task:
        task = Task(id=self._next_id, description=description)
        self._next_id += 1
        self._tasks.append(task)
        return task

    def next_task(self) -> Optional[Task]:
        """Return the next task to execute or None if all are completed/blocked.

        This method uses the anti‑loop heuristics to avoid choosing a task from
        a saturated family.  If all pending tasks are from saturated families
        then None is returned.
        """
        pending = [t for t in self._tasks if t.status == TaskStatus.pending]
        if not pending:
            return None
        # Compute family counts from recent task descriptions
        counts = anti_loop.compute_family_counts([t.description for t in self._tasks])
        saturated = set(anti_loop.saturated_families(counts))
        for task in pending:
            family = anti_loop.classify_task_family(task.description)
            if family not in saturated:
                return task
        return None