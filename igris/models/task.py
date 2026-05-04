"""
Data models for tasks executed by the agent.

Tasks represent individual unit of work that the agent can perform, such as
running tests, editing a file or searching for information.  The task engine
selects among pending tasks based on anti‑loop heuristics and priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    blocked = "blocked"


@dataclass
class Task:
    """Representation of a task scheduled or performed by the agent."""

    id: int
    description: str
    status: TaskStatus = TaskStatus.pending
    result: Optional[str] = None