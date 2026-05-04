"""
Anti‑loop heuristics for preventing repetitive or runaway behaviour.

The anti‑loop module classifies tasks into broad families and keeps counts of
recent occurrences.  When a family is saturated (i.e. repeated too many
times) the agent should switch strategies or seek human intervention.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Iterable, List, Optional


def classify_task_family(text: str) -> str:
    """Classify a task into a simple family based on keywords.

    This is a very naive implementation; future versions might use an LLM
    classifier or a more sophisticated parser.  Families are coarse grained
    categories such as "test", "edit", "search", "plan".
    """
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["test", "pytest", "unittest"]):
        return "testing"
    if any(keyword in lowered for keyword in ["fix", "edit", "modify", "refactor"]):
        return "editing"
    if any(keyword in lowered for keyword in ["search", "find", "grep"]):
        return "search"
    if any(keyword in lowered for keyword in ["write", "create", "generate"]):
        return "writing"
    return "other"


def compute_family_counts(tasks: Iterable[str], maxlen: int = 20) -> Counter:
    """Compute a counter of task families for the most recent tasks.

    :param tasks: An iterable of task descriptions (strings).
    :param maxlen: Only the last `maxlen` tasks are considered.
    :return: A Counter mapping family names to counts.
    """
    recent: Deque[str] = deque(tasks, maxlen=maxlen)
    counts: Counter = Counter(classify_task_family(t) for t in recent)
    return counts


def saturated_families(counts: Counter, threshold: int = 3) -> List[str]:
    """Return the list of families whose counts meet or exceed the threshold."""
    return [family for family, n in counts.items() if n >= threshold]


def should_force_strategy_shift(tasks: Iterable[str], threshold: int = 3) -> bool:
    """Determine whether the agent should shift strategies.

    Returns True if any task family is saturated beyond the given threshold.
    """
    counts = compute_family_counts(tasks)
    return bool(saturated_families(counts, threshold=threshold))