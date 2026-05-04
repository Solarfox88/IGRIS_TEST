"""Explainable task selection for IGRIS_GPT.

Wraps task_selection.select_next_task with detailed explanations
of why each candidate was selected, rejected, or skipped.

Inspired by IGRIS_DECO task selector explanations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core import anti_loop, semantic_dedup
from igris.core.decision_memory import (
    get_blocked_families_from_memory,
    get_recent_decisions,
    get_recent_failures,
    get_saturated_families,
)
from igris.core.safety import redact_secrets
from igris.core.task_selection import SelectionResult, select_next_task
from igris.models.task import Task, TaskStatus


# ---------------------------------------------------------------------------
# Explanation models
# ---------------------------------------------------------------------------

@dataclass
class CandidateExplanation:
    """Explanation for a single task candidate."""
    task_id: int
    title: str
    family: str
    priority: int
    risk: str
    status: str
    selected: bool = False
    score: float = 0.0
    why: str = ""
    rejected_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": redact_secrets(self.title),
            "family": self.family,
            "priority": self.priority,
            "risk": self.risk,
            "status": self.status,
            "selected": self.selected,
            "score": self.score,
            "why": self.why,
            "rejected_reasons": self.rejected_reasons,
        }


@dataclass
class SelectionExplanation:
    """Full explanation of the task selection process."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    selected: Optional[Dict[str, Any]] = None
    candidates: List[CandidateExplanation] = field(default_factory=list)
    saturated_families: List[str] = field(default_factory=list)
    blocked_families: List[str] = field(default_factory=list)
    recent_failure_count: int = 0
    recent_decision_count: int = 0
    selection_source: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "selected": self.selected,
            "candidates": [c.to_dict() for c in self.candidates],
            "saturated_families": self.saturated_families,
            "blocked_families": self.blocked_families,
            "recent_failure_count": self.recent_failure_count,
            "recent_decision_count": self.recent_decision_count,
            "selection_source": self.selection_source,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_task(
    task: Task,
    saturated: set,
    blocked_families: List[str],
    history: List[str],
    failure_families: Dict[str, int],
) -> tuple:
    """Score a task and return (score, reasons_if_rejected)."""
    score = 0.0
    reasons: List[str] = []
    family = task.family or anti_loop.classify_task_family(task.description)

    # Base priority score
    score += task.priority * 10.0

    # Risk penalty
    if task.risk == "high":
        score -= 50.0
        reasons.append("high risk (-50)")
    elif task.risk == "medium":
        score -= 10.0

    # Blocked family
    if family in blocked_families:
        score -= 100.0
        reasons.append(f"family '{family}' blocked")

    # Saturated family
    if family in saturated:
        score -= 80.0
        reasons.append(f"family '{family}' saturated")

    # Failure history penalty
    fail_count = failure_families.get(family, 0)
    if fail_count >= 3:
        score -= 30.0
        reasons.append(f"family '{family}' has {fail_count} recent failures")
    elif fail_count >= 1:
        score -= fail_count * 5.0

    # Semantic duplicate penalty
    if semantic_dedup.is_semantic_duplicate(task.description, history):
        score -= 60.0
        reasons.append("semantic duplicate")

    # Status penalty
    if task.status == TaskStatus.blocked:
        score -= 200.0
        reasons.append("task is blocked")
    elif task.status == TaskStatus.completed:
        score -= 200.0
        reasons.append("task is completed")
    elif task.status == TaskStatus.running:
        score -= 200.0
        reasons.append("task already running")

    return score, reasons


# ---------------------------------------------------------------------------
# Explain selection
# ---------------------------------------------------------------------------

def explain_task_selection(
    candidate_tasks: List[Task],
    advisory_next_task_id: Optional[int] = None,
    history: Optional[List[str]] = None,
    blocked_families: Optional[List[str]] = None,
    project_root: Optional[str] = None,
) -> SelectionExplanation:
    """Run task selection with full explanations."""
    history = history or []
    blocked_families = list(blocked_families or [])
    memory_blocked = get_blocked_families_from_memory(project_root)
    for fam in memory_blocked:
        if fam not in blocked_families:
            blocked_families.append(fam)

    # Get context
    saturated_fams = get_saturated_families(project_root=project_root)
    counts = anti_loop.compute_family_counts(history)
    saturated_from_history = set(anti_loop.saturated_families(counts))
    all_saturated = list(set(saturated_fams) | saturated_from_history)

    failures = get_recent_failures(limit=30, project_root=project_root)
    decisions = get_recent_decisions(limit=30, project_root=project_root)

    failure_families: Dict[str, int] = {}
    for f in failures:
        fam = f.get("family", "other")
        failure_families[fam] = failure_families.get(fam, 0) + 1

    # Run actual selection
    result = select_next_task(
        candidate_tasks,
        advisory_next_task_id=advisory_next_task_id,
        history=history,
        blocked_families=blocked_families,
        project_root=project_root,
    )

    # Build candidate explanations
    candidates: List[CandidateExplanation] = []
    for task in candidate_tasks:
        family = task.family or anti_loop.classify_task_family(task.description)
        score, reasons = _score_task(
            task, saturated_from_history, blocked_families, history, failure_families,
        )
        is_selected = result.selected_task is not None and task.id == result.selected_task.id

        why = ""
        if is_selected:
            why = f"Selected via {result.selected_source}"
            if result.advisory_honored:
                why = "Advisory recommendation honored"
        elif reasons:
            why = "; ".join(reasons)
        elif task.status != TaskStatus.pending:
            why = f"Status is {task.status.value if isinstance(task.status, TaskStatus) else task.status}"
        else:
            why = "Lower priority than selected task"

        candidates.append(CandidateExplanation(
            task_id=task.id,
            title=task.title or task.description[:80],
            family=family,
            priority=task.priority,
            risk=task.risk,
            status=task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
            selected=is_selected,
            score=round(score, 1),
            why=why,
            rejected_reasons=reasons if not is_selected else [],
        ))

    # Sort candidates by score descending
    candidates.sort(key=lambda c: -c.score)

    # Build explanation
    explanation = SelectionExplanation(
        selected=result.selected_task.to_dict() if result.selected_task else None,
        candidates=candidates,
        saturated_families=all_saturated,
        blocked_families=blocked_families,
        recent_failure_count=len(failures),
        recent_decision_count=len(decisions),
        selection_source=result.selected_source,
    )

    # Summary
    if result.selected_task:
        task = result.selected_task
        explanation.summary = (
            f"Selected task #{task.id} '{task.title or task.description[:40]}' "
            f"(family={task.family}, source={result.selected_source}). "
            f"{len(candidates)} candidates evaluated, "
            f"{len(all_saturated)} families saturated, "
            f"{len(blocked_families)} blocked."
        )
    else:
        explanation.summary = (
            f"No task selected. "
            f"{len(candidates)} candidates evaluated, "
            f"all rejected or unavailable."
        )

    return explanation
