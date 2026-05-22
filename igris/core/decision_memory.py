"""
Decision & failure memory for IGRIS_GPT.

Stores structured decision events, failure events, saturation markers
and remediation attempts under `.igris/memory/`.  Provides query
functions used by teacher payload and task selection to avoid
repeating mistakes and to respect saturated families.

All stored text is redacted for secrets before persistence.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DecisionEvent:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    event_type: str = "decision"  # decision | failure | saturation | remediation
    family: str = ""
    task_id: str = ""
    title: str = ""
    description: str = ""
    outcome: str = ""  # success | failure | blocked | skipped
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["title"] = redact_secrets(d.get("title", ""))
        d["description"] = redact_secrets(d.get("description", ""))
        d["reason"] = redact_secrets(d.get("reason", ""))
        d["outcome"] = redact_secrets(d.get("outcome", ""))
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionEvent":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _memory_dir(project_root: Optional[str] = None) -> Path:
    root = Path(project_root) if project_root else CONFIG.project_root
    d = root / ".igris" / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_events(event_type: str, project_root: Optional[str] = None) -> List[DecisionEvent]:
    path = _memory_dir(project_root) / f"{event_type}s.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [DecisionEvent.from_dict(e) for e in data]
    except (json.JSONDecodeError, KeyError):
        return []


def _save_events(event_type: str, events: List[DecisionEvent], project_root: Optional[str] = None) -> None:
    path = _memory_dir(project_root) / f"{event_type}s.json"
    data = [e.to_dict() for e in events]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_event(event: DecisionEvent, project_root: Optional[str] = None) -> DecisionEvent:
    events = _load_events(event.event_type, project_root)
    events.append(event)
    _save_events(event.event_type, events, project_root)
    try:
        from igris.core.memory_graph import MemoryGraph
        root = project_root or str(CONFIG.project_root)
        graph = MemoryGraph(root)
        mapping = {"decision": "decision", "failure": "lesson", "saturation": "capability", "remediation": "run_event"}
        graph.add_node(mapping.get(event.event_type, "run_event"), event.to_dict())
    except Exception:
        pass
    return event


# ---------------------------------------------------------------------------
# Public API — record events
# ---------------------------------------------------------------------------

def record_decision(
    title: str,
    family: str = "",
    task_id: str = "",
    description: str = "",
    outcome: str = "success",
    reason: str = "",
    context: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> DecisionEvent:
    """Record a decision event."""
    event = DecisionEvent(
        event_type="decision",
        title=redact_secrets(title),
        family=family,
        task_id=task_id,
        description=redact_secrets(description),
        outcome=outcome,
        reason=redact_secrets(reason),
        context=context or {},
    )
    return _append_event(event, project_root)


def record_failure(
    title: str,
    family: str = "",
    task_id: str = "",
    description: str = "",
    reason: str = "",
    context: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> DecisionEvent:
    """Record a failure event."""
    event = DecisionEvent(
        event_type="failure",
        title=redact_secrets(title),
        family=family,
        task_id=task_id,
        description=redact_secrets(description),
        outcome="failure",
        reason=redact_secrets(reason),
        context=context or {},
    )
    return _append_event(event, project_root)


def record_saturation(
    family: str,
    reason: str = "",
    context: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> DecisionEvent:
    """Record a family saturation event."""
    event = DecisionEvent(
        event_type="saturation",
        title=f"Family '{family}' saturated",
        family=family,
        outcome="blocked",
        reason=redact_secrets(reason),
        context=context or {},
    )
    recorded = _append_event(event, project_root)
    try:
        from igris.core.memory_graph import MemoryGraph
        root = project_root or str(CONFIG.project_root)
        MemoryGraph(root).add_node("capability", {"family": family, "saturated": True})
    except Exception:
        pass
    return recorded


def record_remediation_attempt(
    title: str,
    family: str = "",
    task_id: str = "",
    description: str = "",
    outcome: str = "pending",
    reason: str = "",
    context: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> DecisionEvent:
    """Record a remediation attempt."""
    event = DecisionEvent(
        event_type="remediation",
        title=redact_secrets(title),
        family=family,
        task_id=task_id,
        description=redact_secrets(description),
        outcome=outcome,
        reason=redact_secrets(reason),
        context=context or {},
    )
    return _append_event(event, project_root)


# ---------------------------------------------------------------------------
# Public API — query events
# ---------------------------------------------------------------------------

def get_recent_decisions(limit: int = 20, project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent decision events."""
    events = _load_events("decision", project_root)
    return [e.to_dict() for e in events[-limit:]]


def get_recent_failures(limit: int = 20, project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent failure events."""
    events = _load_events("failure", project_root)
    return [e.to_dict() for e in events[-limit:]]


def get_saturated_families(project_root: Optional[str] = None) -> List[str]:
    """Return list of currently saturated family names."""
    events = _load_events("saturation", project_root)
    families = set()
    for e in events:
        if e.family:
            families.add(e.family)
    return sorted(families)


def should_avoid_family(family: str, project_root: Optional[str] = None) -> bool:
    """Check if a family should be avoided based on saturation and recent failures."""
    saturated = get_saturated_families(project_root)
    if family in saturated:
        return True
    failures = _load_events("failure", project_root)
    recent_failures = failures[-10:]
    family_failure_count = sum(1 for f in recent_failures if f.family == family)
    return family_failure_count >= 3


def explain_memory_constraints(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Return a summary of memory constraints for teacher/planner use."""
    saturated = get_saturated_families(project_root)
    failures = get_recent_failures(limit=10, project_root=project_root)
    decisions = get_recent_decisions(limit=10, project_root=project_root)
    remediations = _load_events("remediation", project_root)

    failed_families: Dict[str, int] = {}
    for f in failures:
        fam = f.get("family", "")
        if fam:
            failed_families[fam] = failed_families.get(fam, 0) + 1

    avoid_families = list(set(saturated) | {
        fam for fam, count in failed_families.items() if count >= 3
    })

    return {
        "saturated_families": saturated,
        "recently_failed_families": failed_families,
        "avoid_families": avoid_families,
        "recent_failure_count": len(failures),
        "recent_decision_count": len(decisions),
        "remediation_count": len(remediations),
        "recommendation": (
            f"Avoid families: {', '.join(avoid_families)}" if avoid_families
            else "No constraints — all families available"
        ),
    }


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def get_memory_constraints_for_teacher(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Return memory constraints formatted for teacher payload integration."""
    constraints = explain_memory_constraints(project_root)
    return {
        "memory_constraints": constraints,
        "avoid_families": constraints["avoid_families"],
        "saturated_families": constraints["saturated_families"],
    }


def get_blocked_families_from_memory(project_root: Optional[str] = None) -> List[str]:
    """Return families that should be blocked in task selection."""
    constraints = explain_memory_constraints(project_root)
    return constraints["avoid_families"]
