"""Context-enriched chat.

Enriches chat messages with project context: mission status, tasks,
reports, memory constraints, git state, patch proposals, validation
state, cost/routing info.

The chat can propose patches or tasks but NOT apply them without
going through the proper workflow (validate → apply).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from igris.core import decision_memory
from igris.core import mission_planner
from igris.core import patch_proposal as patch_mod
from igris.core import project_state as project_state_mod
from igris.core.safety import redact_secrets
from igris.core.task_engine import TaskEngine
from igris.layers.advisory import router as provider_router
from igris.layers.git_layer.git_status import get_git_info
from igris.models.config import CONFIG
from igris.models.task import TaskStatus


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def build_chat_context(
    task_engine: Optional[TaskEngine] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Build full project context for chat enrichment."""
    pr = project_root or str(CONFIG.project_root)

    context: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sections": {},
    }

    context["sections"]["missions"] = _build_missions_context()
    context["sections"]["tasks"] = _build_tasks_context(task_engine)
    context["sections"]["memory"] = _build_memory_context(pr)
    context["sections"]["git"] = _build_git_context()
    context["sections"]["patches"] = _build_patches_context()
    context["sections"]["cost"] = _build_cost_context()
    context["sections"]["project_state"] = _build_project_state_context(pr)

    return context


def build_context_system_prompt(
    task_engine: Optional[TaskEngine] = None,
    project_root: Optional[str] = None,
) -> str:
    """Build a system prompt enriched with project context."""
    ctx = build_chat_context(task_engine, project_root)
    sections = ctx.get("sections", {})

    parts = [
        "You are IGRIS_GPT, an AI engineering agent. "
        "You help with code, testing, task management, and project operations. "
        "Be concise and actionable.",
        "",
        "=== CURRENT PROJECT CONTEXT ===",
    ]

    # Missions
    m = sections.get("missions", {})
    if m.get("total", 0) > 0:
        parts.append(f"\nMissions: {m['total']} total, {m.get('active', 0)} active, "
                     f"{m.get('completed', 0)} completed")
        for title in m.get("active_titles", [])[:3]:
            parts.append(f"  - Active: {title}")

    # Tasks
    t = sections.get("tasks", {})
    if t.get("total", 0) > 0:
        parts.append(f"\nTasks: {t['total']} total — "
                     f"{t.get('pending', 0)} pending, {t.get('running', 0)} running, "
                     f"{t.get('completed', 0)} completed, {t.get('blocked', 0)} blocked")

    # Memory constraints
    mem = sections.get("memory", {})
    if mem.get("avoid_families"):
        parts.append(f"\nMemory: avoid families {mem['avoid_families']}")
    if mem.get("recent_failure_count", 0) > 0:
        parts.append(f"  Recent failures: {mem['recent_failure_count']}")

    # Git
    g = sections.get("git", {})
    if g.get("branch"):
        parts.append(f"\nGit: branch={g['branch']}, dirty={g.get('dirty', False)}")

    # Patches
    p = sections.get("patches", {})
    if p.get("total", 0) > 0:
        parts.append(f"\nPatches: {p['total']} proposals — "
                     f"{p.get('pending', 0)} pending, {p.get('applied', 0)} applied")

    # Cost
    c = sections.get("cost", {})
    if c.get("provider"):
        parts.append(f"\nRouting: provider={c['provider']}, model={c.get('model', 'unknown')}")

    # Project state
    ps = sections.get("project_state", {})
    if ps.get("cooling_down"):
        parts.append(f"\nCooling down families: {ps['cooling_down']}")
    if ps.get("critical"):
        parts.append(f"Critical families: {ps['critical']}")

    parts.append("")
    parts.append("You can suggest tasks or patches, but do NOT execute commands "
                 "or apply changes directly. All actions go through the proper workflow.")

    return "\n".join(parts)


def get_context_summary(
    task_engine: Optional[TaskEngine] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a concise context summary for API consumers."""
    ctx = build_chat_context(task_engine, project_root)
    sections = ctx["sections"]

    return {
        "timestamp": ctx["timestamp"],
        "missions_active": sections.get("missions", {}).get("active", 0),
        "tasks_pending": sections.get("tasks", {}).get("pending", 0),
        "tasks_blocked": sections.get("tasks", {}).get("blocked", 0),
        "memory_avoid_families": sections.get("memory", {}).get("avoid_families", []),
        "git_branch": sections.get("git", {}).get("branch", ""),
        "git_dirty": sections.get("git", {}).get("dirty", False),
        "patches_pending": sections.get("patches", {}).get("pending", 0),
        "provider": sections.get("cost", {}).get("provider", ""),
        "cooling_down": sections.get("project_state", {}).get("cooling_down", []),
    }


# ---------------------------------------------------------------------------
# Internal context builders
# ---------------------------------------------------------------------------

def _build_missions_context() -> Dict[str, Any]:
    try:
        missions = mission_planner.list_missions()
        active = [m for m in missions if m.get("status") in ("active", "planning", "executing")]
        completed = [m for m in missions if m.get("status") == "completed"]
        return {
            "total": len(missions),
            "active": len(active),
            "completed": len(completed),
            "active_titles": [redact_secrets(m.get("title", "")) for m in active[:5]],
        }
    except Exception:
        return {"total": 0, "active": 0, "completed": 0, "active_titles": []}


def _build_tasks_context(task_engine: Optional[TaskEngine] = None) -> Dict[str, Any]:
    try:
        if task_engine is None:
            task_engine = TaskEngine()
        tasks = task_engine.list_tasks()
        pending = sum(1 for t in tasks if t.status == TaskStatus.pending)
        running = sum(1 for t in tasks if t.status == TaskStatus.running)
        completed = sum(1 for t in tasks if t.status == TaskStatus.completed)
        blocked = sum(1 for t in tasks if t.status == TaskStatus.blocked)
        return {
            "total": len(tasks),
            "pending": pending,
            "running": running,
            "completed": completed,
            "blocked": blocked,
        }
    except Exception:
        return {"total": 0, "pending": 0, "running": 0, "completed": 0, "blocked": 0}


def _build_memory_context(project_root: str) -> Dict[str, Any]:
    try:
        constraints = decision_memory.explain_memory_constraints(project_root=project_root)
        return {
            "avoid_families": constraints.get("avoid_families", []),
            "saturated_families": constraints.get("saturated_families", []),
            "recent_failure_count": constraints.get("recent_failure_count", 0),
            "recent_decision_count": constraints.get("recent_decision_count", 0),
        }
    except Exception:
        return {"avoid_families": [], "saturated_families": [], "recent_failure_count": 0, "recent_decision_count": 0}


def _build_git_context() -> Dict[str, Any]:
    try:
        info = get_git_info()
        return {
            "branch": info.branch,
            "dirty": info.dirty,
            "changed": info.changed,
            "head": info.head[:8] if info.head else "",
        }
    except Exception:
        return {"branch": "", "dirty": False, "changed": 0, "head": ""}


def _build_patches_context() -> Dict[str, Any]:
    try:
        patches = patch_mod.list_proposals()
        pending = sum(1 for p in patches if p.get("status") == "pending")
        applied = sum(1 for p in patches if p.get("status") == "applied")
        return {
            "total": len(patches),
            "pending": pending,
            "applied": applied,
        }
    except Exception:
        return {"total": 0, "pending": 0, "applied": 0}


def _build_cost_context() -> Dict[str, Any]:
    try:
        provider, model = provider_router.choose_provider()
        return {
            "provider": provider,
            "model": model,
        }
    except Exception:
        return {"provider": "", "model": ""}


def _build_project_state_context(project_root: str) -> Dict[str, Any]:
    try:
        state = project_state_mod.get_project_state(project_root=project_root)
        return {
            "cooling_down": state.get("cooling_down", []),
            "critical": state.get("critical_families", []),
            "elevated": state.get("elevated_families", []),
        }
    except Exception:
        return {"cooling_down": [], "critical": [], "elevated": []}
