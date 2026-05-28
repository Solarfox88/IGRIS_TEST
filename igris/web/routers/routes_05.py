"""IGRIS web server router — auto-split from server.py (#725).

Route handlers are extracted from _create_app_impl; shared app state is
received via ``deps`` (SimpleNamespace). Do not edit route logic here;
changes should first be made in the original handler before full migration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from igris.core import anti_loop
from igris.core import chat_context
from igris.core import chat_streaming
from igris.core import decision_memory
from igris.core import diagnostics as diagnostics_mod
from igris.core import execution_report
from igris.core import mission_planner
from igris.core import project_state as project_state_mod
from igris.core import safe_policy
from igris.core import safety
from igris.core import task_selection_explain
from igris.core import decision_report as decision_report_mod
from igris.core import autonomous_loop
from igris.core.chat_engine import chat as chat_llm, check_ollama_available
from igris.core import patch_proposal as patch_mod
from igris.core.memory import recent_memory_events, append_memory_event
from igris.core.memory_graph import MemoryGraph
from igris.core.outcome_router import route_outcome
from igris.core.project_context import build_project_snapshot
from igris.core.teacher import (
    build_teacher_payload, validate_teacher_assignment, propose_remediation_task,
)
from igris.core.task_engine import TaskEngine
from igris.layers.advisory import router as provider_router
from igris.layers.execution import runner as execution_runner
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
from igris.layers.git_layer import git_ops
from igris.layers.git_layer.git_status import get_git_info
from igris.layers.validation import validator as task_validator
from igris.models.config import CONFIG
from igris.models.report import GitStatusResponse, TestRunResponse
from igris.models.task import TaskStatus
from igris.agents import build_default_registry
from igris.a2a.agent_card import build_agent_card
from igris.a2a import task_store as a2a_store


def create_router(deps) -> APIRouter:
    """Router module 5/10 — _create_app_impl chunk 5."""
    router = APIRouter()
    # Unpack shared app state (names match what route bodies use directly)
    _redact = deps.redact
    _check_model_available = deps.check_model_available
    _get_graph = deps.get_graph
    jinja_env = deps.jinja_env
    sessions = deps.sessions
    task_engine = deps.task_engine
    nonlocal_test_running = deps.nonlocal_test_running
    nonlocal_cmd_running = deps.nonlocal_cmd_running

    @router.post("/api/a2a/tasks")
    async def a2a_create_task(task: Dict[str, object] = Body(...)) -> Dict[str, object]:
        description = None
        if isinstance(task, dict):
            description = task.get("description") or task.get("title")
        if not description:
            raise HTTPException(status_code=400, detail="description or title is required")
        created = task_engine.create_task(str(description), source="a2a")
        return created.to_dict()

    @router.get("/api/a2a/tasks/{task_id}")
    async def a2a_get_task(task_id: int) -> Dict[str, object]:
        t = task_engine.get_task(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
        return t.to_dict()

    @router.post("/api/a2a/tasks/{task_id}/messages")
    async def a2a_append_message(task_id: int, message: Dict[str, object] = Body(...)) -> Dict[str, object]:
        t = task_engine.get_task(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Invalid message format")
        entry = {
            "task_id": task_id,
            "sender": message.get("sender", "unknown"),
            "content": message.get("content", ""),
        }
        append_memory_event(f"a2a_messages_{task_id}", entry)
        task_engine.append_timeline_event({"event": "a2a_message", "task_id": task_id})
        return {"status": "ok"}

    @router.get("/.well-known/agent-card.json")
    @router.get("/.well-known/agent.json")
    async def well_known_agent(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url)
        from dataclasses import asdict
        return JSONResponse(content=asdict(card))

    @router.get("/api/a2a/agent-card")
    async def api_a2a_agent_card(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        card = build_agent_card(base_url)
        from dataclasses import asdict
        return JSONResponse(content=asdict(card))

    @router.get("/api/a2a/capabilities")
    async def api_a2a_capabilities() -> Dict[str, object]:
        from igris.agents import list_capabilities
        caps = list_capabilities()
        return {"capabilities": [{"id": c.id, "name": c.name, "description": c.description, "safe": c.safe, "risk": c.risk} for c in caps]}

    # ---- Safety Policy ----

    @router.get("/api/safety/policy")
    async def api_safety_policy() -> Dict[str, object]:
        return safe_policy.get_policy_status()

    @router.post("/api/safety/policy/check")
    async def api_safety_policy_check(request: Request) -> Dict[str, object]:
        content = await request.json()
        command_id = content.get("command_id", "")
        if not command_id:
            raise HTTPException(status_code=400, detail="command_id required")
        context = content.get("context")
        decision = safe_policy.check_command_policy(command_id, context=context)
        return decision.to_dict()

    # ---- Explainable Task Selection ----

    @router.get("/api/tasks/selection/explain")
    async def api_explain_task_selection() -> Dict[str, object]:
        tasks = task_engine.list_tasks()
        history = [t.description for t in tasks if t.status == TaskStatus.completed]
        explanation = task_selection_explain.explain_task_selection(
            candidate_tasks=tasks,
            history=history,
            project_root=str(CONFIG.project_root),
        )
        return explanation.to_dict()

    # ---- Project State + Saturation Cooldown ----

    @router.get("/api/project-state")
    async def api_project_state() -> Dict[str, object]:
        return project_state_mod.get_project_state(project_root=str(CONFIG.project_root))

    @router.get("/api/project-state/recovery")
    async def api_recovery_summary() -> Dict[str, object]:
        return project_state_mod.get_recovery_summary(project_root=str(CONFIG.project_root))

    @router.get("/api/project-state/family/{family}")
    async def api_family_availability(family: str) -> Dict[str, object]:
        return project_state_mod.is_family_available(family, project_root=str(CONFIG.project_root))

    @router.post("/api/project-state/family/{family}/reset-cooldown")
    async def api_reset_cooldown(family: str) -> Dict[str, object]:
        ok = project_state_mod.reset_family_cooldown(family, project_root=str(CONFIG.project_root))
        if not ok:
            raise HTTPException(status_code=404, detail=f"Family '{family}' not found")
        return {"family": family, "cooldown_reset": True}

    @router.get("/api/project-state/fingerprints")
    async def api_recent_fingerprints() -> Dict[str, object]:
        fps = project_state_mod.get_recent_fingerprints(limit=20, project_root=str(CONFIG.project_root))
        return {"fingerprints": fps}

    # ---- Decision Reports ----

    @router.get("/api/decision-reports")
    async def api_list_decision_reports() -> Dict[str, object]:
        reports = decision_report_mod.list_decision_reports(
            limit=20, project_root=str(CONFIG.project_root),
        )
        return {"reports": reports}

    @router.get("/api/decision-reports/{report_id}")
    async def api_get_decision_report(report_id: str) -> Dict[str, object]:
        report = decision_report_mod.get_decision_report(
            report_id, project_root=str(CONFIG.project_root),
        )
        if not report:
            raise HTTPException(status_code=404, detail="Decision report not found")
        return report

    @router.post("/api/decision-reports")
    async def api_create_decision_report(request: Request) -> Dict[str, object]:
        content = await request.json()
        tasks = task_engine.list_tasks()
        report = decision_report_mod.create_decision_report(
            step_number=content.get("step_number", 0),
            tasks=tasks,
            action_type=content.get("action_type", ""),
            action_detail=content.get("action_detail", ""),
            outcome=content.get("outcome", ""),
            outcome_reason=content.get("outcome_reason", ""),
            next_action=content.get("next_action", ""),
            next_action_reason=content.get("next_action_reason", ""),
            safety_decisions=content.get("safety_decisions", []),
            project_root=str(CONFIG.project_root),
        )
        return report.to_dict()

    # ---- Diagnostics ----

    @router.get("/api/diagnostics")
    async def api_diagnostics() -> Dict[str, object]:
        tasks = [t.to_dict() for t in task_engine.list_tasks()]
        timeline = task_engine.recent_timeline_events(limit=50)
        report = diagnostics_mod.run_diagnostics(
            tasks, timeline, project_root=str(CONFIG.project_root),
        )
        return report.to_dict()

    @router.get("/api/diagnostics/summary")
    async def api_diagnostics_summary() -> Dict[str, object]:
        tasks = [t.to_dict() for t in task_engine.list_tasks()]
        timeline = task_engine.recent_timeline_events(limit=50)
        return diagnostics_mod.get_diagnostic_summary(
            tasks, timeline, project_root=str(CONFIG.project_root),
        )

    # ---- A2A Task Store (artifacts, events, cancel) ----

    @router.post("/api/a2a/store/tasks")
    async def api_a2a_store_create(body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        title = body.get("title", "")
        description = body.get("description", "")
        if not title and not description:
            raise HTTPException(status_code=400, detail="title or description required")
        task = a2a_store.create_a2a_task(
            title=str(title), description=str(description),
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A task created: {_redact(str(title)[:80])}",
            "severity": "info",
        })
        return task

    @router.get("/api/a2a/store/tasks")
    async def api_a2a_store_list() -> Dict[str, object]:
        tasks = a2a_store.list_a2a_tasks(project_root=str(CONFIG.project_root))
        return {"tasks": tasks}

    @router.get("/api/a2a/store/tasks/{task_id}")
    async def api_a2a_store_get(task_id: str) -> Dict[str, object]:
        task = a2a_store.get_a2a_task(task_id, project_root=str(CONFIG.project_root))
        if not task:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return task

    @router.post("/api/a2a/store/tasks/{task_id}/status")
    async def api_a2a_store_update_status(task_id: str, body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        status = body.get("status", "")
        if not status or status not in a2a_store.VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status. Valid: {sorted(a2a_store.VALID_STATUSES)}")
        detail = str(body.get("detail", ""))
        result = a2a_store.update_a2a_task_status(
            task_id, str(status), detail=detail,
            project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Task not found or cannot transition")
        return result

    @router.get("/api/a2a/tasks/{task_id}/artifacts")
    async def api_a2a_artifacts_list(task_id: str) -> Dict[str, object]:
        artifacts = a2a_store.get_artifacts(task_id, project_root=str(CONFIG.project_root))
        if artifacts is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"artifacts": artifacts}

    @router.post("/api/a2a/tasks/{task_id}/artifacts")
    async def api_a2a_artifacts_add(task_id: str, body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        name = body.get("name", "")
        content = body.get("content", "")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        mime_type = str(body.get("mime_type", "text/plain"))
        result = a2a_store.add_artifact(
            task_id, name=str(name), content=str(content),
            mime_type=mime_type, project_root=str(CONFIG.project_root),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A artifact added: {_redact(str(name)[:80])}",
            "severity": "info",
        })
        return result

    @router.post("/api/a2a/tasks/{task_id}/cancel")
    async def api_a2a_cancel(task_id: str, body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        reason = str(body.get("reason", "")) if isinstance(body, dict) else ""
        result = a2a_store.cancel_a2a_task(
            task_id, reason=reason,
            project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="A2A task not found or already terminal")
        task_engine.append_timeline_event({
            "type": "a2a", "title": f"A2A task canceled: {task_id}",
            "severity": "warning",
        })
        return result

    @router.get("/api/a2a/tasks/{task_id}/events")
    async def api_a2a_events(task_id: str) -> Dict[str, object]:
        events = a2a_store.get_events(task_id, project_root=str(CONFIG.project_root))
        if events is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"events": events}

    # ---- Teacher Remediation ----

    @router.post("/api/teacher/remediate")
    async def api_teacher_remediate(body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        task_id = body.get("task_id") if isinstance(body, dict) else None
        create = body.get("create", False) if isinstance(body, dict) else False

        # Build teacher payload
        recent_tasks = task_engine.list_tasks()
        history = [t.description for t in recent_tasks]
        payload = build_teacher_payload(tasks=history)

        # Propose remediation
        proposed = propose_remediation_task(payload)

        # Validate if we have enough info
        validation = None
        if proposed.get("task_description"):
            assignment = {
                "diagnosis": proposed.get("reason", ""),
                "selected_family": proposed.get("family", "other"),
                "why_this_family": proposed.get("reason", ""),
                "differentiator": proposed.get("differentiator", ""),
                "task_title": proposed.get("task_title", ""),
                "task_description": proposed.get("task_description", ""),
                "success_criteria": proposed.get("success_criteria", []),
                "safe_command_ids": proposed.get("safe_command_ids", []),
                "expected_next_state": proposed.get("expected_next_state", ""),
                "fallback_if_blocked": proposed.get("fallback_if_blocked", ""),
            }
            validation = validate_teacher_assignment(assignment, history)

        result: Dict[str, object] = {
            "payload": payload,
            "proposed_task": proposed,
            "validation": validation,
            "created_task_id": None,
        }

        # Create the task if requested and valid
        if create and proposed.get("task_description"):
            if validation and validation.get("valid", False):
                created = task_engine.create_task(
                    description=proposed["task_description"],
                    title=proposed.get("task_title"),
                    family=proposed.get("family"),
                    source="teacher",
                )
                result["created_task_id"] = created.id
                task_engine.append_timeline_event({
                    "type": "teacher", "title": "Teacher remediation",
                    "detail": f"Created task: {proposed.get('task_title', '')}",
                    "related_task_id": created.id,
                })

        return result

    # ---- Outcome Router ----


    return router
