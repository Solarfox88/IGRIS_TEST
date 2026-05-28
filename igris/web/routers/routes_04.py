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
    """Router module 4/10 — _create_app_impl chunk 4."""
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

    @router.get("/api/loop/status")
    async def api_loop_status() -> Dict[str, object]:
        return autonomous_loop.get_loop_status().to_dict()

    @router.get("/api/loop/recent")
    async def api_loop_recent(limit: int = 20) -> Dict[str, object]:
        return {"steps": autonomous_loop.get_recent_steps(limit)}

    # ---- Validation ----

    @router.post("/api/tasks/{task_id}/validate")
    async def api_validate_task(task_id: int, request: Request) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        content = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            content = await request.json()
        reports = execution_report.recent_reports(limit=10)
        files_changed = content.get("files_changed", [])
        manual_reason = content.get("manual_completion_reason", "")
        result = task_validator.validate_task_completion(
            task, reports=reports, files_changed=files_changed,
            manual_completion_reason=manual_reason,
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "validation", "task_id": task_id,
            "title": f"Validation: {result.overall_status}",
            "detail": result.reason, "severity": "info" if result.valid else "warning",
        })
        return result.to_dict()

    @router.get("/api/tasks/{task_id}/validations")
    async def api_task_validations(task_id: int) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        validations = task_validator.get_validations_for_task(
            task_id, project_root=str(CONFIG.project_root),
        )
        return {"validations": [v.to_dict() for v in validations]}

    @router.get("/api/validations/{validation_id}")
    async def api_get_validation(validation_id: str) -> Dict[str, object]:
        result = task_validator.get_validation(
            validation_id, project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Validation not found")
        return result.to_dict()

    @router.post("/api/tasks/{task_id}/complete")
    async def api_complete_task_validated(task_id: int, request: Request) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        content = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            content = await request.json()
        # Check existing validations
        validations = task_validator.get_validations_for_task(
            task_id, project_root=str(CONFIG.project_root),
        )
        has_valid = any(v.valid for v in validations)
        manual_reason = content.get("manual_completion_reason", "")
        if not has_valid and not manual_reason:
            raise HTTPException(
                status_code=400,
                detail="Task has no passing validation. Provide manual_completion_reason or validate first.",
            )
        if manual_reason and not has_valid:
            # Create manual validation
            task_validator.validate_task_completion(
                task, manual_completion_reason=manual_reason,
                project_root=str(CONFIG.project_root),
            )
        updated = task_engine.complete_task(task_id, result=manual_reason or "Validated completion")
        task_engine.append_timeline_event({
            "type": "validation", "task_id": task_id,
            "title": "Task completed (validated)",
            "detail": manual_reason or "passed validation",
            "severity": "info",
        })
        return updated.to_dict() if updated else {}

    # ---- Missions ----

    def _redact_mission_dict(d: Dict[str, object]) -> Dict[str, object]:
        """Redact secrets from mission response."""
        for key in ("title", "description", "plan_summary"):
            if key in d and isinstance(d[key], str):
                d[key] = _redact(d[key])
        if "steps" in d and isinstance(d["steps"], list):
            for step in d["steps"]:
                for key in ("title", "description"):
                    if key in step and isinstance(step[key], str):
                        step[key] = _redact(step[key])
        return d

    @router.post("/api/missions")
    async def api_create_mission(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "")
        description = content.get("description", "")
        if not title:
            raise HTTPException(status_code=400, detail="Mission title required")
        m = mission_planner.Mission(title=title, description=description)
        mission_planner.save_mission(m, project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission created: {_redact(title)}",
            "detail": _redact(description[:200]), "severity": "info",
        })
        return _redact_mission_dict(m.to_dict())

    @router.get("/api/missions")
    async def api_list_missions() -> Dict[str, object]:
        missions = mission_planner.list_missions(project_root=str(CONFIG.project_root))
        return {"missions": [_redact_mission_dict(m.to_dict()) for m in missions]}

    @router.get("/api/missions/{mission_id}")
    async def api_get_mission(mission_id: str) -> Dict[str, object]:
        m = mission_planner.load_mission(mission_id, project_root=str(CONFIG.project_root))
        if not m:
            raise HTTPException(status_code=404, detail="Mission not found")
        return _redact_mission_dict(m.to_dict())

    @router.post("/api/missions/{mission_id}/plan")
    async def api_plan_mission(
        mission_id: str,
        mode: str = "deterministic",
    ) -> Dict[str, object]:
        from igris.core import llm_planner
        if mode not in ("deterministic", "llm", "auto"):
            raise HTTPException(status_code=400, detail="Invalid mode. Use: deterministic, llm, auto")
        result = llm_planner.plan_mission_with_mode(
            mission_id, mode=mode, project_root=str(CONFIG.project_root),
        )
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        task_engine.append_timeline_event({
            "type": "mission",
            "title": f"Mission planned ({result['planning']['mode']}): {_redact(result['mission'].get('title', ''))}",
            "detail": f"{len(result['mission'].get('steps', []))} steps",
            "severity": "info",
        })
        result["mission"] = _redact_mission_dict(result["mission"])
        return result

    @router.get("/api/missions/{mission_id}/plan/explain")
    async def api_plan_explain(mission_id: str) -> Dict[str, object]:
        from igris.core import llm_planner
        explanation = llm_planner.explain_plan(
            mission_id, project_root=str(CONFIG.project_root),
        )
        if not explanation:
            raise HTTPException(status_code=404, detail="Mission not found")
        return explanation

    @router.post("/api/missions/{mission_id}/materialize-tasks")
    async def api_materialize_tasks(mission_id: str) -> Dict[str, object]:
        m = mission_planner.materialize_tasks(
            mission_id, task_engine, project_root=str(CONFIG.project_root),
        )
        if not m:
            raise HTTPException(status_code=404, detail="Mission not found or no plan")
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Tasks materialized: {_redact(m.title)}",
            "detail": f"{len(m.task_ids)} tasks created", "severity": "info",
        })
        return _redact_mission_dict(m.to_dict())

    @router.get("/api/missions/{mission_id}/graph")
    async def api_mission_graph(mission_id: str) -> Dict[str, object]:
        graph = mission_planner.get_mission_graph(
            mission_id, project_root=str(CONFIG.project_root),
        )
        if not graph:
            raise HTTPException(status_code=404, detail="Mission not found")
        return graph

    # ---- Task management ----

    @router.get("/api/tasks")
    async def api_list_tasks() -> Dict[str, object]:
        tasks = []
        for t in task_engine.tasks:
            tasks.append(t.to_dict())
        return {"tasks": tasks}

    @router.post("/api/tasks")
    async def api_create_task(content: Dict[str, str] = Body(...)) -> Dict[str, object]:
        description = content.get("description")
        if not description:
            raise HTTPException(status_code=400, detail="description is required")
        title = content.get("title")
        source = content.get("source", "user")
        task = task_engine.create_task(description, title=title, source=source)
        task_engine.append_timeline_event({
            "type": "task", "title": f"Task created: {title or description[:40]}",
            "detail": description[:100], "related_task_id": task.id,
            "severity": "info",
        })
        return task.to_dict()

    @router.get("/api/tasks/{task_id}")
    async def api_get_task(task_id: int) -> Dict[str, object]:
        task = task_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    @router.post("/api/tasks/{task_id}/complete")
    async def api_complete_task(task_id: int, body: Dict[str, str] = Body(default={})):  # type: ignore[assignment]
        result_text = body.get("result") if isinstance(body, dict) else None
        task = task_engine.complete_task(task_id, result_text)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task_engine.append_timeline_event({
            "type": "task", "title": f"Task completed: #{task_id}",
            "detail": result_text or "", "related_task_id": task_id,
            "severity": "info",
        })
        return task.to_dict()

    @router.post("/api/tasks/{task_id}/block")
    async def api_block_task(task_id: int, body: Dict[str, str] = Body(default={})):  # type: ignore[assignment]
        reason = body.get("reason") if isinstance(body, dict) else None
        task = task_engine.block_task(task_id, reason)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_dict()

    # ---- Terminal ----

    @router.get("/api/terminal/commands")
    async def api_terminal_commands() -> Dict[str, object]:
        return {"commands": list(ALLOWED_COMMANDS.keys())}

    @router.post("/api/terminal/run")
    async def api_terminal_run(command: Dict[str, str] = Body(...)) -> Dict[str, object]:
        # Reject if raw 'command' string is passed instead of command_id
        if "command" in command and "command_id" not in command:
            raise HTTPException(status_code=400, detail="Use command_id, not command")
        cmd_id = command.get("command_id")
        if not cmd_id:
            raise HTTPException(status_code=400, detail="command_id is required")
        if not safety.check_command_allowed(cmd_id):
            raise HTTPException(status_code=403, detail="Command not allowed")
        if nonlocal_cmd_running["running"]:
            raise HTTPException(status_code=409, detail="A command is already running")
        nonlocal_cmd_running["running"] = True
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        try:
            result = execution_runner.run_safe_command(cmd_id)
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            duration_ms = int((time.monotonic() - t0) * 1000)
            stdout = _redact(result.get("stdout", ""))
            stderr = _redact(result.get("stderr", ""))
            report = execution_report.create_report(
                command_id=cmd_id, capability_id="execution.run_safe_command",
                returncode=result.get("returncode", 1),
                stdout=result.get("stdout", ""), stderr=result.get("stderr", ""),
                started_at=started, finished_at=finished, duration_ms=duration_ms,
            )
            # Route outcome
            recommendation = route_outcome(report, f"terminal {cmd_id}")
            task_engine.append_timeline_event({
                "type": "action", "title": f"Command: {cmd_id}",
                "detail": f"exit={result.get('returncode', 1)}, {duration_ms}ms",
                "related_report_id": report.get("report_id"),
                "severity": "info" if result.get("returncode") == 0 else "warning",
            })
            return {"command_id": cmd_id, "stdout": stdout, "stderr": stderr, "returncode": result.get("returncode")}
        finally:
            nonlocal_cmd_running["running"] = False

    # ---- Reports ----

    @router.get("/api/reports/recent")
    async def api_reports_recent() -> Dict[str, object]:
        return {"reports": execution_report.recent_reports()}

    @router.get("/api/reports/{report_id}")
    async def api_get_report(report_id: str) -> Dict[str, object]:
        report = execution_report.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    # ---- A2A ----


    return router
