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
    """Router module 7/10 — _create_app_impl chunk 7."""
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

    @router.get("/api/controller/missions")
    async def api_controller_list_missions() -> Dict[str, object]:
        """List all controlled missions."""
        from igris.core.mission_controller import list_controlled_missions
        missions = list_controlled_missions(project_root=str(CONFIG.project_root))
        return {"missions": [m.to_dict() for m in missions], "count": len(missions)}

    @router.get("/api/controller/missions/{mission_id}")
    async def api_controller_get_mission(mission_id: str) -> Dict[str, object]:
        """Get a controlled mission by ID."""
        from igris.core.mission_controller import load_controlled_mission
        mission = load_controlled_mission(mission_id, project_root=str(CONFIG.project_root))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.get("/api/controller/missions/{mission_id}/explain")
    async def api_controller_explain(mission_id: str) -> Dict[str, object]:
        """Explain current mission state and next action."""
        from igris.core.mission_controller import load_controlled_mission
        mission = load_controlled_mission(mission_id, project_root=str(CONFIG.project_root))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.explain_state()

    @router.post("/api/controller/missions/{mission_id}/plan")
    async def api_controller_plan(mission_id: str) -> Dict[str, object]:
        """Generate plan for a controlled mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.plan_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission planned: {mission.title}",
            "detail": f"{mission.total_steps} steps", "severity": "info",
            "mission_id": mission.id, "trace_id": mission.trace_id,
        })
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/execute-next")
    async def api_controller_execute_next(mission_id: str) -> Dict[str, object]:
        """Execute the next step in the mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        result = ctrl.execute_next_step(mission_id)
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @router.post("/api/controller/missions/{mission_id}/report-outcome")
    async def api_controller_report_outcome(mission_id: str, request: Request) -> Dict[str, object]:
        """Report step outcome."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        step_index = content.get("step_index", 0)
        outcome = content.get("outcome", "success")
        detail = content.get("detail", "")
        mission = ctrl.report_step_outcome(mission_id, step_index, outcome, detail)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/pause")
    async def api_controller_pause(mission_id: str, request: Request) -> Dict[str, object]:
        """Pause a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        mission = ctrl.pause_mission(mission_id, content.get("reason", ""))
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/resume")
    async def api_controller_resume(mission_id: str) -> Dict[str, object]:
        """Resume a paused mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.resume_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/block")
    async def api_controller_block(mission_id: str, request: Request) -> Dict[str, object]:
        """Block a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        reason = content.get("reason", "blocked")
        mission = ctrl.block_mission(mission_id, reason)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/unblock")
    async def api_controller_unblock(mission_id: str) -> Dict[str, object]:
        """Unblock a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        mission = ctrl.unblock_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    @router.post("/api/controller/missions/{mission_id}/verify")
    async def api_controller_verify(mission_id: str) -> Dict[str, object]:
        """Verify mission success criteria."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        result = ctrl.verify_mission(mission_id)
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        return result

    @router.get("/api/controller/missions/{mission_id}/report")
    async def api_controller_report(mission_id: str) -> Dict[str, object]:
        """Generate final report for a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        report = ctrl.generate_final_report(mission_id)
        if not report:
            raise HTTPException(status_code=404, detail="Mission not found")
        return report

    @router.get("/api/controller/missions/{mission_id}/context")
    async def api_controller_context(mission_id: str) -> Dict[str, object]:
        """Reconstruct mission context (for restart recovery)."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        ctx = ctrl.reconstruct_context(mission_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Mission not found")
        return ctx

    @router.post("/api/controller/missions/{mission_id}/artifacts")
    async def api_controller_add_artifact(mission_id: str, request: Request) -> Dict[str, object]:
        """Add an artifact to a mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        mission = ctrl.add_artifact(
            mission_id,
            artifact_type=content.get("type", "file"),
            path=content.get("path", ""),
            description=content.get("description", ""),
        )
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return mission.to_dict()

    # ---- Safety / Risk / Rollback (Epic #42) ----

    @router.post("/api/safety/classify-risk")
    async def api_classify_risk(request: Request) -> Dict[str, object]:
        """Classify action risk level."""
        from igris.core.risk_classifier import classify_action_risk
        content = await request.json()
        action_id = content.get("action_id", "")
        description = content.get("description", "")
        risk = classify_action_risk(action_id, description)
        return {"action_id": action_id, "risk_level": risk}

    @router.post("/api/safety/check-approval")
    async def api_check_approval(request: Request) -> Dict[str, object]:
        """Check if an action is approved under current policy."""
        from igris.core.risk_classifier import check_approval
        content = await request.json()
        decision = check_approval(
            action_id=content.get("action_id", ""),
            risk_level=content.get("risk_level", "low"),
            approval_mode=content.get("approval_mode", "safe"),
            has_rollback=content.get("has_rollback", False),
            host=content.get("host", ""),
            authorized_hosts=content.get("authorized_hosts"),
            approval_token=content.get("approval_token"),
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @router.post("/api/safety/guard-secret")
    async def api_guard_secret(request: Request) -> Dict[str, object]:
        """Check if a file path is a secret file."""
        from igris.core.risk_classifier import guard_secret_access
        content = await request.json()
        decision = guard_secret_access(content.get("path", ""), content.get("action", "read"))
        return decision.to_dict()

    @router.post("/api/rollback/backup-file")
    async def api_rollback_backup_file(request: Request) -> Dict[str, object]:
        """Create a file backup for rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        content = await request.json()
        entry = mgr.backup_file(
            file_path=content.get("file_path", ""),
            mission_id=content.get("mission_id", ""),
            action_id=content.get("action_id", ""),
            trace_id=content.get("trace_id", ""),
            description=content.get("description", ""),
        )
        if not entry:
            raise HTTPException(status_code=400, detail="File not found or backup failed")
        return entry.to_dict()

    @router.post("/api/rollback/save-state")
    async def api_rollback_save_state(request: Request) -> Dict[str, object]:
        """Save a state snapshot for rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        content = await request.json()
        entry = mgr.save_state_snapshot(
            state=content.get("state", {}),
            mission_id=content.get("mission_id", ""),
            action_id=content.get("action_id", ""),
            trace_id=content.get("trace_id", ""),
            description=content.get("description", ""),
        )
        return entry.to_dict()

    @router.get("/api/rollback/entries")
    async def api_rollback_list(mission_id: str = "", limit: int = 50) -> Dict[str, object]:
        """List rollback entries."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        entries = mgr.list_entries(mission_id=mission_id or None, limit=limit)
        return {"entries": entries, "count": len(entries)}

    @router.get("/api/rollback/entries/{entry_id}")
    async def api_rollback_get(entry_id: str) -> Dict[str, object]:
        """Get a rollback entry."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        entry = mgr.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Rollback entry not found")
        return entry

    @router.post("/api/rollback/entries/{entry_id}/verify")
    async def api_rollback_verify(entry_id: str) -> Dict[str, object]:
        """Verify if a rollback can be applied."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        return mgr.verify_rollback_applicable(entry_id)

    @router.post("/api/rollback/entries/{entry_id}/apply")
    async def api_rollback_apply(entry_id: str) -> Dict[str, object]:
        """Apply a file rollback."""
        from igris.core.rollback_manager import RollbackManager
        mgr = RollbackManager(project_root=str(CONFIG.project_root))
        success = mgr.apply_file_rollback(entry_id)
        return {"applied": success, "entry_id": entry_id}

    @router.get("/api/safety/events")
    async def api_safety_events(
        event_type: str = "",
        mission_id: str = "",
        severity: str = "",
        limit: int = 100,
    ) -> Dict[str, object]:
        """List safety events."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        events = log.list_events(
            event_type=event_type or None,
            mission_id=mission_id or None,
            severity=severity or None,
            limit=limit,
        )
        return {"events": events, "count": len(events)}

    @router.get("/api/safety/events/{event_id}")
    async def api_safety_event_detail(event_id: str) -> Dict[str, object]:
        """Get a specific safety event."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        event = log.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Safety event not found")
        return event

    @router.get("/api/safety/summary")
    async def api_safety_summary(mission_id: str = "") -> Dict[str, object]:
        """Get safety event summary."""
        from igris.core.safety_event_log import SafetyEventLog
        log = SafetyEventLog(project_root=str(CONFIG.project_root))
        return log.get_summary(mission_id)

    # ---- Tool Runtime (Epic #41) ----


    return router
