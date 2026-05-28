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
    """Router module 6/10 — _create_app_impl chunk 6."""
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

    @router.get("/api/outcome/recent")
    async def api_outcome_recent() -> Dict[str, object]:
        reports = execution_report.recent_reports(limit=10)
        outcomes = []
        for r in reports:
            rec = route_outcome(r)
            outcomes.append(rec)
        return {"outcomes": outcomes}

    # ---- Patch Proposals ----

    @router.get("/api/patches")
    async def api_list_patches() -> Dict[str, object]:
        patches = patch_mod.list_patch_proposals(project_root=str(CONFIG.project_root))
        return {"patches": patches}

    @router.post("/api/patches/generate")
    async def api_generate_patch(request: Request) -> Dict[str, object]:
        from igris.core import llm_patch_generator
        content = await request.json()
        task_title = content.get("title", content.get("task_title", ""))
        if not task_title:
            raise HTTPException(status_code=400, detail="title is required")
        result = llm_patch_generator.generate_patch(
            task_title=task_title,
            task_description=content.get("description", ""),
            context=content.get("context", ""),
        )
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch generated: {task_title[:80]}",
            "detail": f"by={result.get('generated_by', 'unknown')}, files={len(result.get('files', []))}",
            "severity": "info",
        })
        return result

    @router.post("/api/tasks/{task_id}/generate-patch")
    async def api_task_generate_patch(task_id: int) -> Dict[str, object]:
        from igris.core import llm_patch_generator
        task = task_engine.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        td = task.to_dict() if hasattr(task, "to_dict") else {}
        result = llm_patch_generator.generate_patch(
            task_title=td.get("title", ""),
            task_description=td.get("description", ""),
        )
        result["task_id"] = task_id
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch generated for task: {task_id}",
            "detail": f"by={result.get('generated_by', 'unknown')}",
            "severity": "info",
        })
        return result

    @router.post("/api/patches/propose")
    async def api_propose_patch(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "Untitled patch")
        description = content.get("description", "")
        task_id = content.get("task_id")
        files = content.get("files", [])
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        proposal = patch_mod.create_patch_proposal(
            title=title,
            description=description,
            files=files,
            task_id=task_id,
            project_root=str(CONFIG.project_root),
        )
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch proposed: {title}",
            "detail": f"{len(files)} file(s)", "severity": "info",
            "related_task_id": task_id,
            "related_patch_id": proposal.id,
        })
        return patch_mod._proposal_to_dict(proposal)

    @router.get("/api/patches/{proposal_id}")
    async def api_get_patch(proposal_id: str) -> Dict[str, object]:
        proposal = patch_mod.load_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        return patch_mod._proposal_to_dict(proposal)

    @router.post("/api/patches/{proposal_id}/validate")
    async def api_validate_patch(proposal_id: str) -> Dict[str, object]:
        proposal = patch_mod.load_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        result = patch_mod.validate_patch_proposal(proposal, project_root=str(CONFIG.project_root))
        severity = "info" if result.valid else "warning"
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch validated: {proposal.title}",
            "detail": f"valid={result.valid}, risk={result.risk}",
            "severity": severity,
            "related_patch_id": proposal.id,
        })
        return {
            "proposal_id": proposal_id,
            "status": proposal.status,
            "validation": {
                "valid": result.valid,
                "reasons": result.reasons,
                "blocked_paths": result.blocked_paths,
                "secret_findings": result.secret_findings,
                "risk": result.risk,
            },
        }

    @router.post("/api/patches/{proposal_id}/apply")
    async def api_apply_patch(proposal_id: str) -> Dict[str, object]:
        result = patch_mod.apply_patch_proposal(proposal_id, project_root=str(CONFIG.project_root))
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Apply failed"))
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch applied: {proposal_id}",
            "detail": f"{len(result.get('applied_files', []))} file(s) modified",
            "severity": "info",
            "related_patch_id": proposal_id,
        })
        return result

    @router.post("/api/patches/{proposal_id}/reject")
    async def api_reject_patch(proposal_id: str, request: Request) -> Dict[str, object]:
        content = await request.json()
        reason = content.get("reason", "")
        result = patch_mod.reject_patch_proposal(proposal_id, reason=reason, project_root=str(CONFIG.project_root))
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Reject failed"))
        task_engine.append_timeline_event({
            "type": "patch", "title": f"Patch rejected: {proposal_id}",
            "detail": reason or "No reason given",
            "severity": "warning",
            "related_patch_id": proposal_id,
        })
        return result

    # ---- Doctor / Verify / Crash Recovery ----

    @router.get("/api/doctor")
    async def api_doctor() -> Dict[str, object]:
        """Run environment diagnostics (igris doctor)."""
        from igris.core.doctor import run_doctor
        import os as _os
        report = run_doctor(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )
        task_engine.append_timeline_event({
            "type": "doctor", "title": f"Doctor run: {report._compute_overall()}",
            "detail": f"{len(report.checks)} checks, overall={report._compute_overall()}",
            "severity": "info" if report._compute_overall() == "ok" else "warning",
        })
        return report.to_dict()

    @router.get("/api/doctor/markdown")
    async def api_doctor_markdown() -> JSONResponse:
        """Run doctor and return Markdown report."""
        from igris.core.doctor import run_doctor
        import os as _os
        report = run_doctor(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )
        return JSONResponse(content={"markdown": report.to_markdown()})

    @router.get("/api/verify")
    async def api_verify() -> Dict[str, object]:
        """Quick installation verification."""
        from igris.core.doctor import run_verify
        result = run_verify(project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "verify", "title": f"Verify: {'PASS' if result['ok'] else 'FAIL'}",
            "detail": json.dumps({k: v for k, v in result["checks"].items()}, default=str),
            "severity": "info" if result["ok"] else "warning",
        })
        return result

    @router.get("/api/config/validate")
    async def api_config_validate() -> Dict[str, object]:
        """Validate configuration (.env, config.json, providers, budget, safety)."""
        from igris.core.config_validator import validate_all
        result = validate_all(project_root=str(CONFIG.project_root))
        return result.to_dict()

    @router.get("/api/crash-reports")
    async def api_crash_reports(limit: int = 20) -> Dict[str, object]:
        """List recent crash reports."""
        from igris.core.crash_recovery import list_crash_reports
        reports = list_crash_reports(project_root=str(CONFIG.project_root), limit=limit)
        return {"reports": reports, "count": len(reports)}

    @router.get("/api/crash-reports/last-good-state")
    async def api_last_good_state() -> Dict[str, object]:
        """Get the last known good state."""
        from igris.core.crash_recovery import load_good_state
        state = load_good_state(project_root=str(CONFIG.project_root))
        return {"state": state, "available": state is not None}

    @router.post("/api/crash-reports/save-good-state")
    async def api_save_good_state(request: Request) -> Dict[str, object]:
        """Persist the current state as last known good."""
        from igris.core.crash_recovery import save_good_state
        content = await request.json()
        state = content.get("state", {})
        if not state:
            raise HTTPException(status_code=400, detail="State payload required")
        save_good_state(state, project_root=str(CONFIG.project_root))
        task_engine.append_timeline_event({
            "type": "recovery",
            "title": "Good state saved",
            "detail": f"Keys: {', '.join(state.keys())}",
            "severity": "info",
        })
        return {"saved": True}

    @router.get("/api/crash-reports/{crash_id}")
    async def api_crash_report_detail(crash_id: str) -> Dict[str, object]:
        """Get a specific crash report."""
        from igris.core.crash_recovery import get_crash_report
        report = get_crash_report(crash_id, project_root=str(CONFIG.project_root))
        if not report:
            raise HTTPException(status_code=404, detail=f"Crash report {crash_id} not found")
        return report


    _WORK_SESSIONS: Dict[str, object] = {}

    @router.post("/api/work-session/start")
    async def api_work_session_start(request: Request) -> Dict[str, str]:
        from igris.core.work_session import WorkSession
        content = await request.json()
        goal = content.get("goal", "")
        if not goal:
            raise HTTPException(status_code=400, detail="goal required")
        session = WorkSession.create(goal=goal, mission_id=content.get("mission_id"))
        _WORK_SESSIONS[session.session_id] = session
        return {"session_id": session.session_id}

    @router.get("/api/work-session/{session_id}")
    async def api_work_session_get(session_id: str) -> Dict[str, object]:
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        return session.to_dict()

    @router.post("/api/work-session/{session_id}/advance")
    async def api_work_session_advance(session_id: str, request: Request) -> Dict[str, object]:
        from igris.core.work_session import WorkPhase
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        content = await request.json()
        phase = WorkPhase(content.get("phase", "understand"))
        session.advance_phase(phase=phase, outcome=content.get("outcome", "success"), notes=content.get("notes", ""))
        return session.to_dict()

    @router.post("/api/work-session/{session_id}/deliver")
    async def api_work_session_deliver(session_id: str, request: Request) -> Dict[str, object]:
        from igris.core.work_session import DeliveryReport
        session = _WORK_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="WorkSession not found")
        content = await request.json()
        report = DeliveryReport(**content)
        session.complete_deliver(report)
        session.remember(project_root=str(CONFIG.project_root))
        return {"status": "delivered", "delivery_report": report.__dict__}

    # ---- Mission Controller (Epic #40) ----

    @router.post("/api/controller/missions")
    async def api_controller_create_mission(request: Request) -> Dict[str, object]:
        """Create a controlled mission."""
        from igris.core.mission_controller import MissionController
        ctrl = MissionController(project_root=str(CONFIG.project_root))
        content = await request.json()
        title = content.get("title", "")
        goal = content.get("goal", "")
        if not title or not goal:
            raise HTTPException(status_code=400, detail="title and goal required")
        mission = ctrl.create_mission(
            title=title,
            goal=goal,
            description=content.get("description", ""),
            workspace=content.get("workspace", str(CONFIG.project_root)),
            target_hosts=content.get("target_hosts", []),
            constraints=content.get("constraints", []),
            success_criteria=content.get("success_criteria", []),
            risk_level=content.get("risk_level", "low"),
            rollback_plan=content.get("rollback_plan"),
        )
        task_engine.append_timeline_event({
            "type": "mission", "title": f"Mission created: {title}",
            "detail": goal[:200], "severity": "info",
            "mission_id": mission.id, "trace_id": mission.trace_id,
        })
        return mission.to_dict()


    return router
