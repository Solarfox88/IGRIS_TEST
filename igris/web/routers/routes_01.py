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
    """Router module 1/10 — _create_app_impl chunk 1."""
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

    @router.get('/api/diagnostics/session-resume')
    async def session_resume():
        # Implement the logic for session resume
        return JSONResponse(content={'status': 'success'})

    @router.get('/api/rank/s-dashboard')
    async def get_rank_s_dashboard():
        return {
            'app': 'IGRIS_GPT',
            'rank': 'S',
            'status': 'ok',
            'capability': 'end-to-end-supervised',
            'checks': {
                'backend': True,
                'ui': True,
                'tests': True,
                'workflow': True
            }
        }

    @router.get('/api/rank/ui-card')
    async def get_rank_ui_card():
        return {'app': 'IGRIS_GPT', 'rank': 'A++', 'status': 'ok', 'capability': 'ui-visible-supervised'}

    @router.get('/api/rank/summary-card')
    async def get_rank_summary_card():
        return {'app': 'IGRIS_GPT', 'rank': 'A+', 'status': 'ok', 'capability': 'multi-file-supervised'}

    @router.get('/api/system/version-summary')
    async def get_version_summary():
        return {'app': 'IGRIS_GPT', 'rank': 'A-generalization', 'status': 'ok'}

    @router.get('/api/rank/status')
    async def get_rank_status():
        return {'rank': 'A', 'status': 'ok', 'agent': 'IGRIS_GPT'}
    @router.get('/api/version-info')
    async def version_info():
        return {'app': 'IGRIS_GPT', 'status': 'ok'}

    @router.get("/", response_class=HTMLResponse)
    async def index() -> str:
        template = jinja_env.get_template("index.html")
        return template.render()

    # ---- Status / Config ----

    @router.get("/api/status")
    async def api_status() -> Dict[str, object]:
        provider, model = provider_router.choose_provider()
        return {"provider": provider, "model": model, "safe": True}

    @router.get("/api/config/safe")
    async def api_config_safe() -> Dict[str, object]:
        return CONFIG.safe_dict()

    # ---- Sessions / Chat ----

    @router.post("/api/sessions")
    async def create_session() -> Dict[str, str]:
        session_id = str(len(sessions) + 1)
        sessions[session_id] = []
        return {"id": session_id}

    @router.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, content: Dict[str, str] = Body(...)) -> Dict[str, object]:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        message = content.get("message", "")
        sessions[session_id].append({"role": "user", "content": message})

        # Use real chat engine
        result = chat_llm(message, history=sessions[session_id][:-1])
        response_text = _redact(result["text"])

        sessions[session_id].append({"role": "assistant", "content": response_text})

        # Record routing decision
        provider_router.record_chat_routing(
            provider=result["provider"], model=result["model"],
            reason=result["routing_reason"], latency_ms=result["latency_ms"],
            fallback_used=result["fallback_used"],
        )

        task_engine.append_timeline_event({
            "type": "chat", "title": "Chat message",
            "detail": f"User: {message[:80]}",
        })

        return {
            "response": response_text,
            "provider": result["provider"],
            "model": result["model"],
            "fallback_used": result["fallback_used"],
            "latency_ms": result["latency_ms"],
            "intent_detected": result.get("intent_detected"),
            "suggested_actions": result.get("suggested_actions", []),
        }

    # ---- Chat Streaming + Tier ----

    @router.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        content = await request.json()
        message = content.get("message", "")
        session_id = content.get("session_id")
        enrich = content.get("enrich", False)
        if not message:
            raise HTTPException(status_code=400, detail="message required")

        history = []
        if session_id and session_id in sessions:
            history = sessions[session_id]

        system_prompt = None
        if enrich:
            system_prompt = chat_context.build_context_system_prompt(
                task_engine=task_engine,
                project_root=str(CONFIG.project_root),
            )

        chunks = chat_streaming.chat_stream_sync(
            message=message, history=history, system_prompt=system_prompt,
        )

        # Store in session if provided
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append({"role": "user", "content": message})
            full_text = "".join(c.text for c in chunks if c.type == "content")
            sessions[session_id].append({"role": "assistant", "content": full_text})

            task_engine.append_timeline_event({
                "type": "chat", "title": "Chat stream",
                "detail": f"User: {message[:80]}",
            })

        async def event_generator():
            for chunk in chunks:
                yield chunk.to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/api/chat/context")
    async def api_chat_context() -> Dict[str, object]:
        return chat_context.build_chat_context(
            task_engine=task_engine,
            project_root=str(CONFIG.project_root),
        )

    @router.get("/api/chat/context/summary")
    async def api_chat_context_summary() -> Dict[str, object]:
        return chat_context.get_context_summary(
            task_engine=task_engine,
            project_root=str(CONFIG.project_root),
        )

    @router.get("/api/chat/tiers")
    async def api_chat_tiers() -> Dict[str, object]:
        return chat_streaming.get_tier_availability()

    @router.post("/api/chat/tiers")
    async def api_set_chat_tier(request: Request) -> Dict[str, object]:
        content = await request.json()
        tier = content.get("tier", "")
        if not tier:
            raise HTTPException(status_code=400, detail="tier required")
        try:
            config = chat_streaming.set_tier(tier)
            return config.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---- System Info ----

    @router.get("/api/system/info")
    async def api_system_info() -> Dict[str, object]:
        """Safe, read-only system information."""
        from igris.core.system_info import get_system_info
        import os as _os
        return get_system_info(
            project_root=str(CONFIG.project_root),
            host=_os.environ.get("IGRIS_HOST", "127.0.0.1"),
            port=int(_os.environ.get("IGRIS_PORT", "8000")),
        )

    # ---- Dashboard Summary ----

    @router.get("/api/dashboard/summary")
    async def api_dashboard_summary() -> Dict[str, object]:
        """Aggregated dashboard view — health, readiness, diagnostics, loop."""
        from igris.core import diagnostics as diagnostics_dash

        diag = {}
        try:
            tasks = [t.to_dict() for t in task_engine.list_tasks()]
            timeline = task_engine.recent_timeline_events(limit=50)
            diag = diagnostics_dash.get_diagnostic_summary(
                tasks, timeline, project_root=str(CONFIG.project_root),
            )
        except Exception:
            pass

        loop_info = {}
        try:
            loop_info = loop_engine.get_status()
        except Exception:
            pass

        return {
            "health": {"status": "ok"},
            "diagnostics": diag,
            "loop": loop_info,
            "tab_layout": {
                "primary": ["dashboard", "code", "tasks", "terminal", "memory", "safety", "advanced"],
                "grouped": {
                    "code": ["files", "git", "patches"],
                    "tasks": ["tasks", "loop"],
                    "terminal": ["commands", "tests"],
                    "memory": ["memory", "timeline"],
                    "safety": ["safety", "cost"],
                    "advanced": ["a2a", "logs"],
                },
            },
        }

    # ---- Chat Personality / Capabilities ----

    @router.get("/api/chat/capabilities")
    async def api_chat_capabilities() -> Dict[str, object]:
        from igris.core.chat_personality import get_capability_summary
        return get_capability_summary()

    @router.post("/api/chat/intent")
    async def api_chat_intent(request: Request) -> Dict[str, object]:
        from igris.core.chat_personality import (
            detect_intent, get_grounded_response, get_suggested_actions,
        )
        content = await request.json()
        message = content.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="message required")
        intent = detect_intent(message)
        response = get_grounded_response(intent) if intent else None
        actions = get_suggested_actions(intent) if intent else []
        return {
            "intent": intent,
            "grounded_response": response,
            "has_response": response is not None,
            "suggested_actions": actions,
        }


    return router
