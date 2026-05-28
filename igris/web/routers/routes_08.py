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
    """Router module 8/10 — _create_app_impl chunk 8."""
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

    @router.get("/api/tools")
    async def api_tools_list() -> Dict[str, object]:
        """List available tool families."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return {"tools": rt.list_tools()}

    @router.post("/api/tools/shell/execute")
    async def api_tools_shell(request: Request) -> Dict[str, object]:
        """Execute a governed shell command."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.shell_execute(
            command_id=content.get("command_id", ""),
            args=content.get("args"),
            cwd=content.get("cwd"),
            timeout=content.get("timeout", 30),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.post("/api/tools/fs/read")
    async def api_tools_fs_read(request: Request) -> Dict[str, object]:
        """Read a file safely."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_read(
            path=content.get("path", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.post("/api/tools/fs/write")
    async def api_tools_fs_write(request: Request) -> Dict[str, object]:
        """Write to a file safely."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_write(
            path=content.get("path", ""),
            content=content.get("content", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.post("/api/tools/fs/diff")
    async def api_tools_fs_diff(request: Request) -> Dict[str, object]:
        """Preview diff for a file."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.fs_diff(
            path=content.get("path", ""),
            new_content=content.get("new_content", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.get("/api/tools/git/status")
    async def api_tools_git_status(mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git status."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_status(mission_id=mission_id, trace_id=trace_id).to_dict()

    @router.get("/api/tools/git/diff")
    async def api_tools_git_diff(staged: bool = False, mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git diff."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_diff(staged=staged, mission_id=mission_id, trace_id=trace_id).to_dict()

    @router.get("/api/tools/git/log")
    async def api_tools_git_log(count: int = 10, mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git log."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_log(count=count, mission_id=mission_id, trace_id=trace_id).to_dict()

    @router.get("/api/tools/git/branch")
    async def api_tools_git_branch(mission_id: str = "", trace_id: str = "") -> Dict[str, object]:
        """Git branches."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return rt.git_branch(mission_id=mission_id, trace_id=trace_id).to_dict()

    @router.post("/api/tools/git/commit")
    async def api_tools_git_commit(request: Request) -> Dict[str, object]:
        """Gated git commit."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.git_commit(
            message=content.get("message", ""),
            files=content.get("files"),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.post("/api/tools/docker/ps")
    async def api_tools_docker_ps(request: Request) -> Dict[str, object]:
        """Docker ps."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        return rt.docker_ps(
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        ).to_dict()

    @router.post("/api/tools/http/check")
    async def api_tools_http_check(request: Request) -> Dict[str, object]:
        """HTTP health check."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        result = rt.http_check(
            url=content.get("url", ""),
            timeout=content.get("timeout", 10),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.post("/api/tools/test/run")
    async def api_tools_test_run(request: Request) -> Dict[str, object]:
        """Run tests."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        result = rt.run_tests(
            args=content.get("args"),
            timeout=content.get("timeout", 120),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return result.to_dict()

    @router.get("/api/tools/hosts")
    async def api_tools_hosts() -> Dict[str, object]:
        """List registered SSH hosts."""
        from igris.core.tool_runtime import ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        return {"hosts": rt.list_hosts()}

    @router.post("/api/tools/hosts/register")
    async def api_tools_host_register(request: Request) -> Dict[str, object]:
        """Register an SSH host."""
        from igris.core.tool_runtime import SSHHost, ToolRuntime
        rt = ToolRuntime(project_root=str(CONFIG.project_root))
        content = await request.json()
        host = SSHHost.from_dict(content)
        rt.register_host(host)
        return {"registered": host.to_dict()}

    # ---- GOAP Planner (Epic #43) ----

    @router.get("/api/goap/state")
    async def api_goap_state() -> Dict[str, object]:
        """Get current world state."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        return planner.get_current_state().to_dict()

    @router.post("/api/goap/plan")
    async def api_goap_plan(request: Request) -> Dict[str, object]:
        """Generate a GOAP plan for a goal."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        goal = content.get("goal", {})
        mission_id = content.get("mission_id", "")
        plan = planner.generate_plan(goal=goal, mission_id=mission_id)
        planner.save_plan(plan)
        return plan.to_dict()

    @router.get("/api/goap/plans")
    async def api_goap_plans(mission_id: str = "") -> Dict[str, object]:
        """List GOAP plans."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plans = planner.list_plans(mission_id=mission_id)
        return {"plans": plans, "count": len(plans)}

    @router.get("/api/goap/plans/{plan_id}")
    async def api_goap_plan_get(plan_id: str) -> Dict[str, object]:
        """Get a specific GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return plan.to_dict()

    @router.get("/api/goap/plans/{plan_id}/explain")
    async def api_goap_plan_explain(plan_id: str) -> Dict[str, object]:
        """Explain a GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return planner.explain_plan(plan)

    @router.get("/api/goap/plans/{plan_id}/next")
    async def api_goap_plan_next(plan_id: str) -> Dict[str, object]:
        """Explain next action in a GOAP plan."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return planner.explain_next_action(plan)

    @router.post("/api/goap/eligible-actions")
    async def api_goap_eligible(request: Request) -> Dict[str, object]:
        """Get eligible actions for a state."""
        from igris.core.goap_planner import GOAPPlanner, WorldState
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json() if await request.body() else {}
        state = WorldState.from_dict(content) if content else planner.get_current_state()
        eligible = planner.get_eligible_actions(state)
        return {"actions": [a.to_dict() for a in eligible], "count": len(eligible)}

    @router.post("/api/goap/validate-llm-plan")
    async def api_goap_validate(request: Request) -> Dict[str, object]:
        """Validate LLM-generated plan output."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        plan = planner.validate_llm_plan(content)
        if not plan:
            return {"valid": False, "reason": "Plan does not match required schema"}
        return {"valid": True, "plan": plan.to_dict()}

    @router.post("/api/goap/replan")
    async def api_goap_replan(request: Request) -> Dict[str, object]:
        """Replan after failure."""
        from igris.core.goap_planner import GOAPPlanner
        planner = GOAPPlanner(project_root=str(CONFIG.project_root))
        content = await request.json()
        plan_id = content.get("plan_id", "")
        plan = planner.load_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        new_plan = planner.replan_after_failure(
            plan=plan,
            failed_action_id=content.get("failed_action_id", ""),
            failure_reason=content.get("failure_reason", ""),
        )
        planner.save_plan(new_plan)
        return new_plan.to_dict()

    # ---- Teacher/Governor (Epic #46) ----

    @router.post("/api/governor/evaluate")
    async def api_governor_evaluate(request: Request) -> Dict[str, object]:
        """Evaluate a proposed task against governance rules."""
        from igris.core.teacher_governor import TeacherGovernor, TaskFingerprint
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        fp = None
        if content.get("fingerprint"):
            fp = TaskFingerprint(**content["fingerprint"])
        decision = gov.evaluate_task(
            description=content.get("description", ""),
            family=content.get("family", ""),
            differentiator=content.get("differentiator", ""),
            success_criteria=content.get("success_criteria"),
            fingerprint=fp,
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @router.get("/api/governor/summary")
    async def api_governor_summary() -> Dict[str, object]:
        """Get governor state summary."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return gov.get_summary()


    return router
