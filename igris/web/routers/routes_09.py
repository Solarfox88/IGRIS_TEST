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
    """Router module 9/10 — _create_app_impl chunk 9."""
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

    @router.get("/api/governor/saturated")
    async def api_governor_saturated() -> Dict[str, object]:
        """Get saturated families."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return {
            "saturated": gov.get_saturated_families(),
            "counts": gov.get_family_counts(),
        }

    @router.post("/api/governor/block-family")
    async def api_governor_block(request: Request) -> Dict[str, object]:
        """Block a family from future selection."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        decision = gov.block_family(
            family=content.get("family", ""),
            reason=content.get("reason", ""),
        )
        gov.save_state()
        return decision.to_dict()

    @router.post("/api/governor/materialize-alternative")
    async def api_governor_materialize(request: Request) -> Dict[str, object]:
        """Materialize an alternative task."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        decision = gov.materialize_alternative(
            original_family=content.get("family", ""),
            mission_id=content.get("mission_id", ""),
            trace_id=content.get("trace_id", ""),
        )
        return decision.to_dict()

    @router.get("/api/governor/escalation-report")
    async def api_governor_escalation(trace_id: str = "") -> Dict[str, object]:
        """Generate escalation report."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        return gov.generate_escalation_report(trace_id=trace_id)

    @router.post("/api/governor/record-task")
    async def api_governor_record(request: Request) -> Dict[str, object]:
        """Record a task execution."""
        from igris.core.teacher_governor import TeacherGovernor
        gov = TeacherGovernor(project_root=str(CONFIG.project_root))
        gov.load_state()
        content = await request.json()
        gov.record_task(
            description=content.get("description", ""),
            family=content.get("family", ""),
        )
        gov.save_state()
        return {"recorded": True, "history_length": len(gov.get_history())}

    # ------------------------------------------------------------------
    # Agent Action Schema / Prompt Contract / Model Orchestrator — Epic #58
    # ------------------------------------------------------------------

    @router.get("/api/agent/schema")
    async def api_agent_schema() -> Dict[str, object]:
        """Return the Agent Action JSON schema."""
        from igris.core.agent_action_schema import ACTION_JSON_SCHEMA
        return {"schema": ACTION_JSON_SCHEMA}

    @router.get("/api/agent/roles")
    async def api_agent_roles() -> Dict[str, object]:
        """List all registered agent roles."""
        from igris.core.agent_action_schema import list_registry
        return {"roles": list_registry()}

    @router.get("/api/agent/action-types")
    async def api_agent_action_types() -> Dict[str, object]:
        """List all available action types."""
        from igris.core.agent_action_schema import (
            ACTION_TYPES, ACTION_ROUTING,
            READ_ONLY_ACTIONS, WRITE_ACTIONS, RISK_GATED_ACTIONS,
        )
        return {
            "action_types": list(ACTION_TYPES),
            "routing": dict(ACTION_ROUTING),
            "read_only": sorted(READ_ONLY_ACTIONS),
            "write": sorted(WRITE_ACTIONS),
            "risk_gated": sorted(RISK_GATED_ACTIONS),
        }

    @router.get("/api/agent/examples")
    async def api_agent_examples() -> Dict[str, object]:
        """Return example scenarios for the action schema."""
        from igris.core.prompt_contract import get_example_scenarios
        return {"examples": get_example_scenarios()}

    @router.post("/api/agent/validate")
    async def api_agent_validate(request: Request) -> Dict[str, object]:
        """Validate an action against the schema."""
        from igris.core.agent_action_schema import AgentAction, validate_action
        content = await request.json()
        action = AgentAction.from_dict(content)
        result = validate_action(action)
        return result.to_dict()

    @router.post("/api/agent/parse")
    async def api_agent_parse(request: Request) -> Dict[str, object]:
        """Parse raw LLM output into a validated action."""
        from igris.core.agent_action_schema import parse_llm_action
        content = await request.json()
        raw = content.get("raw_output", "")
        action, issues = parse_llm_action(raw)
        return {
            "parsed": action.to_dict() if action else None,
            "issues": issues,
            "valid": action is not None,
        }

    @router.get("/api/orchestrator/providers")
    async def api_orchestrator_providers() -> Dict[str, object]:
        """List configured LLM providers (no secrets)."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return {"providers": orch.list_providers()}

    @router.get("/api/orchestrator/profiles")
    async def api_orchestrator_profiles() -> Dict[str, object]:
        """List task type to profile mappings."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return {"profiles": orch.get_profiles()}

    @router.get("/api/orchestrator/cost")
    async def api_orchestrator_cost() -> Dict[str, object]:
        """Get cost tracking summary."""
        from igris.core.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        return orch.get_cost_summary()

    @router.get("/api/agent/prompt-contract")
    async def api_agent_prompt_contract(role: str = "coder") -> Dict[str, object]:
        """Get the reasoning loop prompt contract for a role."""
        from igris.core.prompt_contract import build_reasoning_prompt
        prompt = build_reasoning_prompt(
            role=role,
            mission_context="Example: Add /api/ping endpoint with tests",
            state_context="repo_clean: true, tests_pass: true",
            recent_actions="No recent actions.",
            file_context="No files loaded.",
        )
        return {"role": role, "prompt": prompt}

    # ------------------------------------------------------------------
    # Code Navigation Tools — Epic #59
    # ------------------------------------------------------------------

    @router.post("/api/nav/search-code")
    async def api_nav_search_code(request: Request) -> Dict[str, object]:
        """Search for patterns in code files."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.search_code(
            pattern=content.get("pattern", ""),
            path=content.get("path"),
            max_results=content.get("max_results", 50),
            context_lines=content.get("context_lines", 0),
        )
        return result.to_dict()

    @router.post("/api/nav/find-files")
    async def api_nav_find_files(request: Request) -> Dict[str, object]:
        """Find files by name/glob pattern."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.find_files(
            pattern=content.get("pattern", ""),
            max_results=content.get("max_results", 100),
        )
        return result.to_dict()

    @router.post("/api/nav/list-directory")
    async def api_nav_list_directory(request: Request) -> Dict[str, object]:
        """List directory contents."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.list_directory(
            path=content.get("path", "."),
            depth=content.get("depth", 1),
            max_entries=content.get("max_entries", 200),
        )
        return result.to_dict()

    @router.post("/api/nav/read-file-range")
    async def api_nav_read_file_range(request: Request) -> Dict[str, object]:
        """Read specific lines from a file."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.read_file_range(
            path=content.get("path", ""),
            start=content.get("start", 1),
            end=content.get("end"),
            max_lines=content.get("max_lines", 500),
        )
        return result.to_dict()

    @router.get("/api/nav/repo-map")
    async def api_nav_repo_map() -> Dict[str, object]:
        """Build a lightweight repository map."""
        from igris.core.code_navigation import CodeNavigator
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.repo_map()
        return result.to_dict()

    @router.post("/api/nav/find-symbol")
    async def api_nav_find_symbol(request: Request) -> Dict[str, object]:
        """Find symbol definitions (function, class, variable)."""
        from igris.core.code_navigation import CodeNavigator
        content = await request.json()
        nav = CodeNavigator(project_root=str(CONFIG.project_root))
        result = nav.find_symbol(
            symbol=content.get("symbol", ""),
            path=content.get("path"),
            max_results=content.get("max_results", 50),
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Context Manager — Epic #60
    # ------------------------------------------------------------------

    @router.post("/api/context/build")
    async def api_context_build(request: Request) -> Dict[str, object]:
        """Build a context packet for the reasoning loop."""
        from igris.core.context_manager import ContextManager
        content = await request.json()
        ctx = ContextManager(project_root=str(CONFIG.project_root))
        packet = ctx.build_context(
            goal=content.get("goal", ""),
            role=content.get("role", "coder"),
            profile=content.get("profile", "default"),
            mission_id=content.get("mission_id", ""),
            mission_status=content.get("mission_status", ""),
            world_state=content.get("world_state"),
            recent_actions=content.get("recent_actions"),
            recent_errors=content.get("recent_errors"),
            memory_items=content.get("memory_items"),
            relevant_files=content.get("relevant_files"),
            file_snippets=content.get("file_snippets"),
            keywords=content.get("keywords"),
        )
        return packet.to_dict()

    @router.get("/api/context/budgets")
    async def api_context_budgets() -> Dict[str, object]:
        """Get token budget information for all profiles."""
        from igris.core.context_manager import ContextManager, TOKEN_BUDGETS
        ctx = ContextManager(project_root=str(CONFIG.project_root))
        return {
            profile: ctx.get_budget_info(profile)
            for profile in TOKEN_BUDGETS
        }

    @router.post("/api/context/score-files")
    async def api_context_score_files(request: Request) -> Dict[str, object]:
        """Score file relevance for a given task."""
        from igris.core.context_manager import score_file_relevance
        content = await request.json()
        files = content.get("files", [])
        keywords = content.get("keywords", [])
        recent_files = content.get("recent_files", [])
        error_files = content.get("error_files", [])
        scored = []
        for f in files:
            s = score_file_relevance(f, keywords, recent_files, error_files)
            scored.append({"path": f, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"files": scored}

    # ------------------------------------------------------------------
    # Agent Reasoning Loop — Epic #61
    # ------------------------------------------------------------------

    @router.post("/api/reasoning/run")
    async def api_reasoning_run(request: Request) -> Dict[str, object]:
        """Run the agent reasoning loop for a goal."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        content = await request.json()

        # Validate and normalise initial_context
        raw_ctx = content.get("initial_context")
        if raw_ctx is not None and not isinstance(raw_ctx, dict):
            if isinstance(raw_ctx, str):
                raw_ctx = {"note": raw_ctx}
            else:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "initial_context must be a dict or string",
                        "received_type": type(raw_ctx).__name__,
                    },
                )

        loop = AgentReasoningLoop(
            project_root=str(CONFIG.project_root),
            max_steps=content.get("max_steps", 50),
            max_consecutive_errors=content.get("max_consecutive_errors", 5),
            role=content.get("role", "coder"),
        )
        result = loop.run(
            goal=content.get("goal", ""),
            mission_id=content.get("mission_id", ""),
            initial_context=raw_ctx,
        )
        return result.to_dict()


    return router
