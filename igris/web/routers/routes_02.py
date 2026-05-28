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
    """Router module 2/10 — _create_app_impl chunk 2."""
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

    @router.get("/api/chat/actions")
    async def api_chat_actions() -> Dict[str, object]:
        from igris.core.chat_personality import get_all_safe_actions
        return {"actions": get_all_safe_actions()}

    @router.get("/api/chat/actions/{intent_name}")
    async def api_chat_actions_by_intent(intent_name: str) -> Dict[str, object]:
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions(intent_name)
        if not actions:
            raise HTTPException(status_code=404, detail=f"No actions for intent: {intent_name}")
        return {"intent": intent_name, "actions": actions}

    # ---- Git ----

    @router.get("/api/git/status", response_model=GitStatusResponse)
    async def api_git_status() -> GitStatusResponse:
        info = get_git_info()
        return GitStatusResponse(
            branch=info.branch, remote=info.remote,
            dirty=info.dirty, changed=info.changed, head=info.head,
        )

    @router.get("/api/git/diff")
    async def api_git_diff(staged: bool = False) -> Dict[str, object]:
        return git_ops.get_diff(staged=staged)

    @router.get("/api/git/diff/stat")
    async def api_git_diff_stat() -> Dict[str, object]:
        return git_ops.get_diff_stat()

    @router.get("/api/git/branches")
    async def api_git_branches() -> Dict[str, object]:
        return git_ops.list_branches()

    @router.post("/api/git/branch")
    async def api_git_create_branch(request: Request) -> Dict[str, object]:
        content = await request.json()
        name = content.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="Branch name required")
        result = git_ops.create_branch(name)
        if result.get("success"):
            task_engine.append_timeline_event({
                "type": "git", "title": f"Branch created: {result.get('branch')}",
                "detail": "", "severity": "info",
            })
        return result

    @router.post("/api/git/commit-proposal")
    async def api_git_commit_proposal(request: Request) -> Dict[str, object]:
        content = await request.json()
        message = content.get("message", "")
        files = content.get("files")
        if not message:
            raise HTTPException(status_code=400, detail="Commit message required")
        proposal = git_ops.create_commit_proposal(message, files)
        return {
            "message": proposal.message,
            "files": proposal.files,
            "safe": proposal.safe,
            "warnings": proposal.warnings,
            "blocked_files": proposal.blocked_files,
            "secret_files": proposal.secret_files,
            "runtime_artifacts": proposal.runtime_artifacts,
        }

    @router.get("/api/git/safety-check")
    async def api_git_safety_check() -> Dict[str, object]:
        return git_ops.pre_commit_safety_check()

    @router.get("/api/git/pr-summary")
    async def api_git_pr_summary(base: str = "main") -> Dict[str, object]:
        return git_ops.generate_pr_summary(base_branch=base)

    # ---- GitHub Workflow (gated) ----

    from igris.layers.git_layer import github_workflow as gh_wf

    @router.post("/api/git/commit")
    async def api_git_commit_gated(request: Request) -> Dict[str, object]:
        content = await request.json()
        message = content.get("message", "")
        approval = content.get("approval", "")
        if not message:
            raise HTTPException(status_code=400, detail="Commit message required")
        result = gh_wf.gated_commit(message=message, approval=approval)
        task_engine.append_timeline_event({
            "type": "git",
            "title": f"Gated commit: {'OK' if result.success else 'blocked'}",
            "detail": safety.redact_secrets(result.message if result.success else result.error),
            "severity": "info" if result.success else "warning",
        })
        return result.to_dict()

    @router.post("/api/github/pr/prepare")
    async def api_github_pr_prepare(request: Request) -> Dict[str, object]:
        content = await request.json()
        base = content.get("base", "main")
        title = content.get("title")
        extra = content.get("extra_context")
        prep = gh_wf.prepare_pr(base_branch=base, title=title, extra_context=extra)
        return prep.to_dict()

    @router.post("/api/github/pr/create")
    async def api_github_pr_create(request: Request) -> Dict[str, object]:
        content = await request.json()
        title = content.get("title", "")
        body = content.get("body", "")
        base = content.get("base", "main")
        approval = content.get("approval", "")
        if not title:
            raise HTTPException(status_code=400, detail="PR title required")
        result = gh_wf.gated_create_pr(
            title=title, body=body, base=base, approval=approval,
        )
        task_engine.append_timeline_event({
            "type": "github",
            "title": f"PR create: {'OK (gated)' if result.success else 'blocked'}",
            "detail": safety.redact_secrets(result.error if result.error else f"PR #{result.pr_number}"),
            "severity": "info" if result.success else "warning",
        })
        return result.to_dict()

    @router.get("/api/github/pr/status")
    async def api_github_pr_status() -> Dict[str, object]:
        return gh_wf.get_pr_status()

    # ---- Vast.ai (gated) ----

    from igris.layers.advisory.vastai_manager import _SHARED_MANAGER as vastai_mgr

    @router.get("/api/vastai/config")
    async def api_vastai_config() -> Dict[str, object]:
        return vastai_mgr.get_config()

    @router.get("/api/vastai/status")
    async def api_vastai_status() -> Dict[str, object]:
        return vastai_mgr.get_status()

    @router.post("/api/vastai/estimate")
    async def api_vastai_estimate(request: Request) -> Dict[str, object]:
        content = await request.json()
        model = content.get("model")
        hours = content.get("hours", 1.0)
        return vastai_mgr.estimate_cost(model=model, hours=hours)

    @router.post("/api/vastai/offers/search")
    async def api_vastai_offers_search(request: Request) -> Dict[str, object]:
        content = await request.json()
        model = content.get("model")
        max_cost = content.get("max_cost")
        result = vastai_mgr.search_offers(model=model, max_cost=max_cost)
        return result.to_dict()

    @router.post("/api/vastai/provision")
    async def api_vastai_provision(request: Request) -> Dict[str, object]:
        content = await request.json()
        approval = content.get("approval", "")
        model = content.get("model")
        offer_id = content.get("offer_id")
        result = vastai_mgr.provision(
            approval=approval, model=model, offer_id=offer_id,
        )
        task_engine.append_timeline_event({
            "type": "vastai",
            "title": f"Provision: {'OK (mock)' if result.get('success') else 'blocked'}",
            "detail": result.get("note", result.get("error", "")),
            "severity": "info" if result.get("success") else "warning",
        })
        return result

    @router.post("/api/vastai/destroy")
    async def api_vastai_destroy(request: Request) -> Dict[str, object]:
        content = await request.json()
        approval = content.get("approval", "")
        result = vastai_mgr.destroy(approval=approval)
        task_engine.append_timeline_event({
            "type": "vastai",
            "title": f"Destroy: {'OK (mock)' if result.get('success') else 'blocked'}",
            "detail": result.get("note", result.get("error", "")),
            "severity": "info" if result.get("success") else "warning",
        })
        return result

    @router.post("/api/vastai/set-mode")
    async def api_vastai_set_mode(request: Request) -> Dict[str, object]:
        content = await request.json()
        mode = content.get("mode", "")
        approval = content.get("approval", "")
        if not mode:
            raise HTTPException(status_code=400, detail="Mode required")
        return vastai_mgr.set_mode(mode=mode, approval=approval)

    # ---- Fleet API ----

    from igris.layers.advisory.vastai_fleet import _SHARED_FLEET

    @router.get("/api/fleet/status")
    async def api_fleet_status() -> Dict[str, object]:
        """Fleet-wide status: all instances, queue, costs."""
        return _SHARED_FLEET.fleet_status()

    @router.post("/api/fleet/provision")
    async def api_fleet_provision(request: Request) -> Dict[str, object]:
        """Manually trigger provisioning of N new fleet instances."""
        body = await request.json()
        approval = body.get("approval", "")
        count = int(body.get("count", 1))
        if approval != "I_APPROVE_VASTAI_COSTS":
            raise HTTPException(status_code=403, detail="approval required: I_APPROVE_VASTAI_COSTS")
        if count < 1 or count > 5:
            raise HTTPException(status_code=400, detail="count must be 1-5")
        new_instances = _SHARED_FLEET._provision_instances(count)
        return {"provisioned": len(new_instances), "fleet": _SHARED_FLEET.fleet_status()}

    @router.post("/api/fleet/release/{instance_id}")
    async def api_fleet_release(instance_id: str, request: Request) -> Dict[str, object]:
        """Manually release a fleet instance back to idle."""
        body = await request.json()
        outcome = body.get("outcome", "manual_release")
        _SHARED_FLEET.release(instance_id, outcome=outcome)
        return {"released": instance_id, "fleet": _SHARED_FLEET.fleet_status()}

    @router.get("/api/fleet/queue")
    async def api_fleet_queue() -> Dict[str, object]:
        """Current task queue waiting for GPU instances."""
        status = _SHARED_FLEET.fleet_status()
        return {"queue_depth": status["queue_depth"], "queue": status.get("queue", [])}

    @router.get("/api/fleet/worktrees")
    async def api_fleet_worktrees() -> Dict[str, object]:
        """Active git worktrees managed by WorktreeManager."""
        status = _SHARED_FLEET.fleet_status()
        return {"worktrees": status.get("worktrees", [])}

    @router.get("/api/fleet/locks")
    async def api_fleet_locks() -> Dict[str, object]:
        """Current file lock registry — which issue holds which paths."""
        status = _SHARED_FLEET.fleet_status()
        return {"file_locks": status.get("file_locks", {})}

    # ---- Routing / Cost ----

    @router.get("/api/routing/history")
    async def api_routing_history() -> Dict[str, object]:
        return {"history": provider_router.get_history()}

    @router.get("/api/cost/summary")
    async def api_cost_summary() -> Dict[str, object]:
        return provider_router.cost_summary()

    @router.get("/api/routing/explain")
    async def api_routing_explain() -> Dict[str, str]:
        return {"explanation": provider_router.explain_routing()}

    @router.get("/api/routing/availability")
    async def api_routing_availability() -> Dict[str, object]:
        return provider_router.check_availability()

    @router.post("/api/routing/estimate")
    async def api_routing_estimate(body: Dict[str, object] = Body(default={})) -> Dict[str, object]:
        task_type = str(body.get("task_type", "chat")) if isinstance(body, dict) else "chat"
        complexity = str(body.get("complexity", "low")) if isinstance(body, dict) else "low"
        return provider_router.estimate_route(task_type=task_type, complexity=complexity)

    @router.get("/api/cost/budget")
    async def api_cost_budget() -> Dict[str, object]:
        return provider_router.get_budget_status()

    @router.post("/api/cost/budget")
    async def api_cost_budget_update(body: Dict[str, object] = Body(...)) -> Dict[str, object]:
        max_cost = body.get("max_session_cost")
        warn = body.get("warn_threshold")
        return provider_router.set_budget_config(
            max_session_cost=float(max_cost) if max_cost is not None else None,
            warn_threshold=float(warn) if warn is not None else None,
        )

    # ---- Files ----

    @router.get("/api/files/tree")
    async def api_files_tree() -> Dict[str, object]:
        root = CONFIG.project_root
        tree = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            filtered_dirs: List[str] = []
            for d in list(dirnames):
                sub_path = Path(dirpath) / d
                if safety.is_runtime_artifact(sub_path):
                    dirnames.remove(d)
                    continue
                if d.startswith('.'):
                    dirnames.remove(d)
                    continue
                filtered_dirs.append(d)
            entries = []
            for d in sorted(filtered_dirs):
                entries.append({"type": "dir", "name": d})
            for f in sorted(filenames):
                if f.startswith('.'):
                    continue
                if safety.is_sensitive_filename(f):
                    continue
                sub = Path(dirpath) / f
                if safety.is_runtime_artifact(sub):
                    continue
                entries.append({"type": "file", "name": f})
            tree.append({"path": rel_dir, "entries": entries})
        return {"tree": tree}


    return router
